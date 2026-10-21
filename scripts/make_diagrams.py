"""
KernelLens v2 diagram generator (Graphviz).

Auto-laid-out, information-dense diagrams for the report and the docs site.
Overwrites the corresponding PNGs produced by make_figures_v2.py.

Run:  python3 scripts/make_diagrams.py
"""
import os
import html
import graphviz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

THEME = os.environ.get("KL_THEME", "light")
if THEME == "dark":
    INK = "#F2EFE6"
    MUTED = "#A8A294"
    LINE = "#3A362E"
    CARD = "#1A1915"
    BG = "transparent"
    BLUE = "#8FBEF5"
    PURPLE = "#C3A6EE"
    GREEN = "#6FD3A6"
    AMBER = "#EDB267"
    RED = "#F2968C"
    TEAL = "#6FD0D0"
    BRAND = "#E0836A"
else:
    INK = "#0F172A"
    MUTED = "#475569"
    LINE = "#CBD5E1"
    CARD = "#FFFFFF"
    BG = "transparent"
    BLUE = "#2563EB"
    PURPLE = "#7C3AED"
    GREEN = "#059669"
    AMBER = "#B45309"
    RED = "#DC2626"
    TEAL = "#0D9488"
    BRAND = "#C06048"

GRAPH_ATTR = dict(rankdir="LR", bgcolor=BG, fontname=FONT,
                  fontsize="11", pad="0.3", nodesep="0.35", ranksep="0.6")


def _html(title, lines, accent, mono=False):
    fn = MONO if mono else FONT
    rows = "".join(
        '<TR><TD ALIGN="LEFT"><FONT FACE="%s" POINT-SIZE="10" COLOR="%s">%s</FONT></TD></TR>'
        % (fn, INK, html.escape(l) if l else "&#160;")
        for l in lines
    )
    return (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5" '
        f'COLOR="{accent}" BGCOLOR="{CARD}">'
        f'<TR><TD BGCOLOR="{accent}"><FONT FACE="{FONT}" POINT-SIZE="11" COLOR="white">'
        f'<B>{html.escape(title)}</B></FONT></TD></TR>{rows}</TABLE>>'
    )


def _render(dot, name):
    if THEME == "dark":
        name = name[:-4] + "_dark.png"
    dot.render(filename=name[:-4], directory=OUT, format="png", cleanup=True)
    print("wrote", os.path.join(OUT, name))


def pipeline():
    d = graphviz.Digraph("pipeline", graph_attr=GRAPH_ATTR)
    d.attr("node", shape="plaintext", fontname=FONT)

    d.node("mod", _html("PyTorch module", [
        "class RMSNorm(nn.Module):",
        "    def forward(x):",
        "        out = torch.empty_like(x)",
        "        rmsnorm_kernel[(B, M)](...)",
        "        return out",
        "",
        "@triton.jit kernel source",
    ], INK, mono=True))
    d.node("inp", _html("Sample input", [
        "x : (4, 512, 4096) fp32",
        "",
        "profiled once on the GPU",
    ], MUTED))

    with d.subgraph(name="cluster_p1") as c:
        c.attr(label="kl.compile  (1/4)  trace", style="rounded", color=BLUE,
               fontcolor=BLUE, fontname=FONT)
        c.node("p1", _html("Dual-pass tracing + AST scan", [
            "eager pass: intercept the compiled PTX,",
            "  shared-mem bytes, num_warps, real grid",
            "symbolic pass: recover the launch grid",
            "  as an expression over runtime shapes",
            "AST scan of tl.load / tl.store (nested",
            "  helpers included): input / output / in-place",
        ], BLUE, mono=True))

    d.node("manifest", _html("KernelManifest", [
        "ptx, shared_memory, num_warps",
        "arguments: name, shape, strides, dtype, kind",
        "grid_cxx_exprs (dynamic launch ABI)",
    ], TEAL, mono=True))

    with d.subgraph(name="cluster_p2") as c:
        c.attr(label="kl.compile  (2/4)  rewrite", style="rounded", color=PURPLE,
               fontcolor=PURPLE, fontname=FONT)
        c.node("p2", _html("torch.onnx.export + injection", [
            "Triton calls -> triton_custom::<kernel> nodes",
            "per-call scalars -> CPU inputs",
            "in-place buffers -> aliased input/output",
        ], PURPLE, mono=True))

    d.node("onnx", _html("<name>.onnx", [
        "standard ONNX graph",
        "+ triton_custom nodes",
    ], PURPLE, mono=True))

    with d.subgraph(name="cluster_p3") as c:
        c.attr(label="kl.compile  (3/4)  synthesize", style="rounded", color=GREEN,
               fontcolor=GREEN, fontname=FONT)
        c.node("p3o", _html("ONNX Runtime plugin", [
            "OrtCustomOp class",
            "Compute() -> cuLaunchKernel",
            "PTX entry-slot argument array",
        ], GREEN, mono=True))
        c.node("p3t", _html("TensorRT plugin", [
            "IPluginV2DynamicExt + IPluginCreator",
            "getOutputDimensions / supportsFormatCombination",
            "enqueue(): grid evaluated in C++",
        ], GREEN, mono=True))

    with d.subgraph(name="cluster_p4") as c:
        c.attr(label="kl.compile  (4/4)  build", style="rounded", color=AMBER,
               fontcolor=AMBER, fontname=FONT)
        c.node("p4", _html("nvcc + g++", [
            "compile the generated sources",
            "PTX ISA version clamping",
            "16-byte alignment guard",
            "build the TRT engine (lazy)",
        ], AMBER, mono=True))

    d.node("so", _html("Artifacts", [
        "libtriton_ort_plugins.so",
        "libtriton_trt_plugins.so",
        "<name>.engine",
    ], GREEN, mono=True))

    with d.subgraph(name="cluster_run") as c:
        c.attr(label="compiled_model.run(inputs, backend=...)", style="rounded",
               color=AMBER, fontcolor=AMBER, fontname=FONT)
        c.node("rort", _html("Run in ONNX Runtime", [
            "create Ort::Session, load the .so",
            "IOBinding on raw CUDA pointers",
            "RunWithBindings",
        ], BLUE, mono=True))
        c.node("rtrt", _html("Run in TensorRT", [
            "deserialize -> IExecutionContext",
            "IExecutionContext::setTensorAddress",
            "Enqueue Execution",
        ], PURPLE, mono=True))
    d.node("out", _html("torch tensors", [
        "on the same CUDA stream",
        "no host staging copies",
    ], INK))

    d.edge("mod", "p1", label="forward()", fontcolor=MUTED, color=INK, fontname=FONT, fontsize="10")
    d.edge("inp", "p1", label="shapes", fontcolor=MUTED, color=INK, fontname=FONT, fontsize="10")
    d.edge("p1", "manifest", label="analysis", color=BLUE, fontcolor=BLUE, fontname=FONT, fontsize="10")
    d.edge("manifest", "p2", label="manifest", color=TEAL, fontcolor=TEAL, fontname=FONT, fontsize="10")
    d.edge("manifest", "p3o", label="manifest", color=TEAL, fontcolor=TEAL, fontname=FONT, fontsize="10")
    d.edge("manifest", "p3t", label="manifest", color=TEAL, fontcolor=TEAL, fontname=FONT, fontsize="10")
    d.edge("p2", "onnx", color=PURPLE)
    d.edge("onnx", "rort", label=".onnx", color=PURPLE, fontcolor=PURPLE, fontname=FONT, fontsize="10")
    d.edge("p3o", "p4", style="invis")
    d.edge("p3t", "p4", style="invis")
    d.edge("p3o", "so", color=GREEN)
    d.edge("p3t", "so", color=GREEN)
    d.edge("so", "rort", label=".so", color=GREEN, fontcolor=GREEN, fontname=FONT, fontsize="10")
    d.edge("so", "rtrt", label=".so / .engine", color=GREEN, fontcolor=GREEN, fontname=FONT, fontsize="10")
    d.edge("rort", "out", color=BLUE)
    d.edge("rtrt", "out", color=PURPLE)
    _render(d, "pipeline_end_to_end.png")


def anatomy():
    d = graphviz.Digraph("anatomy", graph_attr=dict(rankdir="LR", bgcolor="white",
                                                    fontname=FONT, pad="0.3",
                                                    nodesep="0.3", ranksep="0.7"))
    d.attr("node", shape="plaintext")
    code = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" COLOR="#CBD5E1">'
        '<TR><TD BGCOLOR="#F1F5F9"><FONT FACE="' + FONT + '" POINT-SIZE="11">'
        '<B>@triton.jit  rmsnorm_kernel</B></FONT></TD></TR>'
        '<TR><TD ALIGN="LEFT"><FONT FACE="' + MONO + '" POINT-SIZE="10">'
        "x_ptr, weight_ptr, out_ptr,<BR/>"
        "stride_x_b, stride_x_m,<BR/>"
        "stride_out_b, stride_out_m,<BR/>"
        "N: tl.constexpr,<BR/>"
        "eps: tl.constexpr,<BR/>"
        "BLOCK_SIZE: tl.constexpr<BR/>"
        "<BR/>"
        "grid = (B, M)"
        '</FONT></TD></TR></TABLE>>'
    )
    d.node("code", code)

    d.node("cats", _html("Parameter taxonomy -> plugin mechanism", [
        "P_in   input pointers  -> plugin inputs",
        "P_out  output pointers -> plugin outputs",
        "P_inplace  load+store same ptr -> aliased IO",
        "S   strides          -> int64 runtime args",
        "D   host scalars     -> CPU inputs / members",
        "C   tl.constexpr     -> baked in PTX, dropped",
        "G   launch grid      -> C++ integer math",
    ], BLUE))

    d.node("mech", _html("Why it is hard", [
        "PTX re-orders / drops parameters and",
        "fixes bitwidths, so the host must rebuild",
        "the exact entry-slot ABI.",
        "",
        "Grids are arbitrary Python expressions",
        "over runtime shapes, so they must be",
        "re-evaluated inside the C++ engine.",
        "",
        "Scalars change every forward pass, so",
        "they cannot be hard-coded properties.",
    ], RED, mono=True))

    d.edge("code", "cats", style="invis")
    d.edge("cats", "mech", style="dashed", color="#94A3B8", dir="none")
    _render(d, "kernel_anatomy.png")


def deployment_gap():
    d = graphviz.Digraph("gap", graph_attr=dict(rankdir="TB", bgcolor="white",
                                                fontname=FONT, pad="0.3"))
    d.attr("node", shape="plaintext")

    with d.subgraph(name="cluster_bad") as c:
        c.attr(label="Without KernelLens", style="rounded", color=RED,
               fontcolor=RED, fontname=FONT)
        c.node("export", _html("torch.onnx.export(model)", [
            "traces TorchScript / TorchDynamo IR",
            "imperative @triton.jit calls are opaque",
            "-> export fails or drops the kernel",
        ], RED))
        c.node("optA", _html("Option A - stay in Python", [
            "Triton kernels run, but you forfeit",
            "TensorRT / ORT graph optimisation,",
            "engine tuning and a C++-only server",
        ], MUTED))
        c.node("optB", _html("Option B - hand-write C++", [
            "OrtCustomOp + IPluginV2DynamicExt,",
            "ONNX schema, PTX entry ABI, launch",
            "grids, build scripts: 1000s of lines",
        ], MUTED))
        c.edge("export", "optA"); c.edge("export", "optB")

    with d.subgraph(name="cluster_good") as c:
        c.attr(label="With KernelLens", style="rounded", color=GREEN,
               fontcolor=GREEN, fontname=FONT)
        c.node("call", _html("kl.compile(model, (x,))", [
            "dual-pass tracing + AST analysis +",
            "ONNX injection + C++/CUDA synthesis",
            "all run automatically",
        ], GREEN, mono=True))
        c.node("wins", _html("Whole-engine wins, retained", [
            "Standard layers fused / tuned by",
            "TensorRT & ORT graph compilers;",
            "custom Triton kernels run as native",
            "plugins sharing the VRAM planner",
        ], GREEN))
        c.node("cpp", _html("Pure C++ at inference", [
            "no Python, no libtorch in serving;",
            "multithread-safe; reproducible",
            "cold start from cached .so/.engine",
        ], GREEN))
        c.edge("call", "wins"); c.edge("wins", "cpp")

    _render(d, "deployment_gap.png")


def zero_copy():
    d = graphviz.Digraph("zc", graph_attr=dict(rankdir="TB", bgcolor="white",
                                               fontname=FONT, pad="0.3"))
    d.attr("node", shape="plainbox", style="rounded,filled", fontname=MONO,
           fontsize="10", fillcolor="#F8FAFC", color="#CBD5E1")

    with d.subgraph(name="cluster_ort") as c:
        c.attr(label="ONNX Runtime", color=BLUE, fontcolor=BLUE, fontname=FONT)
        c.node("o1", "io = session.io_binding()")
        c.node("o2", "io.bind_input(name, 'cuda', x.data_ptr())")
        c.node("o3", "allocate outputs on CUDA device")
        c.node("o4", "io.bind_output(name, 'cuda', out.data_ptr())")
        c.node("o5", "session.run_with_iobinding(io)")
        c.edges([("o1", "o2"), ("o2", "o3"), ("o3", "o4"), ("o4", "o5")])

    with d.subgraph(name="cluster_trt") as c:
        c.attr(label="NVIDIA TensorRT", color=PURPLE, fontcolor=PURPLE, fontname=FONT)
        c.node("t1", "ctx.set_input_shape(name, shape)")
        c.node("t2", "ctx.set_tensor_address(name, x.data_ptr())")
        c.node("t3", "allocate outputs on CUDA device")
        c.node("t4", "ctx.set_tensor_address(out, out.data_ptr())")
        c.node("t5", "ctx.execute_async_v3(stream)")
        c.edges([("t1", "t2"), ("t2", "t3"), ("t3", "t4"), ("t4", "t5")])

    d.node("note", "T = O(1) pointer registration + kernel   (no H2D/D2H staging copies)",
           shape="note", fontname=FONT, fontsize="11", fillcolor="#EFF6FF", color=BLUE)
    d.edge("o5", "note", style="invis")
    d.edge("t5", "note", style="invis")
    _render(d, "zero_copy_binding.png")


def manual_vs_kernellens():
    d = graphviz.Digraph("manual_vs_kl", graph_attr=dict(rankdir="LR", bgcolor="white",
                                                         fontname=FONT, pad="0.3",
                                                         nodesep="0.28", ranksep="0.55"))
    d.attr("node", shape="plaintext")

    with d.subgraph(name="cluster_manual") as c:
        c.attr(label="Without KernelLens: the pipeline you build by hand",
               style="rounded", color=RED, fontcolor=RED, fontname=FONT, fontsize="13")
        c.node("m1", _html("Kernel port", [
            "PyTorch model + @triton.jit kernels",
            "rewrite every kernel in CUDA C++",
        ], RED, mono=True))
        c.node("m2", _html("ONNX integration", [
            "Ort::CustomOp + Ort::CustomOpApi",
            "schema + domain registration",
        ], RED, mono=True))
        c.node("m3", _html("TensorRT integration", [
            "nvinfer1::IPluginV2DynamicExt",
            "nvinfer1::IPluginCreator + serialization",
        ], RED, mono=True))
        c.node("m4", _html("Build", [
            "hand-written CMakeLists",
            "nvcc + g++ -> libtriton_*.so",
        ], RED, mono=True))
        c.node("m5", _html("Patch the graph", [
            "PyTorch C++ extension for the FX graph",
            "@torch.library.custom_op wrappers",
            "inject triton_custom nodes",
        ], RED, mono=True))
        c.node("m6", _html("Artifacts", [
            "<name>.onnx  +  <name>.engine",
        ], RED, mono=True))
        c.node("m7", _html("Run on ONNX Runtime", [
            "CreateOrt::Session + load .so",
            "Ort::Binding on raw CUDA pointers",
            "RunWithBindings",
        ], RED, mono=True))
        c.node("m8", _html("Run on TensorRT", [
            "CreateIExecutionContext + deserialize",
            "IExecutionContext::setTensorAddress",
            "Enqueue Execution",
        ], RED, mono=True))
        c.edge("m1", "m2", style="invis")
        c.edge("m2", "m4", style="invis")
        c.edge("m3", "m4", style="invis")
        c.edge("m5", "m6", style="invis")
        c.edge("m6", "m7", style="invis")
        c.edge("m6", "m8", style="invis")
        c.edges([("m1", "m3"), ("m1", "m5"), ("m4", "m5"), ("m2", "m5"),
                 ("m3", "m5"), ("m6", "m7"), ("m6", "m8")])

    with d.subgraph(name="cluster_kl") as c:
        c.attr(label="With KernelLens", style="rounded", color=GREEN,
               fontcolor=GREEN, fontname=FONT, fontsize="13")
        c.node("k1", _html("Compile", [
            "kl.compile(model, sample_input,",
            "           backends=['onnx','tensorrt'])",
        ], GREEN, mono=True))
        c.node("k2", _html("Run", [
            "compiled_model.run(inputs, backend='onnx')",
            "compiled_model.run(inputs, backend='tensorrt')",
        ], GREEN, mono=True))
        c.edge("k1", "k2")
    _render(d, "manual_vs_kernellens.png")


if __name__ == "__main__":
    pipeline()
    anatomy()
    deployment_gap()
    zero_copy()
    manual_vs_kernellens()