"""
KernelLens v2 figure generator.

Produces explicit, information-dense figures used by both the rewritten
technical report and the GitHub Pages documentation site.

Run:  python3 scripts/make_figures_v2.py
Outputs land in docs/assets/figures/ and are copied to the repo root for LaTeX.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

THEME = os.environ.get("KL_THEME", "light")
if THEME == "dark":
    INK = "#F2EFE6"; MUTED = "#A8A294"; LINE = "#3A362E"; LIGHT = "#1A1915"; FACE = "none"
    BLUE = "#8FBEF5"; PURPLE = "#C3A6EE"; GREEN = "#6FD3A6"; AMBER = "#EDB267"; RED = "#F2968C"; TEAL = "#6FD0D0"
else:
    INK = "#0F172A"; MUTED = "#475569"; LINE = "#CBD5E1"; LIGHT = "#F8FAFC"; FACE = "white"
    BLUE = "#2563EB"; PURPLE = "#7C3AED"; GREEN = "#059669"; AMBER = "#D97706"; RED = "#DC2626"; TEAL = "#0D9488"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": LINE,
    "axes.linewidth": 1.0,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": FACE,
    "axes.facecolor": FACE,
    "savefig.facecolor": FACE,
    "savefig.transparent": True,
})


def _out(name):
    if THEME == "dark":
        return os.path.join(OUT, name[:-4] + "_dark.png")
    return os.path.join(OUT, name)


def _box(ax, x, y, w, h, title, body, edge, fill="#FFFFFF", title_size=10.5, body_size=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.6",
                                linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 3.1), w, 3.1, boxstyle="round,pad=0.32,rounding_size=0.5",
                                linewidth=0, facecolor=edge, alpha=0.14, zorder=3))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="center", color=edge,
            fontsize=title_size, fontweight="bold", zorder=4)
    for i, line in enumerate(body):
        ax.text(x + 1.4, y + h - 4.6 - i * 2.9, line, ha="left", va="top",
                color=MUTED, fontsize=body_size, zorder=4)


def _arrow(ax, p0, p1, color=LINE, lw=2.2, label=None, label_off=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 linewidth=lw, color=color, zorder=1,
                                 shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + label_off
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.2, color=color,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.9))


# ---------------------------------------------------------------------------
# 1. End-to-end pipeline (data-flow with concrete artifacts)
# ---------------------------------------------------------------------------
def pipeline_end_to_end(path):
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(50, 97, "KernelLens Compilation Pipeline: from Python kernel to C++ engine",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(50, 93.4, "Every arrow is a concrete on-disk or in-memory artifact, not a conceptual hand-off",
            ha="center", va="center", fontsize=10.5, color=MUTED, style="italic")

    # Inputs
    _box(ax, 1.5, 68, 21, 22, "PyTorch module",
         ["class RMSNorm(nn.Module):",
          "  def forward(x):",
          "    out = torch.empty_like(x)",
          "    rmsnorm_kernel[(B, M)](...)",
          "    return out",
          "",
          "@triton.jit  kernel source"],
         INK)
    _box(ax, 1.5, 40, 21, 22, "Sample inputs",
         ["x : (4, 512, 4096) fp32",
          "",
          "Profiles: run once on GPU",
          "to observe real grid,",
          "shared memory and strides"],
         MUTED)

    p1x, p1w = 26.5, 25
    _box(ax, p1x, 44, p1w, 46, "Phase 1 - Dual-pass tracing",
         ["Pass 1 (eager): intercepts compiled",
          "  PTX, shared-mem bytes, num_warps,",
          "  concrete grid + tensor strides",
          "Pass 2 (make_fx symbolic): extracts",
          "  grid expressions as SymInt strings",
          "Static AST scan: tl.load / tl.store",
          "  -> input / output / in-place kinds",
          "",
          "Artifact: KernelManifest",
          "  ptx, smem, warps, arg taxonomy,",
          "  grid exprs, mutability flags"],
         BLUE)

    p2x, p2w = 55.5, 21
    _box(ax, p2x, 68, p2w, 22, "Phase 2 - ONNX graph rewrite",
         ["torch.onnx.export (opset 17)",
          "Triton calls replaced by custom",
          "nodes  triton_custom::<kernel>",
          "Host scalars bound as CPU inputs",
          "In-place buffers aliased in graph",
          "",
          "Artifact: <name>.onnx"],
         PURPLE)
    _box(ax, p2x, 40, p2w, 22, "Phase 3 - C++/CUDA synthesis",
         ["Emit OrtCustomOp and TRT",
          "IPluginV2DynamicExt sources,",
          "PTX entry-slot ABI mapping,",
          "PTX ISA version clamping,",
          "dynamic grid C++ evaluation",
          "",
          "Artifacts: .cu/.h -> .so libraries"],
         GREEN)

    _box(ax, 80, 54, 18.5, 36, "Phase 4 - Zero-copy runtime",
         ["ctypes.CDLL loads .so",
          "",
          "ORT: IOBinding binds",
          "  torch data_ptr() directly",
          "TRT: set_tensor_address()",
          "  registers VRAM pointers",
          "",
          "No host staging copies.",
          "Outputs: torch tensors",
          "on the same CUDA stream",
          "",
          "Result: parity vs native",
          "Triton within tolerance"],
         AMBER)

    _arrow(ax, (22.5, 79), (26.5, 70), LINE, label="forward()", label_off=1.2)
    _arrow(ax, (22.5, 51), (26.5, 58), LINE, label="shapes/strides", label_off=1.2)
    _arrow(ax, (51.5, 74), (55.5, 79), BLUE, label="manifest", label_off=1.4)
    _arrow(ax, (51.5, 60), (55.5, 52), BLUE, label="manifest", label_off=-0.2)
    _arrow(ax, (66, 67.5), (66, 62.5), PURPLE, label="graph", label_off=1.2)
    _arrow(ax, (76.5, 78.5), (80, 76), PURPLE, label=".onnx", label_off=1.2)
    _arrow(ax, (76.5, 51), (80, 60), GREEN, label=".so", label_off=1.2)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Kernel anatomy -> parameter taxonomy
# ---------------------------------------------------------------------------
def kernel_anatomy(path):
    fig, ax = plt.subplots(figsize=(15.5, 7.6), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97, "What a Triton kernel actually presents to a C++ engine",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(50, 93.2, "Each of the seven parameter categories maps to a distinct mechanism in the generated plugin",
            ha="center", va="center", fontsize=10.5, color=MUTED, style="italic")

    ax.add_patch(FancyBboxPatch((2, 8), 40, 80, boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc=LIGHT, ec=LINE, lw=1.4, zorder=1))
    ax.text(22, 84, "Python @triton.jit signature", ha="center", fontsize=11.5,
            fontweight="bold", color=INK)
    code = [
        "def rmsnorm_kernel(",
        "    x_ptr, weight_ptr, out_ptr,   # tensors",
        "    stride_x_b, stride_x_m,       # strides",
        "    stride_out_b, stride_out_m,",
        "    N: tl.constexpr,              # compile-time",
        "    eps: tl.constexpr,            # compile-time",
        "    BLOCK_SIZE: tl.constexpr ):",
        "    ...",
        "grid = (B, M)  # -> (cdiv(M,BLOCK),)",
    ]
    for i, l in enumerate(code):
        ax.text(3.4, 79 - i * 4.2, l, ha="left", va="top", fontsize=8.6,
                family="DejaVu Sans Mono", color="#1E293B")

    cats = [
        ("P_in  Input pointers", "read-only VRAM addresses -> plugin inputs", BLUE),
        ("P_out  Output pointers", "write-only VRAM addresses -> plugin outputs", GREEN),
        ("P_inplace  Mutated pointers", "load+store same ptr -> aliased input/output", PURPLE),
        ("S  Strides", "element strides -> int64 runtime arguments", TEAL),
        ("D  Host scalars", "per-call values -> OrtMemTypeCPUInput / plugin member", AMBER),
        ("C  tl.constexpr", "baked into PTX, filtered from the ABI", MUTED),
        ("G  Launch grid", "SymInt expression -> C++ integer math at run time", RED),
    ]
    y = 78
    for name, desc, col in cats:
        ax.add_patch(FancyBboxPatch((46, y - 7.5), 52, 8.4, boxstyle="round,pad=0.3,rounding_size=0.5",
                                    fc="white", ec=col, lw=1.6, zorder=2))
        ax.add_patch(Rectangle((46, y - 7.5), 1.4, 8.4, fc=col, ec="none", zorder=3))
        ax.text(48.6, y - 1.4, name, ha="left", va="center", fontsize=9.6,
                fontweight="bold", color=col)
        ax.text(48.6, y - 5.2, desc, ha="left", va="center", fontsize=8.4, color=MUTED)
        y -= 11
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Deployment gap / value proposition
# ---------------------------------------------------------------------------
def deployment_gap(path):
    fig, ax = plt.subplots(figsize=(15, 7.4), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 97, "The deployment gap and how KernelLens closes it",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)

    ax.add_patch(FancyBboxPatch((2, 6), 45, 84, boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc="#FEF2F2", ec=RED, lw=1.6))
    ax.text(24.5, 85, "Without KernelLens", ha="center", fontsize=13, fontweight="bold", color=RED)
    _box(ax, 5, 72, 39, 10, "torch.onnx.export(model)",
         ["traces TorchScript/TorchDynamo IR ...", "imperative @triton.jit calls are opaque",
          "-> export fails or drops the kernel"], RED, fill="white")
    _arrow(ax, (24.5, 71.5), (24.5, 66.5), RED)
    _box(ax, 5, 50, 39, 16, "Option A - stay in Python",
         ["Triton kernels run, but you give up", "TensorRT / ONNX Runtime graph",
          "optimisation, engine tuning and", "a dependency-free C++ server"],
         MUTED, fill="white")
    _box(ax, 5, 28, 39, 20, "Option B - hand-write C++ plugins",
         ["Re-implement OrtCustomOp +", "IPluginV2DynamicExt, ONNX schema,",
          "PTX entry-slot ABI, launch grids,", "build scripts, memory rules",
          "-> thousands of lines per kernel"], MUTED, fill="white")

    ax.add_patch(FancyBboxPatch((51, 6), 47, 84, boxstyle="round,pad=0.4,rounding_size=0.8",
                                fc="#ECFDF5", ec=GREEN, lw=1.6))
    ax.text(74.5, 85, "With KernelLens", ha="center", fontsize=13, fontweight="bold", color=GREEN)
    _box(ax, 53.5, 66, 42.5, 16, "One call: kl.compile(model, (x,))",
         ["Dual-pass tracing + AST analysis +", "ONNX node injection + C++/CUDA synthesis",
          "run automatically"], GREEN, fill="white")
    _box(ax, 53.5, 42, 42.5, 20, "You still get the whole-engine wins",
         ["Standard layers are fused and tuned", "by the TRT/ORT graph compilers;",
          "custom Triton kernels run as", "native plugins in the same engine,",
          "sharing the VRAM planner"], GREEN, fill="white")
    _box(ax, 53.5, 20, 42.5, 18, "And it is plain C++ at inference",
         ["No Python, no libtorch in the", "serving path; multithread-safe;",
          "reproducible cold-start from", "a cached .so / .engine"], GREEN, fill="white")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Zero-copy binding (ORT + TRT)
# ---------------------------------------------------------------------------
def zero_copy_binding(path):
    fig, ax = plt.subplots(figsize=(15, 6.6), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 96, "Zero-copy VRAM binding at execution time",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(50, 92, "PyTorch tensor storage is handed to the engine by address; no H2D/D2H staging buffers",
            ha="center", va="center", fontsize=10.5, color=MUTED, style="italic")

    lanes = [
        ("ONNX Runtime", BLUE, [
            "session.io_binding()",
            "bind_input(name, 'cuda', t.data_ptr())",
            "allocate outputs on CUDA device",
            "bind_output(name, 'cuda', out.data_ptr())",
            "session.run_with_iobinding(io)",
        ]),
        ("NVIDIA TensorRT", PURPLE, [
            "context.set_input_shape(name, shape)",
            "context.set_tensor_address(name, t.data_ptr())",
            "allocate outputs on CUDA device",
            "context.set_tensor_address(out_name, out.data_ptr())",
            "context.execute_async_v3(stream)",
        ]),
    ]
    for i, (title, col, steps) in enumerate(lanes):
        x = 6 + i * 48
        ax.add_patch(FancyBboxPatch((x, 22), 40, 60, boxstyle="round,pad=0.4,rounding_size=0.7",
                                    fc="white", ec=col, lw=1.8))
        ax.text(x + 20, 77, title, ha="center", fontsize=12, fontweight="bold", color=col)
        for j, s in enumerate(steps):
            yy = 70 - j * 9.5
            ax.add_patch(FancyBboxPatch((x + 3, yy - 5.6), 34, 6.6,
                                        boxstyle="round,pad=0.25,rounding_size=0.4",
                                        fc=LIGHT, ec=col, lw=1.1))
            ax.text(x + 5, yy - 2.3, f"{j+1}. {s}", ha="left", va="center", fontsize=8.3,
                    family="DejaVu Sans Mono", color="#1E293B")

    ax.text(50, 12, "T_preregistered = O(1) pointer registration  +  T_kernel      "
                    "(vs  O(N) H2D copy + kernel + D2H copy for staging designs)",
            ha="center", va="center", fontsize=9.6, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec=BLUE, lw=1.2))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Micro-benchmark latencies (fresh RTX 5060 Ti data)
# ---------------------------------------------------------------------------
MICRO = {
    "LLaMA 3 RMSNorm\n(B=4,M=512,N=4096)": (0.5158, 0.1726, 0.1582, 0.1849),
    "PaLM SwiGLU\n(2048 x 4096)": (0.4290, 0.2524, 0.2525, 0.2825),
    "Qwen 2.5 RoPE\n(4,512,32,128)": (0.6704, 0.1745, 0.1595, 0.1951),
    "Fused CrossEntropy\n(B=32,V=32000)": (0.0130, 0.0133, 0.0105, 0.0398),
}


def results_latency(path):
    ops = list(MICRO.keys())
    eager = [MICRO[o][0] for o in ops]
    comp = [MICRO[o][1] for o in ops]
    trit = [MICRO[o][2] for o in ops]
    kl = [MICRO[o][3] for o in ops]
    x = np.arange(len(ops)); w = 0.2
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=200)
    ax.bar(x - 1.5 * w, eager, w, label="PyTorch Eager", color=RED)
    ax.bar(x - 0.5 * w, comp, w, label="torch.compile (Inductor)", color=AMBER)
    ax.bar(x + 0.5 * w, trit, w, label="Native Triton", color="#3B82F6")
    ax.bar(x + 1.5 * w, kl, w, label="KernelLens C++ plugin (ORT)", color=GREEN)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Single-operator latency on RTX 5060 Ti (fp32, CUDA event timing)",
                 fontsize=13.5, fontweight="bold", color=INK, pad=12)
    ax.set_xticks(x); ax.set_xticklabels(ops, fontsize=9.2)
    ax.legend(frameon=True, fontsize=9.5)
    ax.grid(axis="y", ls="--", alpha=0.45)
    for xi, v in zip(x + 1.5 * w, kl):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=GREEN, fontweight="bold")
    ax.text(0.5, 0.94, "KernelLens runs the same Triton-level kernel inside a C++ engine;\n"
                       "the residual gap vs native Triton is host/IO-binding overhead",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec=GREEN, lw=1.2), color="#065F46")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. End-to-end LLaMA decoder (fresh data)
# ---------------------------------------------------------------------------
E2E = {
    "PyTorch Eager": (7.82, 0.866, 0.118, 478.0),
    "torch.compile": (7.91, 0.866, 0.118, 473.0),
    "KernelLens ORT": (4.38, 1.016, 0.133, 464.2),
    "KernelLens TensorRT": (4.05, 0.854, 0.112, 464.2),
}


def results_e2e(path):
    names = list(E2E.keys())
    colors = [RED, AMBER, BLUE, GREEN]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.6), dpi=200)
    metrics = [("TTFT  S=512 (ms)", 0, False), ("ITL  S=1 (ms)", 1, False),
               ("Total 128 tok (s)", 2, False), ("Peak VRAM (MB)", 3, False)]
    for ax, (title, idx, _) in zip(axes, metrics):
        vals = [E2E[n][idx] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, width=0.66)
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["Eager", "compile", "KL-ORT", "KL-TRT"], fontsize=9)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.4, fontweight="bold", color="#334155")
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_ylim(0, max(vals) * 1.22)
    fig.suptitle("End-to-end 4-layer LLaMA decoder (D=1024) on RTX 5060 Ti",
                 fontsize=14, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Speedups vs eager / compile (TRT plugins)
# ---------------------------------------------------------------------------
SPEED = {
    "Mamba SSM\nrecurrence": (3.32, 0.104, 0.160),
    "Squared-ReLU\nFlashAttn": (5.96, 3.16, 0.949),
    "Fused Res+RMSNorm": (3.40, 1.407, 1.407),
    "Fused CrossEntropy\n(V=128k)": (9.09, 2.46, 4.90),
}


def results_speedup(path):
    names = list(SPEED.keys())
    eager = np.array([SPEED[n][0] for n in names])
    comp = np.array([SPEED[n][1] for n in names])
    kl = np.array([SPEED[n][2] for n in names])
    vs_eager = eager / kl
    vs_comp = comp / kl
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=200)
    b1 = ax.bar(x - w / 2, vs_eager, w, label="vs PyTorch Eager", color=BLUE)
    b2 = ax.bar(x + w / 2, vs_comp, w, label="vs torch.compile", color=AMBER)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.1)
    ax.set_ylabel("Speedup of KernelLens TensorRT plugin (x)")
    ax.set_title("Where the C++ TensorRT plugin wins on real LLM kernels",
                 fontsize=13.5, fontweight="bold", color=INK, pad=12)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.2)
    ax.legend(fontsize=9.5)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:.2f}x", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.2, fontweight="bold", color="#334155")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Verification matrix
# ---------------------------------------------------------------------------
def verification_matrix(path):
    rows = [
        "RMSNorm / SwiGLU / RoPE / FusedCE",
        "Fused Residual + RMSNorm",
        "Mamba SSM recurrence",
        "Tricky LayerNorm + GEMM-sigmoid",
        "FP8 scaled GEMM / INT4 W4A16 GEMM",
        "In-place mutation (fp32/fp16)",
        "Dynamic scalars / 4D grid / nested store",
        "Edge: multi-output rank4/rank3 LSE",
        "Edge: INT8 dequant + bias + SiLU",
        "Edge: SwiGLU multi-scalar",
        "Squared-ReLU FlashAttn (vs eager)",
    ]
    ort = ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "n/a"]
    trt = ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "n/a"]
    note = ["exact 0.0", "<=9.5e-7", "4.8e-7", "exact", "exact",
            "exact (via ORT)", "exact", "exact", "exact (via ORT)", "exact", "0.78 vs eager"]
    cmap = {"PASS": GREEN, "FAIL": RED, "n/a": "#94A3B8"}
    fig, ax = plt.subplots(figsize=(12.6, 7.2), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(0, len(rows) + 1.6); ax.axis("off")
    ax.text(5, len(rows) + 1.0, "Verification matrix (RTX 5060 Ti, sm_120a)",
            ha="center", fontsize=14.5, fontweight="bold", color=INK)
    ax.text(1.55, len(rows) + 0.1, "Workload", ha="left", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(6.0, len(rows) + 0.1, "ORT", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(7.3, len(rows) + 0.1, "TRT", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(9.0, len(rows) + 0.1, "Measured result", ha="center", fontsize=9.5,
            fontweight="bold", color=INK)
    for i, name in enumerate(rows):
        y = len(rows) - i - 0.35
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.2, y - 0.45), 9.6, 0.9, fc=LIGHT, ec="none"))
        ax.text(0.35, y, name, ha="left", va="center", fontsize=8.9, color="#1E293B")
        for xpos, val in ((6.0, ort[i]), (7.3, trt[i])):
            ax.add_patch(FancyBboxPatch((xpos - 0.42, y - 0.3), 0.84, 0.6,
                                        boxstyle="round,pad=0.05,rounding_size=0.15",
                                        fc=cmap[val], ec="none"))
            ax.text(xpos, y, val, ha="center", va="center", fontsize=7.8,
                    color="white", fontweight="bold")
        ax.text(9.0, y, note[i], ha="center", va="center", fontsize=8.2, color=MUTED)
    ax.text(5, 0.1, "Parity is measured against native Triton execution of the same kernel.",
            ha="center", fontsize=8.6, color=MUTED, style="italic")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    # Charts only. The flow diagrams are produced by scripts/make_diagrams.py
    # (Graphviz) so they stay in sync with the manual-pipeline diagram.
    figs = {
        "results_micro_latency.png": results_latency,
        "results_e2e.png": results_e2e,
        "results_speedup.png": results_speedup,
        "verification_matrix.png": verification_matrix,
    }
    for name, fn in figs.items():
        out = _out(name)
        fn(out)
        print("wrote", out)


if __name__ == "__main__":
    main()