"""
Modern pipeline diagrams (Mermaid) in light and dark variants.

Run:  python3 scripts/make_mermaid_diagrams.py
Requires the mermaid CLI (mmdc) on PATH and a Chrome binary for Puppeteer.
"""
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

PALETTES = {
    "light": dict(
        ink="#0F172A", muted="#475569", line="#94A3B8",
        blue_f="#EFF6FF", blue_s="#2563EB",
        purple_f="#F5F3FF", purple_s="#7C3AED",
        green_f="#ECFDF5", green_s="#059669",
        amber_f="#FFFBEB", amber_s="#D97706",
        teal_f="#F0FDFA", teal_s="#0D9488",
    ),
    "dark": dict(
        ink="#F2EFE6", muted="#A8A294", line="#5A554A",
        blue_f="#132033", blue_s="#5B9BD0",
        purple_f="#1D1730", purple_s="#A273C4",
        green_f="#0F2A22", green_s="#43A97F",
        amber_f="#2A2113", amber_s="#E0A458",
        teal_f="#0F2626", teal_s="#4FB8B8",
    ),
}


def classdefs(p):
    return (
        f"  classDef ink fill:transparent,stroke:{p['line']},stroke-width:1.2px,color:{p['ink']},rx:8,ry:8\n"
        f"  classDef muted fill:transparent,stroke:{p['line']},stroke-width:1.2px,color:{p['muted']},rx:8,ry:8\n"
        f"  classDef blue fill:{p['blue_f']},stroke:{p['blue_s']},stroke-width:1.5px,color:{p['ink']},rx:8,ry:8\n"
        f"  classDef purple fill:{p['purple_f']},stroke:{p['purple_s']},stroke-width:1.5px,color:{p['ink']},rx:8,ry:8\n"
        f"  classDef green fill:{p['green_f']},stroke:{p['green_s']},stroke-width:1.5px,color:{p['ink']},rx:8,ry:8\n"
        f"  classDef amber fill:{p['amber_f']},stroke:{p['amber_s']},stroke-width:1.5px,color:{p['ink']},rx:8,ry:8\n"
        f"  classDef teal fill:{p['teal_f']},stroke:{p['teal_s']},stroke-width:1.5px,color:{p['ink']},rx:8,ry:8\n"
    )


def init(p):
    return (
        '%%{init: {"theme":"base","themeVariables":{'
        f'"fontFamily":"Inter, DejaVu Sans, sans-serif","fontSize":"14px",'
        f'"lineColor":"{p["line"]}","primaryColor":"{p["blue_f"]}",'
        f'"primaryTextColor":"{p["ink"]}","clusterBkg":"transparent",'
        f'"clusterBorder":"{p["line"]}","edgeLabelBackground":"transparent"'
        '}}}%%\n'
    )


def compile_pipeline(p):
    return init(p) + "flowchart TB\n" + classdefs(p) + """
  M["PyTorch module<br/>@triton.jit kernels"]:::ink
  X["sample input"]:::muted
  subgraph T["kl.compile · 1 trace"]
    direction TB
    T1["eager pass<br/>PTX · smem · warps · grid"]:::blue
    T2["symbolic pass<br/>grid expression"]:::blue
    T3["AST scan<br/>input / output / in-place"]:::blue
  end
  MAN["KernelManifest<br/>ptx · shared memory · num_warps<br/>argument taxonomy · grid_cxx_exprs"]:::teal
  subgraph RW["kl.compile · 2 rewrite"]
    direction TB
    R1["torch.onnx.export"]:::purple
    R2["inject triton_custom nodes"]:::purple
    R3["scalars → CPU inputs<br/>in-place → aliased"]:::purple
  end
  ONNX["&lt;name&gt;.onnx"]:::purple
  subgraph SY["kl.compile · 3 synthesize"]
    direction TB
    S1["OrtCustomOp class"]:::green
    S2["IPluginV2DynamicExt<br/>+ IPluginCreator"]:::green
    S3["PTX entry-slot ABI<br/>grid evaluated in C++"]:::green
  end
  subgraph BD["kl.compile · 4 build"]
    direction TB
    B1["nvcc + g++"]:::amber
    B2["PTX ISA clamp<br/>16-byte alignment guard"]:::amber
  end
  SO["libtriton_*.so<br/>+ &lt;name&gt;.engine"]:::green
  subgraph RUN["compiled_model.run(inputs, backend=...)"]
    direction TB
    RO["ONNX Runtime<br/>session → IOBinding → RunWithBindings"]:::blue
    RT["TensorRT<br/>deserialize → setTensorAddress → enqueue"]:::purple
  end
  OUT["torch tensors · zero-copy"]:::ink
  M --> T
  X --> T
  T --> MAN
  MAN --> RW
  RW --> ONNX
  MAN --> SY
  SY --> BD
  BD --> SO
  ONNX --> RO
  SO --> RO
  SO --> RT
  RO --> OUT
  RT --> OUT
"""


def run_pipeline(p):
    return init(p) + "flowchart LR\n" + classdefs(p) + """
  A["compiled_model.run(inputs, backend)"]:::ink
  subgraph ORT["ONNX Runtime lane"]
    direction LR
    O1["create Ort::Session<br/>load libtriton_ort_plugins.so"]:::blue --> O2["IOBinding on<br/>raw CUDA pointers"]:::blue --> O3["RunWithBindings"]:::blue
  end
  subgraph TRT["TensorRT lane"]
    direction LR
    T1["deserialize &lt;name&gt;.engine<br/>→ IExecutionContext"]:::purple --> T2["IExecutionContext::<br/>setTensorAddress"]:::purple --> T3["Enqueue Execution"]:::purple
  end
  OUT["torch tensors on the CUDA stream<br/>no host staging copies"]:::ink
  A --> ORT
  A --> TRT
  O3 --> OUT
  T3 --> OUT
"""


def render(name, src, theme):
    suffix = "_dark" if theme == "dark" else ""
    out = os.path.join(OUT, f"{name}{suffix}.png")
    with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as f:
        f.write(src)
        mmd = f.name
    env = dict(os.environ)
    env.setdefault("PUPPETEER_EXECUTABLE_PATH", "/usr/bin/google-chrome-stable")
    cmd = ["mmdc", "-i", mmd, "-o", out, "-b", "transparent", "-s", "3"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    os.unlink(mmd)
    if res.returncode != 0:
        raise RuntimeError(f"mmdc failed for {name} ({theme}): {res.stderr[-400:]}")
    print("wrote", out)


def main():
    for theme, p in PALETTES.items():
        render("pipeline_end_to_end", compile_pipeline(p), theme)
        render("run_pipeline", run_pipeline(p), theme)


if __name__ == "__main__":
    main()