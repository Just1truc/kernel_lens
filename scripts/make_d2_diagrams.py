"""
Modern pipeline diagrams (D2) in light and dark variants.

D2 renders to SVG; we rasterise with headless Chrome (D2's own PNG export needs
Playwright, which is unavailable here).

Run:  python3 scripts/make_d2_diagrams.py
Needs: d2 on PATH (or D2_BIN), google-chrome for rasterisation.
"""
import os
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "figures")
os.makedirs(OUT, exist_ok=True)

D2 = shutil.which("d2") or os.environ.get("D2_BIN", "")
CHROME = shutil.which("google-chrome-stable") or shutil.which("google-chrome") or shutil.which("chromium")

P = {
    "light": dict(ink="#0F172A", muted="#475569", line="#94A3B8",
                  blue_f="#EFF6FF", blue_s="#2563EB", blue_bg="#DBEAFE",
                  purple_f="#F5F3FF", purple_s="#7C3AED", purple_bg="#EDE9FE",
                  green_f="#ECFDF5", green_s="#059669", green_bg="#D1FAE5",
                  amber_f="#FFFBEB", amber_s="#D97706", amber_bg="#FEF3C7",
                  teal_f="#F0FDFA", teal_s="#0D9488", teal_bg="#CCFBF1"),
    "dark": dict(ink="#F2EFE6", muted="#A8A294", line="#5A554A",
                 blue_f="#132033", blue_s="#5B9BD0", blue_bg="#0E1A2B",
                 purple_f="#1D1730", purple_s="#A273C4", purple_bg="#171226",
                 green_f="#0F2A22", green_s="#43A97F", green_bg="#0C211B",
                 amber_f="#2A2113", amber_s="#E0A458", amber_bg="#211A0F",
                 teal_f="#0F2626", teal_s="#4FB8B8", teal_bg="#0C1F1F"),
}


def classes(p):
    out = ["classes: {"]
    for name, f, s in [("ink", "transparent", p["line"]), ("muted", "transparent", p["line"]),
                       ("blue", p["blue_f"], p["blue_s"]), ("purple", p["purple_f"], p["purple_s"]),
                       ("green", p["green_f"], p["green_s"]), ("amber", p["amber_f"], p["amber_s"]),
                       ("teal", p["teal_f"], p["teal_s"])]:
        col = p["ink"] if name not in ("muted",) else p["muted"]
        out.append(f'  {name}: {{ style: {{ fill: "{f}"; stroke: "{s}"; font-color: "{col}"; '
                   f'border-radius: 8; stroke-width: 2 }} }}')
    for name, bg, s in [("cblue", p["blue_bg"], p["blue_s"]), ("cpurple", p["purple_bg"], p["purple_s"]),
                        ("cgreen", p["green_bg"], p["green_s"]), ("camber", p["amber_bg"], p["amber_s"])]:
        out.append(f'  {name}: {{ style: {{ fill: "{bg}"; stroke: "{s}"; font-color: "{p["ink"]}"; '
                   f'border-radius: 10; stroke-width: 2 }} }}')
    out.append("}")
    return "\n".join(out)


def compile_d2(p):
    return f"""direction: down

{classes(p)}

mod: "PyTorch module\\n@triton.jit kernels" {{ class: ink }}
inp: "sample input" {{ class: muted }}

manifest: "KernelManifest\\nptx · shared memory · num_warps\\nargument taxonomy · grid_cxx_exprs" {{ class: teal }}

trace: "kl.compile · 1 trace" {{
  class: cblue
  t1: "eager pass\\nPTX · smem · warps · grid" {{ class: blue }}
  t2: "symbolic pass\\ngrid expression" {{ class: blue }}
  t3: "AST scan\\ninput / output / in-place" {{ class: blue }}
}}

rewrite: "kl.compile · 2 rewrite" {{
  class: cpurple
  r1: "torch.onnx.export" {{ class: purple }}
  r2: "inject triton_custom nodes" {{ class: purple }}
  r3: "scalars → CPU inputs\\nin-place → aliased" {{ class: purple }}
}}

synthesize: "kl.compile · 3 synthesize" {{
  class: cgreen
  s1: "OrtCustomOp class" {{ class: green }}
  s2: "IPluginV2DynamicExt\\n+ IPluginCreator" {{ class: green }}
  s3: "PTX entry-slot ABI\\ngrid evaluated in C++" {{ class: green }}
}}

build: "kl.compile · 4 build" {{
  class: camber
  b1: "nvcc + g++" {{ class: amber }}
  b2: "PTX ISA clamp\\n16-byte alignment guard" {{ class: amber }}
}}

onnx: "<name>.onnx" {{ class: purple }}
so: "libtriton_*.so\\n+ <name>.engine" {{ class: green }}

run: "compiled_model.run(inputs, backend=...)" {{
  class: camber
  ro: "ONNX Runtime\\nsession → IOBinding → RunWithBindings" {{ class: blue }}
  rt: "TensorRT\\ndeserialize → setTensorAddress → enqueue" {{ class: purple }}
}}

out: "torch tensors · zero-copy" {{ class: ink }}

mod -> trace: forward()
inp -> trace: shapes
trace -> manifest: analysis
manifest -> rewrite
rewrite -> onnx
manifest -> synthesize
synthesize -> build
build -> so
onnx -> run.ro
so -> run.ro
so -> run.rt
run.ro -> out
run.rt -> out
"""


def run_d2(p):
    return f"""direction: right

{classes(p)}

a: "compiled_model.run(inputs, backend)" {{ class: ink }}

ort: "ONNX Runtime lane" {{
  class: cblue
  direction: right
  o1: "create Ort::Session\\nload libtriton_ort_plugins.so" {{ class: blue }}
  o2: "IOBinding on\\nraw CUDA pointers" {{ class: blue }}
  o3: "RunWithBindings" {{ class: blue }}
  o1 -> o2 -> o3
}}

trt: "TensorRT lane" {{
  class: cpurple
  direction: right
  t1: "deserialize <name>.engine\\n→ IExecutionContext" {{ class: purple }}
  t2: "IExecutionContext::\\nsetTensorAddress" {{ class: purple }}
  t3: "Enqueue Execution" {{ class: purple }}
  t1 -> t2 -> t3
}}

out: "torch tensors on the CUDA stream\\nno host staging copies" {{ class: ink }}

a -> ort
a -> trt
ort.o3 -> out
trt.t3 -> out
"""


def svg_to_png(svg, png):
    head = open(svg, encoding="utf-8").read(3000)
    w = int(re.search(r'width="(\d+)"', head).group(1))
    h = int(re.search(r'height="(\d+)"', head).group(1))
    scale = 2
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--hide-scrollbars", "--default-background-color=00000000",
                    f"--force-device-scale-factor={scale}",
                    f"--window-size={w},{h}",
                    f"--user-data-dir={tempfile.mkdtemp()}",
                    f"--screenshot={png}", f"file://{svg}"],
                   check=True, capture_output=True)


def render(name, src, theme):
    suffix = "_dark" if theme == "dark" else ""
    d2f = tempfile.NamedTemporaryFile("w", suffix=".d2", delete=False)
    d2f.write(src)
    d2f.close()
    svg = d2f.name[:-3] + ".svg"
    subprocess.run([D2, "--theme", "0", "--pad", "30", d2f.name, svg], check=True, capture_output=True)
    # D2 draws an opaque white root rect; strip it so the figure sits on the page.
    s = open(svg, encoding="utf-8").read()
    s = s.replace('fill="#FFFFFF" class=" fill-N7"', 'fill="none"', 1)
    open(svg, "w", encoding="utf-8").write(s)
    png = os.path.join(OUT, f"{name}{suffix}.png")
    svg_to_png(svg, png)
    for f in (d2f.name, svg):
        os.unlink(f)
    print("wrote", png)


def main():
    for theme, pal in P.items():
        render("pipeline_end_to_end", compile_d2(pal), theme)
        render("run_pipeline", run_d2(pal), theme)


if __name__ == "__main__":
    main()