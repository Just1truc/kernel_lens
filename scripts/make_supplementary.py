"""
Build the anonymized supplementary archive for the TMLR submission.

    python3 scripts/make_supplementary.py

Writes kernel_lens_supplementary.zip at the repository root: the package, the
evaluation scripts, the figure scripts and the raw measurements, with every
identifying detail removed (author, affiliation, e-mail, GitHub and docs URLs,
absolute home paths).
"""
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join("/tmp", "opencode", "kl_supp", "kernel_lens_supplementary")
OUT = os.path.join(ROOT, "kernel_lens_supplementary.zip")

FILES = ["pyproject.toml", "LICENSE", "README.md", "logo.png"]
TREES = ["kernel_lens", "tests", "docs/assets/data"]
SCRIPTS = [
    "benchmark_quantization.py",
    "generate_plots.py",
    "make_d2_diagrams.py",
    "make_figures_v2.py",
]
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".cfg", ".json", ".cu",
            ".cuh", ".h", ".sh", ".yaml", ".yml", ".in", ".rst", ""}

# identifying strings -> neutral replacements (order matters: longest first)
SUBS = [
    (r"git clone https://github\.com/Just1truc/kernel_lens\.git && cd kernel_lens\n", ""),
    (r"## Technical report\n\n[\s\S]*?\n\n(?=## )", ""),
    (r"\[!\[DOI\]\(https://zenodo\.org/badge/DOI/[^)]*\)\]\(https://doi\.org/[^)]*\)\s*", ""),
    (r"duc2026kernellens", "anonymous2026kernellens"),
    (r"10\.5281/zenodo\.\d+", "10.0000/anonymous.kernellens"),
    (r"https://raw\.githubusercontent\.com/Just1truc/kernel_lens/main/logo\.png", "logo.png"),
    (r"https?://(?:www\.)?just1truc\.github\.io/kernel_lens/?",
     "https://example.org/kernel_lens/"),
    (r"https://github\.com/Just1truc/kernel_lens/issues",
     "https://example.org/kernel_lens/issues"),
    (r"https://github\.com/Just1truc/kernel_lens", "https://example.org/kernel_lens"),
    (r"git@github\.com:Just1truc/kernel_lens\.git", "https://example.org/kernel_lens.git"),
    (r"Just1truc/kernel_lens", "anonymous/kernel_lens"),
    (r"Just1truc", "anonymous"),
    (r"justin\.duc@[\w.\-]+", "anonymous@example.org"),
    (r"Justin Duc", "Anonymous Author"),
    (r"Duc,\s*Justin", "Anonymous, Author"),
    (r"Duc Justin", "Anonymous Author"),
    (r"Galiad Research", "Anonymous Institution"),
    (r'"/home/ostentatoire/Celia/auto_kernel_gen/kernel_lens"',
     'os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'),
    (r"/home/ostentatoire/\.kernel_lens_cache", "/tmp/.kernel_lens_cache"),
    (r"/home/ostentatoire", "/home/user"),
    (r"\bostentatoire\b", "user"),
]
FORBIDDEN = [r"\bjustin", r"galiad", r"just1truc", r"epitech", r"ostentatoire",
             r"\bduc\b", r"\bduc\d", r"zenodo"]


def ignored(name):
    return name in ("__pycache__", ".pytest_cache", ".git") or name.endswith(".pyc")


def scrub(path):
    if os.path.splitext(path)[1].lower() not in TEXT_EXT:
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (UnicodeDecodeError, OSError):
        return 0
    original = text
    for pat, rep in SUBS:
        text = re.sub(pat, rep, text)
    if text != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return 1
    return 0


REPRODUCE = """# Reproducing the paper's results

This archive accompanies the submission *KernelLens: Recovering the
Triton-to-C++ Engine Contract*. It contains the `kernel_lens` package, the
evaluation scripts and the raw measurements behind every table.

    kernel_lens/          the compiler (trace -> manifest -> C++ -> engine)
    tests/                evaluation and regression scripts
    scripts/              figure and quantization scripts
    docs/assets/data/     raw JSON measurements

## Environment

All measurements were produced on a single NVIDIA RTX 5060 Ti (Blackwell,
sm_120), CUDA 12.8, PyTorch 2.x, Triton 3.x, TensorRT 10.16, ONNX Runtime 1.23.
`nvcc` and the vendored TensorRT headers (`kernel_lens/include/tensorrt/`) are
needed to build the generated plugins.

    pip install -e ".[all]"

## Table -> script -> raw data

| Paper table | Script | Raw data |
|---|---|---|
| 3  (generated C++ volume) | `tests/measure_codegen.py` | - |
| 4  (numerical parity) | `tests/test_research_kernels.py` | - |
| 5  (folding plus kernels) | `tests/benchmark_folding_vs_triton.py` | `measured_folding_vs_triton.json` |
| 6  (single-operator latency) | `tests/benchmark_research_kernels.py` | `measured_latencies_rtx5060ti.json` |
| 7  (end-to-end) | `tests/benchmark_e2e_llama.py` | `measured_e2e_llama_rtx5060ti.json` |
| 8  (quantized kernels) | `tests/test_fp8_gemm.py`, `tests/test_int4_dequant_gemm.py` | `measured_quantization_latencies.json` |

Tables 1 and 2 are conceptual (the contract and where each component is
recovered from) and have no measurement behind them.

The coverage claim of Section 8.3 is demonstrated by `tests/test_conformance.py`
(fourteen module patterns). Raw measurements are committed as JSON under
`docs/assets/data/`.
"""


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    with open(os.path.join(STAGE, "REPRODUCE.md"), "w") as f:
        f.write(REPRODUCE)

    for f in FILES:
        shutil.copy2(os.path.join(ROOT, f), os.path.join(STAGE, f))
    for tree in TREES:
        shutil.copytree(os.path.join(ROOT, tree), os.path.join(STAGE, tree),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    os.makedirs(os.path.join(STAGE, "scripts"))
    for s in SCRIPTS:
        shutil.copy2(os.path.join(ROOT, "scripts", s), os.path.join(STAGE, "scripts", s))

    changed = 0
    for base, dirs, files in os.walk(STAGE):
        dirs[:] = [d for d in dirs if not ignored(d)]
        for name in files:
            if not ignored(name):
                changed += scrub(os.path.join(base, name))
    print(f"scrubbed {changed} files")

    hits = []
    for base, dirs, files in os.walk(STAGE):
        for name in files:
            path = os.path.join(base, name)
            if os.path.splitext(path)[1].lower() not in TEXT_EXT:
                continue
            try:
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            for pat in FORBIDDEN:
                if re.search(pat, text, re.I):
                    hits.append((os.path.relpath(path, STAGE), pat))
    if hits:
        print("REMAINING IDENTIFYING STRINGS:")
        for rel, pat in hits:
            print(f"  {rel}: {pat}")
        raise SystemExit(1)
    print("no identifying strings remain")

    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for base, dirs, files in os.walk(STAGE):
            for name in sorted(files):
                path = os.path.join(base, name)
                z.write(path, os.path.relpath(path, os.path.dirname(STAGE)))
                count += 1
    print(f"wrote {OUT} ({count} files, {os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
