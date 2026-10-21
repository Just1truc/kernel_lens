"""
Derive the TMLR versions from the fresh report.

  kernel_lens_tmlr_fresh.tex       non-anonymous preprint (arXiv / the site)
  kernel_lens_tmlr_submission.tex  anonymized submission (for OpenReview)

TMLR reviews double blind, so the submission drops the author block, the
GitHub URL and any other identifying detail; the style file then prints the
"Anonymous authors / Paper under double-blind review" block itself.

Run:  python3 scripts/make_tmlr_fresh.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "kernel_lens_fresh.tex")

with open(SRC) as f:
    BASE = f.read()

TITLE = (
    "\\title{\\textbf{KernelLens: Recovering the Triton-to-C++ Engine Contract} \\\\\n"
    "\\large Automatic TensorRT and ONNX Runtime Plugins from \\texttt{@triton.jit} Kernels}\n"
)
AUTHOR = (
    "\\author{\\name Justin Duc \\email justin.duc@galiad-research.com \\\\\n"
    "      \\addr Galiad Research}\n"
)
DATES = "\\def\\month{10}\n\\def\\year{2026}\n"

# identifying string that must not survive into the submission
GITHUB = " and on GitHub (\\url{https://github.com/Just1truc/kernel_lens})"


def build(preprint: bool) -> str:
    text = BASE
    pkg = "\\usepackage[preprint]{tmlr}" if preprint else "\\usepackage{tmlr}"
    text = text.replace(
        "\\documentclass[11pt,a4paper]{article}",
        "\\documentclass[10pt]{article}\n" + pkg + "\n\\setcitestyle{numbers,square}",
        1,
    )
    text = text.replace("\\usepackage{geometry}", "% \\usepackage{geometry}", 1)
    text = text.replace("\\geometry{margin=1in}", "% \\geometry{margin=1in}", 1)

    header = TITLE + (AUTHOR if preprint else "") + DATES
    text, n = re.subn(r"\\title\{.*?\\date\{", lambda m: header + "\\date{",
                      text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("could not locate the title block")

    if not preprint:
        text = text.replace(GITHUB, "")
    return text


for preprint, name in [(True, "kernel_lens_tmlr_fresh.tex"),
                       (False, "kernel_lens_tmlr_submission.tex")]:
    path = os.path.join(ROOT, name)
    with open(path, "w") as f:
        f.write(build(preprint))
    print("wrote", path)
