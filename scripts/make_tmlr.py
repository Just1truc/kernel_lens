"""
Derive the TMLR submission (kernel_lens_tmlr.tex) from the main report.

The main report (architecture_report.tex) is the single source of truth; this
script only swaps the preamble/class, title and author block, and removes the
table of contents, so the two never drift.

Run:  python3 scripts/make_tmlr.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "architecture_report.tex")
DST = os.path.join(ROOT, "kernel_lens_tmlr.tex")

with open(SRC, "r") as f:
    text = f.read()

# 1. Class + tmlr package, numeric citations.
text = text.replace(
    "\\documentclass[11pt,a4paper]{article}",
    "\\documentclass[10pt]{article}\n\\usepackage[preprint]{tmlr}\n\\setcitestyle{numbers,square}",
    1,
)

# 2. tmlr sets its own margins.
text = text.replace("\\usepackage{geometry}", "% \\usepackage{geometry}", 1)
text = text.replace("\\geometry{margin=1in}", "% \\geometry{margin=1in}", 1)

# 3. algorithmic's \AND clashes with tmlr's \AND.
text = text.replace("\\usepackage{algorithmic}", "\\let\\AND\\relax\n\\usepackage{algorithmic}", 1)

# 4. Drop the table of contents (tmlr submissions do not use one).
text = text.replace("\\tableofcontents\n\\newpage", "", 1)

# 5. Replace the whole \title{...}\author{...} region with a tmlr block.
tmlr_header = (
    "\\title{\\textbf{KernelLens: Compiling PyTorch Triton Kernels into C++ "
    "Inference Plugins}}\n"
    "\\author{\\name Justin Duc \\email justin.duc@epitech.eu \\\\\n"
    "      \\addr Galiad Research}\n"
    "\\def\\month{10}\n"
    "\\def\\year{2026}\n"
)
text, n = re.subn(r"\\title\{.*?\\date\{", lambda m: tmlr_header + "\\date{", text, count=1, flags=re.S)
if n != 1:
    raise SystemExit("Failed to locate the title/author block")

with open(DST, "w") as f:
    f.write(text)

print("Wrote", DST)