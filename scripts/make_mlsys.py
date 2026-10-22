"""
Derive the MLSys 2027 submission from the technical report.

    python3 scripts/make_mlsys.py

Writes kernel_lens_mlsys.tex: MLSys 2027 two-column format (10 pages +
references), anonymized for double-blind review, with the mandatory AI Usage
Disclosure section after the bibliography.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "kernel_lens_fresh.tex")
DST = os.path.join(ROOT, "kernel_lens_mlsys.tex")

with open(SRC) as f:
    text = f.read()

PREAMBLE = r"""\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{tabularx}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{colortbl}
\usepackage{placeins}
\usepackage{hyperref}

\usepackage{mlsys2025}

% the bibliography is numeric; the style defaults to author-year
\setcitestyle{numbers,square}

\mlsystitlerunning{KernelLens: Recovering the Triton-to-C++ Engine Contract}

\graphicspath{{docs/assets/figures/}{./}}

\definecolor{primaryblue}{RGB}{24, 76, 120}
\definecolor{backcolour}{rgb}{0.97,0.97,0.97}
\definecolor{codegreen}{rgb}{0,0.5,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{bordergray}{rgb}{0.85,0.85,0.85}

\hypersetup{colorlinks=true, linkcolor=primaryblue, citecolor=primaryblue, urlcolor=primaryblue}
\emergencystretch=6em

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    keywordstyle=\color{primaryblue}\bfseries,
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\tiny,
    breaklines=true, captionpos=b, keepspaces=true,
    numbers=left, numbersep=5pt,
    showspaces=false, showstringspaces=false, showtabs=false,
    tabsize=2, frame=single, rulecolor=\color{bordergray}
}
\lstset{style=mystyle}
\newcommand{\code}[1]{\texttt{#1}}

\title{dummy}
\date{dummy}

\begin{document}
"""

TITLE = r"""\twocolumn[
\mlsystitle{KernelLens: Recovering the Triton-to-C++ Engine Contract}

\begin{mlsysauthorlist}
\mlsysauthor{Justin Duc}{galiad}
\end{mlsysauthorlist}

\mlsysaffiliation{galiad}{Galiad Research}
\mlsyscorrespondingauthor{Justin Duc}{justin.duc@galiad-research.com}

\vskip 0.3in

"""

# ---- 1. swap the preamble and the title/author block -----------------------
head, rest = text.split("\\begin{document}", 1)
# drop everything up to and including the old \maketitle
rest = rest.split("\\maketitle", 1)[1]
text = PREAMBLE + TITLE + rest

# close the \twocolumn[ ... ] bracket and emit the affiliation footnote
text = text.replace(
    "\\end{abstract}\n",
    "\\end{abstract}\n]\n\n\\printAffiliationsAndNotice{}\n\n",
    1,
)

# ---- 2. wide floats must span both columns --------------------------------
text = re.sub(r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure*}[t]", text)
text = re.sub(r"\\end\{figure\}", r"\\end{figure*}", text)

# tables were made non-floating for the 1-column report; restore them as
# full-width floats
def table_star(m):
    body, cap, lab = m.group(1).strip("\n"), m.group(2), m.group(3)
    return (f"\\begin{{table*}}[t]\n\\centering\n{body}\n"
            f"\\caption{{{cap}}}\n\\label{{{lab}}}\n\\end{{table*}}")

text, n_tables = re.subn(
    r"\\begin\{center\}\n(.*?)\\captionof\{table\}\{(.*?)\}\n\\label\{(.*?)\}\n\\end\{center\}",
    table_star, text, flags=re.S)

# ---- 3. anonymize ----------------------------------------------------------
text = text.replace(
    " and on GitHub (\\url{https://github.com/Just1truc/kernel_lens})", "")

# ---- 4. mandatory AI usage disclosure, after the references ---------------
DISCLOSURE = r"""
\section*{AI Usage Disclosure}

An AI coding assistant (DeepSeek~V4.1~Flash, driven through OpenCode) was used
extensively in producing this work, and the authors take full responsibility
for all content. It was used for: (i)~implementing and debugging parts of the
\texttt{kernel\_lens} package, including the PTX entry-signature recovery, the
ONNX graph rewriting and the TensorRT and ONNX~Runtime code generators;
(ii)~writing the evaluation and regression scripts under \texttt{tests/};
(iii)~producing the diagrams and result figures and the project website;
(iv)~executing the benchmarks on the GPU host and tabulating the measurements;
and (v)~drafting and editing the manuscript text, including the abstract, the
introduction, the related-work synthesis and the presentation of the results.
The literature search and the experimental design were performed by the author.
Every reported measurement was produced by executing the code on real hardware;
no result, citation or number was invented by a model. The author has verified
all claims, references and measurements.
"""

text = text.replace("\\end{thebibliography}", "\\end{thebibliography}\n" + DISCLOSURE, 1)

with open(DST, "w") as f:
    f.write(text)
print(f"wrote {DST} (converted {n_tables} tables to full-width floats)")
