"""
Build the arXiv and TMLR submission archives from the rewritten sources.

Run:  python3 scripts/make_submissions.py
"""
import os
import glob
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "docs", "assets", "figures")


def build(zip_path, files):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, arc)
    print("wrote", zip_path, f"({len(files)} files)")


def main():
    figures = sorted(glob.glob(os.path.join(FIGS, "*.png")))

    arxiv_files = [(os.path.join(ROOT, "architecture_report.tex"), "architecture_report.tex")]
    arxiv_files += [(f, os.path.basename(f)) for f in figures]
    build(os.path.join(ROOT, "kernel_lens_arxiv_submission.zip"), arxiv_files)

    tmlr_files = [(os.path.join(ROOT, "kernel_lens_tmlr.tex"), "kernel_lens_tmlr.tex"),
                  (os.path.join(ROOT, "tmlr.sty"), "tmlr.sty")]
    tmlr_files += [(f, os.path.basename(f)) for f in figures]
    build(os.path.join(ROOT, "kernel_lens_tmlr_submission.zip"), tmlr_files)


if __name__ == "__main__":
    main()