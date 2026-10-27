#!/usr/bin/env python3
"""
Stack the rows of several make_figure.py outputs into ONE figure under one header.

    python figures/combine_rows.py --name mgwins_all \
        mgwins_brain_R8:"Brain, $R{=}8$" mgwins_brain_R12:"Brain, $R{=}12$" \
        mgwins_knee_R8:"Knee, $R{=}8$"   mgwins_knee_R12:"Knee, $R{=}12$"

make_figure.py draws one dataset per figure (one DUMP_ROOT, one set of runs),
so a figure with a row per task is built per task first and merged here. Each
argument is `<figures/out/ name>:<row label>`; every row of that figure gets the
label (use a one-row figure per task, e.g. scripts/pick_mg_wins.py -k 1). The
label REPLACES make_figure's "<subject>, slice <n>", so no subject numbers are
shown. The figures must share their columns (same headers) -- checked.

Writes figures/out/<name>/: <name>.tex (standalone) and imgs/ (copies of every
source PNG, prefixed by its figure, so the folder is self-contained for upload),
and <name>.pdf / .png when pdflatex / pdftoppm are found.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figure import find_tool                                    # noqa: E402

LABEL_RE = re.compile(r"(\\node\[rotate=90\]\{\\small )(.*?)(\};\\end\{tikzpicture\})")


def split(tex):
    """(preamble up to the tabular, the header row, [row blocks])."""
    head, rest = tex.split(r"\begin{tabular}", 1)
    spec, rest = rest.split("\n", 1)
    header, rest = rest.split("\n", 1)
    body = rest.split(r"\end{tabular}", 1)[0]
    rows = [r for r in re.split(r"(?<=\\\\\[2pt\])\n", body) if r.strip()]
    return head + r"\begin{tabular}" + spec, header, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parts", nargs="+", metavar="NAME:LABEL")
    p.add_argument("--name", required=True, help="output figure name")
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args(argv)

    out = os.path.join(HERE, "out", a.name)
    os.makedirs(os.path.join(out, "imgs"), exist_ok=True)
    head = header = None
    rows = []
    for part in a.parts:
        src, _, label = part.partition(":")
        sdir = os.path.join(HERE, "out", src)
        with open(os.path.join(sdir, src + ".tex"), encoding="utf-8") as f:
            h, hdr, rr = split(f.read())
        if header is None:
            head, header = h, hdr
        elif hdr != header:
            raise SystemExit(f"{src}: columns differ from the first figure's\n  {hdr}\n  {header}")
        for name in os.listdir(os.path.join(sdir, "imgs")):
            shutil.copy2(os.path.join(sdir, "imgs", name),
                         os.path.join(out, "imgs", f"{src}-{name}"))
        for r in rr:
            r = r.replace("{imgs/", "{imgs/" + src + "-")
            if label:
                r = LABEL_RE.sub(lambda m: m.group(1) + label + m.group(3), r, count=1)
            rows.append(r)
        print(f"{src}: {len(rr)} row(s)" + (f" labelled {label!r}" if label else ""))

    tex = os.path.join(out, a.name + ".tex")
    with open(tex, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([head, header] + rows + [r"\end{tabular}", r"\end{document}"]) + "\n")
    print(f"wrote {tex}")
    if a.no_compile:
        return 0
    pdflatex = find_tool("pdflatex")
    if not pdflatex:
        print("pdflatex not found: compile the .tex yourself")
        return 0
    subprocess.run([pdflatex, "-interaction=nonstopmode", a.name + ".tex"], cwd=out,
                   check=True, stdout=subprocess.DEVNULL)
    print(f"wrote {os.path.join(out, a.name + '.pdf')}")
    pdftoppm = find_tool("pdftoppm")
    if pdftoppm:
        subprocess.run([pdftoppm, "-png", "-r", "150", "-singlefile", a.name + ".pdf", a.name],
                       cwd=out, check=True)
        print(f"wrote {os.path.join(out, a.name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
