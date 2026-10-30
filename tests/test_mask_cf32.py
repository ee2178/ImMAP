"""
The exp18 cells (scripts/make_mg_recon_configs.py::EXP18): center_frac = 0.32 / R,
adjust_accel off, written over the measured protocol's mask.

Run with `python -m tests.test_mask_cf32`. Needs PyYAML (the generator writes configs);
no torch.

1. Every cf32 cell's config has the rule's mask at each R, in its own run dir.
2. A base cell's config still has the measured protocol's mask.
3. VarNet takes its ACS from the mask, and acs_lines is derived, as for any fraction.
4. Names are distinct from the base cell's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

FAIL = []
CELLS = ["varnetmaps", "lpdsnet", "ram", "mg81v6vtxAnestU", "mg169v6vtxAnestU"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main():
    with tempfile.TemporaryDirectory() as out:
        names = [c + "cf32" for c in CELLS] + CELLS
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_mg_recon_configs.py"),
                        "--out", out, "--anatomy", "knee", "--accels", "8", "12", "--only", *names],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

        def cfg(cell, r):
            with open(os.path.join(out, "knee", "mg", f"{cell}_R{r}.json")) as f:
                return json.load(f)

        for c in CELLS:
            for r in (8, 12):
                a, b = cfg(c + "cf32", r), cfg(c, r)
                m = a["mri"]
                check(f"{c} R{r}: center_frac 0.32/R, adjust off",
                      abs(m["center_frac"] - 0.32 / r) < 1e-12 and m["adjust_accel"] is False
                      and m["R"] == r and m["acs_lines"] is None, str(m["center_frac"]))
                check(f"{c} R{r}: the base keeps the measured mask",
                      b["mri"]["center_frac"] == 0.04 and b["mri"]["adjust_accel"] is True)
                check(f"{c} R{r}: its own run dir and name",
                      a["paths"]["save_dir"].endswith(f"/{c}cf32_R{r}")
                      and a["experiment"]["name"] != b["experiment"]["name"])
                drop = lambda x: {k: v for k, v in x.items() if k not in ("center_frac", "adjust_accel")}
                check(f"{c} R{r}: nothing else in mri differs", drop(m) == drop(b["mri"]))
                check(f"{c} R{r}: same model", a["model"] == b["model"])
        check("VarNet counts its ACS from the mask",
              cfg("varnetmapscf32", 8)["model"]["params"]["acs_lines"] == "mask")
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
