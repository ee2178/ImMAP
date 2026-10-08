"""
`training.planar_state`: the config key behind `MGLPDSNet.PLANAR_STATE`.

Run with `python -m tests.test_planar_state_config`.

What the planar state DOES is tests/test_planar_state.py. This is the plumbing
that gets a config's value to the class switch, and keeps it from disturbing
anything that does not ask for it:

1. `set_planar_state`: None leaves the default, a bool sets it, the
   IMMAP_PLANAR_STATE override wins over the config, and anything that is not
   plainly on or off is refused;
2. the generator: without --planar-state NO config mentions the key (so every
   run dir launched before it existed still matches); with it, the key is
   written on exactly the cells whose net can use it, and nothing else in any
   config moves. The generator's rule is checked against the nets themselves;
3. the launch guard in torch/_mg_recon_body.sh treats the key as execution,
   not setup: a cell launched without it is recognised (not refused) when the
   generator now writes it, and the other way round. NEGATIVE CONTROL: a real
   difference in the same config is still refused, and named.

CPU only; needs the repo root as the working directory (as the launcher does).
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.mg_lpds import (MGLPDSNet, PLANAR_STATE_ENV,                  # noqa: E402
                            planar_state_override, planar_state_report,
                            set_planar_state)

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


class env:
    """IMMAP_PLANAR_STATE set (or removed, for None) inside the block."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.keep = os.environ.pop(PLANAR_STATE_ENV, None)
        if self.value is not None:
            os.environ[PLANAR_STATE_ENV] = self.value

    def __exit__(self, *exc):
        os.environ.pop(PLANAR_STATE_ENV, None)
        if self.keep is not None:
            os.environ[PLANAR_STATE_ENV] = self.keep


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------------------
def test_switch():
    keep = MGLPDSNet.PLANAR_STATE
    try:
        with env(None):
            MGLPDSNet.PLANAR_STATE = False
            check("None (key absent) leaves the class default",
                  set_planar_state(None) is False and MGLPDSNet.PLANAR_STATE is False
                  and planar_state_override() is None)
            check("true switches it on, false off, and the value in force is returned",
                  set_planar_state(True) is True and MGLPDSNet.PLANAR_STATE is True
                  and set_planar_state(None) is True
                  and set_planar_state(False) is False and MGLPDSNet.PLANAR_STATE is False)
            check("a value that is not a bool is refused (\"true\", 1)",
                  raises(lambda: set_planar_state("true")) and raises(lambda: set_planar_state(1))
                  and MGLPDSNet.PLANAR_STATE is False)
        with env("1"):
            check("IMMAP_PLANAR_STATE=1 overrides an absent key and an explicit false",
                  planar_state_override() is True and set_planar_state(None) is True
                  and set_planar_state(False) is True)
        with env("0"):
            check("IMMAP_PLANAR_STATE=0 overrides an explicit true",
                  planar_state_override() is False and set_planar_state(True) is False)
        with env("yes"):
            check("an override that is neither 1 nor 0 is refused",
                  raises(planar_state_override) and raises(lambda: set_planar_state(None)))
    finally:
        MGLPDSNet.PLANAR_STATE = keep


# ---------------------------------------------------------------------------
def generate(out, *extra):
    subprocess.run([sys.executable, "scripts/make_mg_recon_configs.py", "--out", out,
                    "--anatomy", "brain", "--organ-mask", *extra],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    cfgs = {}
    d = os.path.join(out, "brain", "mg")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                cfgs[f[:-5]] = json.load(fh)
    return cfgs


def flat(d, p=""):
    if not isinstance(d, dict):
        return {p: d}
    acc = {}
    for k, v in d.items():
        acc.update(flat(v, f"{p}.{k}" if p else k))
    return acc


def test_generator(tmp):
    off = generate(os.path.join(tmp, "off"))
    on = generate(os.path.join(tmp, "on"), "--planar-state")
    check("without --planar-state no config mentions the key",
          len(off) > 20 and not any("training.planar_state" in flat(c) for c in off.values()),
          f"{len(off)} configs")
    have = sorted(n for n, c in on.items() if c["training"].get("planar_state") is True)
    tags = sorted({n.rsplit("_R", 1)[0] for n in have})
    expect = {"lpdsnet", "mglpds", "mg169v6", "mg121v6", "mg100v6", "mg81v6", "mgunet"}
    never = {"varnetmaps", "varnet", "mggrouplpds"}
    check("--planar-state writes it on the LPDS and multigrid-LPDS cells, every R",
          expect <= set(tags) and not (never & set(tags))
          and all(f"{t}_R{r}" in have for t in ("lpdsnet", "mg81v6") for r in (4, 8, 12, 16)),
          f"{len(have)} configs: {' '.join(tags)}")
    moved = {k for n in on for k in set(flat(on[n])) | set(flat(off[n]))
             if flat(on[n]).get(k) != flat(off[n]).get(k)}
    check("...and it is the ONLY key that differs, in any config",
          on.keys() == off.keys() and moved == {"training.planar_state"}, f"{sorted(moved)}")

    # the generator's rule against the nets themselves
    from models import build_model
    keep = MGLPDSNet.PLANAR_STATE
    MGLPDSNet.PLANAR_STATE = True
    try:
        for tag in ("lpdsnet", "mglpds", "mg81v6", "mggrouplpds", "varnetmaps"):
            cfg = on[f"{tag}_R12"]
            try:
                net = build_model(copy.deepcopy(cfg))
            except Exception as e:                                # noqa: BLE001
                print(f"[skip] {tag}: could not build here ({type(e).__name__})")
                continue
            nets = [m for m in net.modules() if isinstance(m, MGLPDSNet)]
            active = bool(nets) and all(m.planar_state_active() for m in nets)
            check(f"{tag}: key written <=> the net runs the planar state",
                  active == bool(cfg["training"].get("planar_state"))
                  and ("ON" in planar_state_report(net)) == active,
                  planar_state_report(net))
    finally:
        MGLPDSNet.PLANAR_STATE = keep
    return off, on


# ---------------------------------------------------------------------------
def launch_guard():
    """The launcher's per-cell bookkeeping script, as a callable.
    -> f(base, out, save_dir) returning None (config written), 3 (launched
    before under this config) or the refusal message."""
    with open(os.path.join(ROOT, "torch", "_mg_recon_body.sh"), encoding="utf-8") as f:
        body = f.read().replace("\r\n", "\n")
    m = re.search(r'"\$\{SWEEP_EPOCHS\}" <<\'PY\'\n(.*?)\nPY\n', body, re.S)
    assert m, "could not find the launch-guard heredoc in torch/_mg_recon_body.sh"
    code = compile(m.group(1), "_mg_recon_body.sh:<launch guard>", "exec")

    def run(base, out, save_dir):
        argv, cwd = sys.argv, os.getcwd()
        keep = {k: os.environ.pop(k, None)
                for k in ("RUN_TAG", "FORCE_RESTART", "ONLINE_SMAPS_KWS", "VAL_EVERY")}
        sys.argv = ["-", base, out, save_dir, ""]
        os.chdir(ROOT)
        try:
            exec(code, {"__name__": "__guard__"})
            return None
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = argv
            os.chdir(cwd)
            for k, v in keep.items():
                if v is not None:
                    os.environ[k] = v
    return run


def test_launch_guard(tmp, off, on):
    guard = launch_guard()

    def write(name, cfg):
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            json.dump(cfg, f)
        return path

    plain, planar = write("plain.json", off["lpdsnet_R12"]), write("planar.json", on["lpdsnet_R12"])
    changed = copy.deepcopy(on["lpdsnet_R12"])
    changed["training"]["clip_grad"] = 2.0
    changed = write("changed.json", changed)

    for first, again, tag in ((plain, planar, "launched WITHOUT the key, generator now writes it"),
                              (planar, plain, "launched WITH the key, generator no longer writes it")):
        run_dir = tempfile.mkdtemp(dir=tmp)
        gen = os.path.join(run_dir, "config.gen.json")
        r0 = guard(first, gen, run_dir)
        with open(gen) as f:
            before = f.read()
        r1 = guard(first, gen, run_dir)
        r2 = guard(again, gen, run_dir)
        with open(gen) as f:
            after = f.read()
        check(f"{tag}: recognised as the same launch, not refused",
              r0 is None and r1 == 3 and r2 == 3 and before == after,
              f"first {r0!r}, same config {r1!r}, other state {r2!r}")
        r3 = guard(changed, gen, run_dir)
        check("  control: a real difference in that config is still refused, and named",
              isinstance(r3, str) and "training.clip_grad" in r3
              and "planar_state" not in r3, str(r3)[:110])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print("\n--- test_switch")
        test_switch()
        print("\n--- test_generator")
        off, on = test_generator(tmp)
        print("\n--- test_launch_guard")
        test_launch_guard(tmp, off, on)
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
