#!/usr/bin/env python3
"""
What differs between two (or more) trained runs, beyond the one thing you meant to change?

    python scripts/diff_runs.py RUN_A RUN_B [RUN_C ...]

Prints every config.json key whose value differs across the runs (nested keys dotted), then, from
each run's checkpoint, facts a config does not record because they were CODE defaults at the time:

  tied W's     whether the LGGS layers share W_theta / W_phi / W_alpha / W_beta / gamma (code from
               2026-11-09 on) or each layer has its own (runs trained before)
  tau degree   the prox threshold's polynomial degree in sigma (noise-adaptive threshold: 1)
  params       the checkpoint's parameter count (shared tensors once)
  step         the training step the checkpoint was saved at, when recorded

Use it before reading a difference in wandb as the effect of the one change you made (guide R,
separate guide W's, ...): a run tag, noise preset, LR or code-version difference is just as
likely to explain a small gap.
"""

import json
import os
import sys

import torch

CKPTS = ("net_bestval.ckpt", "net.ckpt")
TIED = ("Wtheta", "Wphi", "Walpha", "Wbeta", "gamma")


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def ckpt_facts(run_dir):
    path = next((os.path.join(run_dir, c) for c in CKPTS
                 if os.path.exists(os.path.join(run_dir, c))), None)
    if path is None:
        return {"ckpt": "(none)"}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or {}
    facts = {"ckpt": os.path.basename(path)}
    for k in ("step", "global_step", "epoch"):
        if k in ck:
            facts[k] = ck[k]
    seen, n = set(), 0
    for v in sd.values():
        if torch.is_tensor(v) and v.data_ptr() not in seen:
            seen.add(v.data_ptr())
            n += v.numel()
    facts["params (state dict, shared once)"] = n
    # LGGS layers: are the W's the same in every layer?
    by_tail = {}
    for k, v in sd.items():
        parts = k.split(".")
        if "layers" in parts and any(w in parts for w in TIED):
            i = parts.index("layers")
            by_tail.setdefault(".".join(parts[i + 2:]), []).append(v)
    if by_tail:
        tied = all(all(torch.equal(t[0], x) for x in t[1:]) for t in by_tail.values())
        facts["tied W's / gamma"] = "yes" if tied else "NO (trained untied: code before 2026-11-09)"
    tau = [v for k, v in sd.items() if k.endswith("prox.prox.tau.weight")]
    if tau:
        facts["prox tau degree"] = int(tau[0].shape[0]) - 1
    return facts


def main(runs):
    cfgs = []
    for r in runs:
        with open(os.path.join(r, "config.json")) as f:
            cfgs.append(flat(json.load(f)))
    names = [os.path.basename(os.path.normpath(r)) for r in runs]
    w = max(len(n) for n in names)
    keys = sorted(set().union(*cfgs))
    print("== config.json differences")
    shown = 0
    for k in keys:
        vals = [c.get(k, "(absent)") for c in cfgs]
        if any(v != vals[0] for v in vals[1:]):
            shown += 1
            print(f"  {k}")
            for n, v in zip(names, vals):
                print(f"      {n:{w}s}  {v!r}")
    if not shown:
        print("  (none)")
    print("\n== from the checkpoints")
    facts = [ckpt_facts(r) for r in runs]
    for k in sorted(set().union(*facts)):
        vals = [f.get(k, "-") for f in facts]
        flag = "   <-- differs" if any(v != vals[0] for v in vals[1:]) else ""
        print(f"  {k}{flag}")
        for n, v in zip(names, vals):
            print(f"      {n:{w}s}  {v}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
