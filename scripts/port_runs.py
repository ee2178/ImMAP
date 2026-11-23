"""Make run directories copied from another cluster loadable here: point their configs' k-space
roots at THIS cluster's paired knee data.

A run directory is self-contained (config.json + net.ckpt / net_bestval.ckpt), so moving a trained
net between torch and bigpurple is a copy. The one thing that differs between the clusters is
where the raw k-space lives (scripts/make_guided_recon_configs.py::KNEE_ROOTS; the maps, scales and
splits are at the same relative paths on both). This rewrites data.{train,val}.kspace_roots in each
run's config.json to the target cluster's roots, keeping the original as config.<from>.json. Nothing
else changes, so the nets load exactly as trained.

    # on bigpurple, after copying the run dirs over (see the docstring's end)
    python scripts/port_runs.py trained_nets/guided_recon/knee/s4_noise-vlow/{lpdsnet_R8,tgvn_c6_R8_g1,lggs_R8_g1,lggs_guide_w_R8_g1,lggs_guide_conv_R8_g1}

Copying torch -> bigpurple (neither cluster reaches the other directly; stream through a machine
that reaches both, e.g. a laptop on the NYU VPN, with plain ssh + tar):

    ssh torch "cd /home/ee2178/scratch/ee2178/ImMAP && tar czf - <run dirs>" \\
        | ssh bigpurple "cd /gpfs/scratch/ee2178/ImMAP && tar xzf -"
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.make_guided_recon_configs import KNEE_ROOTS  # noqa: E402


def port(run_dir, to, frm):
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{run_dir}: no config.json")
    with open(path) as f:
        cfg = json.load(f)
    changed = []
    for split in ("train", "val"):
        d = cfg.get("data", {}).get(split)
        if d is None or "kspace_roots" not in d:
            continue
        if d["kspace_roots"] != KNEE_ROOTS[to]:
            changed.append(f"{split}: {d['kspace_roots']} -> {KNEE_ROOTS[to]}")
            d["kspace_roots"] = list(KNEE_ROOTS[to])
    if not changed:
        print(f"{run_dir}: already on {to}'s roots")
        return
    backup = os.path.join(run_dir, f"config.{frm}.json")
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    ckpts = [c for c in ("net.ckpt", "net_bestval.ckpt") if os.path.isfile(os.path.join(run_dir, c))]
    print(f"{run_dir}: {'; '.join(changed)} (original kept as {os.path.basename(backup)}; "
          f"checkpoints: {', '.join(ckpts) or 'NONE'})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--to", default=os.environ.get("IMMAP_CLUSTER", "bigpurple"),
                   choices=sorted(KNEE_ROOTS), help="the cluster the runs are now on")
    p.add_argument("--from", dest="frm", default="torch", choices=sorted(KNEE_ROOTS))
    a = p.parse_args()
    for r in a.run_dirs:
        port(r.rstrip("/"), a.to, a.frm)


if __name__ == "__main__":
    main()
