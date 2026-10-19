"""
The relative backtrack threshold (training/recon.py, `BACKTRACK_RATIO`).

Run with `python -m tests.test_backtrack_ratio`.

An epoch whose average loss exceeds `backtrack_ratio` x the best epoch's is
restored from the checkpoint at a lower learning rate -- the case the loop used
to miss, a loss that goes large but stays FINITE. `train_recon` is driven on a
toy problem with one epoch's loss inflated by a known factor:

1. inflated 3x, ratio 2 (set explicitly): the epoch is restored -- the weights
   are the checkpoint's again, Adam's state is cleared, the best loss on disk
   is untouched, and the reduction is counted;
2. NEGATIVE CONTROL: the same run with the check off (`backtrack_ratio=None`)
   trains straight through it;
3. inflated 1.5x (under the ratio): not restored;
4. inflated in the FIRST epoch, before any checkpoint exists: not restored and
   no error -- there is no best to compare with yet;
5. the default is 100 (2 fired on the unstable early epochs of healthy runs;
   500 let two MG-Group cells diverge for good): a 3x epoch trains through it,
   a 600x one is restored; and the absolute margin (`backtrack_thresh`) still
   works on its own.

The VALIDATION check is tests/test_val_backtrack.py.

CPU only, a few seconds.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import torch

import training.recon as recon_mod
from tests.test_skip_nonfinite_step import STEPS, Toy, batches
from training.losses import LOSS_REGISTRY
from training.recon import train_recon

FAIL = []
EPOCHS = 3


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def run(factor, epoch, **kws):
    """Train EPOCHS epochs with `epoch`'s loss multiplied by `factor`.
    -> dict(net, opt, log, ckpt) with the checkpoint as left on disk."""
    torch.manual_seed(0)
    net = Toy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * STEPS,
                                                       eta_min=1e-3)
    calls = [0]
    base = LOSS_REGISTRY["complex-mse"]

    def loss_fn(x, y, sigma):
        k = calls[0]
        calls[0] += 1
        loss = base(x, y, sigma)
        return loss * factor if k // STEPS == epoch else loss

    LOSS_REGISTRY["test-inflated"] = loss_fn
    buf = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            train_recon(net, opt, sched, batches(6), None, torch.device("cpu"),
                        R=2, acs_lines=4, kspace_type="simulated",
                        noise_std=(0.0, 0.01), loss_type="test-inflated",
                        num_epochs=EPOCHS, steps_per_epoch=STEPS, clip_grad=1.0,
                        val_every_epochs=0, save_dir=d, **kws)
            path = os.path.join(d, "net.ckpt")
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:                               # torch < 1.13
                ckpt = torch.load(path, map_location="cpu")
    finally:
        del LOSS_REGISTRY["test-inflated"]
    return dict(net=net, opt=opt, log=buf.getvalue(), ckpt=ckpt)


def at_checkpoint(r):
    sd = r["ckpt"]["model_state_dict"]
    return all(torch.equal(p.detach(), sd[n]) for n, p in r["net"].named_parameters())


def main():
    last = EPOCHS - 1                     # inflate the final epoch: nothing trains after it

    r = run(3.0, last, backtrack_ratio=2.0)
    check("3x the best, ratio 2 set explicitly: the epoch is restored",
          "backtracking" in r["log"] and "x best" in r["log"] and "backtrack #1" in r["log"],
          [ln for ln in r["log"].splitlines() if "backtrack" in ln][:1])
    check("...the weights are the checkpoint's again and Adam's state is cleared",
          at_checkpoint(r) and len(r["opt"].state) == 0)
    check("...the best loss on disk is the earlier epoch's, not the inflated one",
          r["ckpt"]["step"] <= last * STEPS and r["ckpt"].get("backtrack_count", 0) == 0,
          f"checkpoint step {r['ckpt']['step']}, best_loss {r['ckpt'].get('best_loss'):.4f}")

    off = run(3.0, last, backtrack_ratio=None)
    check("  control: with the check off the same epoch trains straight through",
          "backtracking" not in off["log"] and not at_checkpoint(off)
          and len(off["opt"].state) > 0)

    mild = run(1.5, last, backtrack_ratio=2.0)
    check("1.5x the best (under the ratio): not restored",
          "backtracking" not in mild["log"] and not at_checkpoint(mild))

    first = run(3.0, 0, backtrack_ratio=2.0)
    check("inflated in the first epoch, no checkpoint yet: not restored, no error",
          "backtracking" not in first["log"] and first["ckpt"]["step"] > STEPS)

    check("the default ratio is 100",
          recon_mod.BACKTRACK_RATIO == 100.0,
          f"BACKTRACK_RATIO = {recon_mod.BACKTRACK_RATIO}")
    dflt = run(3.0, last)
    check("  ...so at the default a 3x epoch trains straight through",
          "backtracking" not in dflt["log"] and not at_checkpoint(dflt))
    blown = run(600.0, last)
    check("  ...and a 600x epoch (a blow-up) is restored",
          "backtracking" in blown["log"] and "100 x best" in blown["log"]
          and at_checkpoint(blown), str([ln for ln in blown["log"].splitlines() if "backtrack" in ln][:1]))

    margin = run(3.0, last, backtrack_ratio=None, backtrack_thresh=1e-6)
    check("the absolute margin still works on its own",
          "backtracking" in margin["log"] and "x best" not in margin["log"]
          and at_checkpoint(margin))

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
