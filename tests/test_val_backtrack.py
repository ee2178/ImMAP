"""
Validation backtracking (training/recon.py, `VAL_BACKTRACK_RATIO`).

Run with `python -m tests.test_val_backtrack`.

A validation pass whose loss exceeds `val_backtrack_ratio` x the best one's, or
is non-finite, is restored from the BEST-VALIDATION checkpoint
(`net_bestval.ckpt`) at a lower learning rate, and `net.ckpt` is rewritten with
the restored weights. `train_recon` is driven on a toy problem, validating
every epoch, with one validation pass's loss inflated by a known factor (the
training loss is left alone, so only the validation check can fire):

1. 3x on the last pass, ratio 2: restored -- the weights are the best-val
   checkpoint's, Adam's state is cleared, the reduction is counted, the bad pass
   did not overwrite the best-val checkpoint, and net.ckpt now holds the
   restored weights with the training loss recorded beside them;
2. 1.5x (under the ratio): not restored;
3. NEGATIVE CONTROL, ratio None: the 3x pass is not restored...
4. ...but a NON-FINITE validation loss is, whatever the ratio;
5. inflated on the FIRST pass: no best yet, so not restored and no error; a
   non-finite first pass falls back to net.ckpt;
6. RESUME: the bar is read back from net_bestval.ckpt; on a FRESH start a
   best-val checkpoint left in the directory is ignored and overwritten;
7. the defaults are 2 (validation) and 100 (training).

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
from training.recon import BESTVAL_CKPT, train_recon

FAIL = []
EPOCHS = 3
NVAL = 2                                   # validation slices per pass


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                       # torch < 1.13
        return torch.load(path, map_location="cpu")


def run(vfactor=1.0, vpass=None, d=None, start_epoch=0, epochs=EPOCHS, **kws):
    """Train with validation every epoch; validation pass `vpass` (0-based,
    counted within this call) has its loss multiplied by `vfactor`.
    -> dict(net, opt, log, ckpt, bestval) with both checkpoints as on disk."""
    torch.manual_seed(0)
    net = Toy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * STEPS,
                                                       eta_min=1e-3)
    vcalls = [0]
    base = LOSS_REGISTRY["complex-mse"]

    def loss_fn(x, y, sigma):
        loss = base(x, y, sigma)
        if torch.is_grad_enabled():                     # a training step
            return loss
        k = vcalls[0]
        vcalls[0] += 1
        return loss * vfactor if k // NVAL == vpass else loss

    LOSS_REGISTRY["test-val-inflated"] = loss_fn
    buf = io.StringIO()
    tmp = tempfile.TemporaryDirectory() if d is None else contextlib.nullcontext(d)
    try:
        with tmp as dd, contextlib.redirect_stdout(buf):
            train_recon(net, opt, sched, batches(6), batches(NVAL), torch.device("cpu"),
                        R=2, acs_lines=4, kspace_type="simulated",
                        noise_std=(0.0, 0.01), val_noise_std=0.005, val_seed=0,
                        loss_type="test-val-inflated", num_epochs=epochs,
                        steps_per_epoch=STEPS, clip_grad=1.0, val_every_epochs=1,
                        start_epoch=start_epoch, save_dir=dd, **kws)
            ckpt = _load(os.path.join(dd, "net.ckpt"))
            bestval = _load(os.path.join(dd, BESTVAL_CKPT))
    finally:
        del LOSS_REGISTRY["test-val-inflated"]
    return dict(net=net, opt=opt, log=buf.getvalue(), ckpt=ckpt, bestval=bestval)


def same(sd, net):
    return all(torch.equal(p.detach(), sd[n]) for n, p in net.named_parameters())


def lines(r, key="backtrack"):
    return [ln for ln in r["log"].splitlines() if key in ln]


def main():
    last = EPOCHS - 1
    end = EPOCHS * STEPS

    r = run(3.0, last, val_backtrack_ratio=2.0)
    check("3x the best val loss, ratio 2: restored from the best-val checkpoint",
          f"backtracking to {BESTVAL_CKPT}" in r["log"] and "val backtrack #1" in r["log"],
          lines(r)[:1])
    check("...the weights are the best-val checkpoint's, Adam's state is cleared",
          same(r["bestval"]["model_state_dict"], r["net"]) and len(r["opt"].state) == 0)
    check("...the bad pass did not overwrite the best-val checkpoint",
          r["bestval"]["step"] < end, f"best-val step {r['bestval']['step']} of {end}")
    check("...net.ckpt now holds the restored weights, the count, and that model's "
          "training loss",
          same(r["ckpt"]["model_state_dict"], r["net"]) and r["ckpt"]["step"] == end
          and r["ckpt"]["backtrack_count"] == 1
          and r["ckpt"]["best_loss"] == r["bestval"]["best_loss"]
          and r["ckpt"]["best_val_loss"] == r["bestval"]["best_val_loss"],
          f"net.ckpt best_loss {r['ckpt']['best_loss']}, best-val's "
          f"{r['bestval']['best_loss']}")

    mild = run(1.5, last, val_backtrack_ratio=2.0)
    check("1.5x the best val loss (under the ratio): not restored",
          "backtracking" not in mild["log"] and len(mild["opt"].state) > 0)

    off = run(3.0, last, val_backtrack_ratio=None)
    check("  control: ratio None, the 3x pass is not restored",
          "backtracking" not in off["log"] and len(off["opt"].state) > 0)
    nan = run(float("nan"), last, val_backtrack_ratio=None)
    check("...but a non-finite val loss is, whatever the ratio",
          "non-finite val loss" in nan["log"] and "val backtrack #1" in nan["log"]
          and same(nan["bestval"]["model_state_dict"], nan["net"]), lines(nan)[:1])

    first = run(3.0, 0, val_backtrack_ratio=2.0)
    check("inflated on the first pass, no best yet: not restored, no error",
          "backtracking" not in first["log"] and first["bestval"]["step"] > STEPS)
    nan0 = run(float("nan"), 0, val_backtrack_ratio=2.0)
    check("a non-finite FIRST pass falls back to net.ckpt (no best-val yet)",
          "backtracking to net.ckpt" in nan0["log"], lines(nan0)[:1])

    with tempfile.TemporaryDirectory() as d:
        a = run(d=d, epochs=2)
        bar = a["bestval"]["best_val_loss"]
        b = run(3.0, 0, d=d, start_epoch=2, epochs=3, val_backtrack_ratio=2.0)
        check("RESUME: the bar is read back from the best-val checkpoint...",
              f"resuming with best validation loss {bar:.4e}" in b["log"],
              lines(b, "resuming with best")[:1])
        check("...so a 3x pass right after the resume is caught",
              "val backtrack #1" in b["log"], lines(b)[:1])
        c = run(d=d, epochs=1)
        check("FRESH start: a best-val checkpoint left in the directory is ignored "
              "and overwritten",
              "resuming with best validation" not in c["log"]
              and c["bestval"]["step"] == STEPS, f"best-val step {c['bestval']['step']}")

    check("the defaults: validation 2, training 100",
          recon_mod.VAL_BACKTRACK_RATIO == 2.0 and recon_mod.BACKTRACK_RATIO == 100.0,
          f"{recon_mod.VAL_BACKTRACK_RATIO}, {recon_mod.BACKTRACK_RATIO}")

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
