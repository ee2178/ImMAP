"""
A training step whose loss or gradient is NaN / inf is SKIPPED, not applied
(training/recon.py, `MAX_SKIPPED_FRACTION`).

Run with `python -m tests.test_skip_nonfinite_step`.

`train_recon` is driven end to end on a toy problem, with a NaN injected at a
chosen step -- into the loss, or into the gradient alone -- and checked for:

1. the weights and Adam's state are exactly what they were before the bad
   step (nothing of it was applied), and stay finite to the end;
2. the epoch average is over the applied steps only, so the epoch is NOT
   treated as non-finite and no restore happens;
3. the LR schedule still advances once per step, skipped or not;
4. a clean run skips nothing (negative control);
5. more than MAX_SKIPPED_FRACTION of an epoch skipped still hands the epoch to
   the existing restore-and-lower-the-LR path.

CPU only, a few seconds.
"""

from __future__ import annotations

import contextlib
import io
import tempfile

import torch
import torch.nn as nn

import training.recon as recon_mod
from training.losses import LOSS_REGISTRY
from training.recon import train_recon

FAIL = []
H, W, COILS, STEPS = 16, 12, 3, 10


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


class Toy(nn.Module):
    """recon = w * E^H y: one complex-safe scalar, enough for a gradient."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.5]))
        self.b = nn.Parameter(torch.tensor([0.1]))

    def forward(self, y, E=None, sigma=None):
        return (self.w + 0.0 * self.b) * E.adjoint(y) + self.b * 0.01, None


def batches(n):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        smaps = torch.randn(1, COILS, H, W, dtype=torch.complex64, generator=g)
        smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
        image = torch.randn(1, 1, H, W, dtype=torch.complex64, generator=g)
        kspace = torch.zeros(1, COILS, H, W, dtype=torch.complex64)
        out.append((kspace, smaps, image, torch.ones(1, 1, H, W), torch.tensor([[H, W]])))
    return out


def run(bad_loss=(), bad_grad=(), epochs=2):
    """-> (net, opt, sched, log, snapshots). `bad_*` are global step indices."""
    torch.manual_seed(0)
    net = Toy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * STEPS)
    calls, snaps, applied = [0], {}, []
    base = LOSS_REGISTRY["complex-mse"]

    def loss_fn(x, y, sigma):
        k = calls[0]
        calls[0] += 1
        # the state every step STARTS from: step k's snapshot is taken before
        # its update, step k + 1's after it
        snaps[k] = ([p.detach().clone() for p in net.parameters()],
                    {i: int(opt.state[p]["step"]) if p in opt.state else 0
                     for i, p in enumerate(net.parameters())})
        loss = base(x, y, sigma)
        if k in bad_loss:
            return loss * float("nan")
        if k in bad_grad:
            # finite loss, NaN gradient
            return loss + (net.w * 0.0).sum().pow(0.5) * 0.0
        applied.append(float(loss))
        return loss

    LOSS_REGISTRY["test-injected"] = loss_fn
    buf = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
            train_recon(net, opt, sched, batches(6), None, torch.device("cpu"),
                        R=2, acs_lines=4, kspace_type="simulated",
                        noise_std=(0.0, 0.01), loss_type="test-injected",
                        num_epochs=epochs, steps_per_epoch=STEPS, clip_grad=1.0,
                        val_every_epochs=0, save_dir=d)
    finally:
        del LOSS_REGISTRY["test-injected"]
    return net, opt, sched, buf.getvalue(), snaps, applied


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_clean():
    net, opt, sched, log, snaps, applied = run()
    steps = {int(opt.state[p]["step"]) for p in net.parameters()}
    check("clean run: nothing skipped, every step applied",
          "skipped" not in log and "backtracking" not in log and steps == {2 * STEPS}
          and not same(snaps[0][0], snaps[1][0]), f"Adam steps {steps}")


def test_skip(kind, **kw):
    k = 13                                               # epoch 1, step 3
    net, opt, sched, log, snaps, applied = run(**{kw["where"]: (k,)})
    finite = all(bool(torch.isfinite(p).all()) for p in net.parameters())
    adam_ok = all(bool(torch.isfinite(v).all()) for s in opt.state.values()
                  for v in s.values() if torch.is_tensor(v))
    steps = {int(opt.state[p]["step"]) for p in net.parameters()}
    check(f"{kind}: the weights after the bad step are exactly those before it",
          same(snaps[k][0], snaps[k + 1][0]) and snaps[k][1] == snaps[k + 1][1]
          and not same(snaps[k - 1][0], snaps[k][0])
          and not same(snaps[k + 1][0], snaps[k + 2][0]))
    check(f"{kind}: weights and Adam's moments stay finite; one update fewer",
          finite and adam_ok and steps == {2 * STEPS - 1}, f"Adam steps {steps}")
    check(f"{kind}: reported, and the epoch is NOT restored",
          f"skipped 1 of {STEPS} steps" in log and "backtracking" not in log,
          [ln for ln in log.splitlines() if "skipped" in ln][:1])
    avg = [ln for ln in log.splitlines() if "'epoch': 1" in ln]
    want = sum(applied[STEPS:]) / (STEPS - 1)
    got = float(avg[0].split("'avg_loss': ")[1].split(",")[0].rstrip("}")) if avg else float("nan")
    check(f"{kind}: the epoch average is over the {STEPS - 1} applied steps",
          abs(got - want) < 1e-5 * max(1.0, abs(want)), f"{got:.6f} vs {want:.6f}")
    check(f"{kind}: the schedule advanced once per step, skipped or not",
          sched.last_epoch == 2 * STEPS, f"last_epoch {sched.last_epoch}")


def test_too_many():
    n = int(recon_mod.MAX_SKIPPED_FRACTION * STEPS) + 1
    bad = tuple(range(STEPS + 2, STEPS + 2 + n))          # all in epoch 1
    net, opt, sched, log, snaps, applied = run(bad_loss=bad)
    finite = all(bool(torch.isfinite(p).all()) for p in net.parameters())
    check(f"{n} of {STEPS} steps skipped (> {recon_mod.MAX_SKIPPED_FRACTION:.0%}): "
          f"the epoch goes to the restore, at a lower LR",
          f"skipped {n} of {STEPS} steps" in log and "backtracking" in log
          and "backtrack #1" in log and finite,
          [ln for ln in log.splitlines() if "backtrack" in ln][:2])
    one = int(recon_mod.MAX_SKIPPED_FRACTION * STEPS)
    if one >= 1:
        _, _, _, log1, _, _ = run(bad_loss=tuple(range(STEPS + 2, STEPS + 2 + one)))
        check(f"  control: {one} of {STEPS} (at the limit) is not restored",
              "backtracking" not in log1)


def main():
    print("\n--- clean")
    test_clean()
    print("\n--- NaN loss")
    test_skip("NaN loss", where="bad_loss")
    print("\n--- NaN gradient, finite loss")
    test_skip("NaN gradient", where="bad_grad")
    print("\n--- too many")
    test_too_many()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
