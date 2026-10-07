"""
Per-phase wall clock for the first N training steps.

Opt-in: `PROFILE_STEPS=300` in the environment (0 / unset = off, and then every
call here returns immediately -- no sync, no overhead).

Why it exists: tqdm's it/s lumps the data loader, the online coil-map estimate,
the network, the projection and the optimizer into one number, and on a GPU an
unsynchronised timer charges a phase's work to whichever later call happens to
block. So "is the variation the net or the loop?" cannot be read off the
progress bar. Here every phase boundary calls `torch.cuda.synchronize()`, which
is exactly why it is limited to N steps: the syncs themselves slow training.

Phases of one step
    data      waiting on the loader + host->GPU copies
    maps      the online coil-map estimate (physics.online_smaps), timed inside
              prepare_measurement by wrapping the estimator
    measure   the rest of prepare_measurement (noise, mask) + building E
    forward   net(y, E, sigma) with autograd
    backward  loss + loss.backward()
    opt       grad clip + optimizer.step() + scheduler.step()
    project   net.project()
and, outside the step total,
    infer     ONE extra eval-mode, no-grad forward on the same batch -- the
              inference time, with nothing of the loop in it

The report splits by image shape and by whether the operator was EMBEDDED (a
measured size that is not a multiple of the model's stride gets `E @ Truncate`;
the rediscretized coarse Gram covers it, on the measured grid's half and
quarter), and separates the FIRST step at each shape (cuDNN autotunes every
conv for a new shape).
"""

from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict

import torch

PHASES = ("data", "maps", "measure", "forward", "backward", "opt", "project")


class StepProfiler:
    def __init__(self, device, n=None):
        if n is None:
            n = int(os.environ.get("PROFILE_STEPS", "0") or 0)
        self.n = int(n)
        self.on = self.n > 0
        self.cuda = torch.device(device).type == "cuda"
        self.rows = []
        self._seen = set()
        self._restore = None
        if self.on:
            print(f"[profile] timing the first {self.n} training steps per phase "
                  f"(GPU-synchronised, so these steps run slower than normal)")
            self._wrap_maps()

    # -- timing primitives ---------------------------------------------------
    def _now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self):
        if not self.on:
            return
        self.cur = defaultdict(float)
        self.t = self._now()

    def mark(self, phase):
        if not self.on:
            return
        now = self._now()
        self.cur[phase] += now - self.t
        self.t = now

    def _wrap_maps(self):
        """Time the online map estimate from inside prepare_measurement.

        prepare_measurement re-imports `physics.online_smaps.online_smaps` on
        every call, so swapping the module attribute is enough; the time is
        moved from `measure` (where the caller's mark would put it) to `maps`.
        """
        import physics.online_smaps as om
        real = om.online_smaps

        def timed(*a, **k):
            t0 = self._now()
            out = real(*a, **k)
            dt = self._now() - t0
            self.cur["maps"] += dt
            self.cur["measure"] -= dt
            return out

        om.online_smaps = timed
        self._restore = lambda: setattr(om, "online_smaps", real)

    # -- per step ------------------------------------------------------------
    def infer(self, net, y, E, sigma):
        """One eval-mode no-grad forward on this batch; not part of the step."""
        if not self.on:
            return
        was_training = net.training
        net.eval()
        with torch.no_grad():
            t0 = self._now()
            net(y, E=E, sigma=sigma)
            self.cur["infer"] = self._now() - t0
        net.train(was_training)
        self.t = self._now()            # keep the eval forward out of `data`

    def end(self, shape, embedded=False):
        if not self.on:
            return
        key = (tuple(int(s) for s in shape), bool(embedded))
        row = dict(self.cur)
        row["total"] = sum(row.get(p, 0.0) for p in PHASES)
        row["shape"], row["first"] = key, key not in self._seen
        self._seen.add(key)
        self.rows.append(row)
        if len(self.rows) >= self.n:
            self.report()
            self.on = False
            if self._restore:
                self._restore()

    # -- report --------------------------------------------------------------
    @staticmethod
    def _stats(v):
        v = sorted(v)
        if not v:
            return 0.0, 0.0, 0.0, 0.0
        p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
        return statistics.fmean(v), statistics.median(v), p95, v[-1]

    def report(self):
        ms = 1e3
        warm = [r for r in self.rows if not r["first"]] or self.rows
        tot_mean = statistics.fmean(r["total"] for r in warm)
        print(f"\n[profile] ===== {len(self.rows)} steps; {len(warm)} after the "
              f"first at each shape =====")
        print(f"[profile] {'phase':<10}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}"
              f"{'std':>9}{'share':>8}   (ms)")
        for p in PHASES + ("total", "infer"):
            v = [r.get(p, 0.0) * ms for r in warm]
            mean, p50, p95, mx = self._stats(v)
            std = statistics.pstdev(v) if len(v) > 1 else 0.0
            share = "" if p in ("total", "infer") else f"{100 * mean / (tot_mean * ms):>7.1f}%"
            print(f"[profile] {p:<10}{mean:>9.1f}{p50:>9.1f}{p95:>9.1f}{mx:>9.1f}"
                  f"{std:>9.1f}{share:>8}")
        print("[profile] `std` says which phase carries the step-to-step variation; "
              "`infer` is one eval-mode no-grad forward, outside the step total.")

        by = defaultdict(list)
        for r in warm:
            by[r["shape"]].append(r)
        print(f"\n[profile] by image shape (p50 ms; embedded = E @ Truncate, a measured "
              f"size that is not a multiple of the stride)")
        print(f"[profile] {'shape':<14}{'emb':>5}{'n':>6}{'total':>9}{'maps':>9}"
              f"{'forward':>9}{'backward':>10}{'infer':>9}{'infer p95':>11}")
        for (hw, emb), rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            col = lambda p: statistics.median(r.get(p, 0.0) * ms for r in rs)  # noqa: E731
            inf = sorted(r.get("infer", 0.0) * ms for r in rs)
            p95 = inf[min(len(inf) - 1, int(round(0.95 * (len(inf) - 1))))]
            print(f"[profile] {str(hw[0]) + 'x' + str(hw[1]):<14}{'yes' if emb else 'no':>5}"
                  f"{len(rs):>6}{col('total'):>9.1f}{col('maps'):>9.1f}{col('forward'):>9.1f}"
                  f"{col('backward'):>10.1f}{col('infer'):>9.1f}{p95:>11.1f}")

        first = [r for r in self.rows if r["first"]]
        if first and len(warm) < len(self.rows):
            f_fwd = statistics.fmean(r.get("forward", 0.0) * ms for r in first)
            w_fwd = statistics.fmean(r.get("forward", 0.0) * ms for r in warm)
            print(f"\n[profile] first step at each of {len(first)} shape(s): forward "
                  f"{f_fwd:.0f} ms vs {w_fwd:.0f} ms afterwards (cuDNN autotuning, "
                  f"lazy kernel compiles)")
        print("[profile] ===== profiling done; training continues unsynchronised =====\n",
              flush=True)
