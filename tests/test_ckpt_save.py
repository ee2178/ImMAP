"""training/common.py::save_ckpt under a failing filesystem.

    python tests/test_ckpt_save.py

  * atomic: a write that fails part-way leaves the previous checkpoint loadable, no temp file
  * a FULL filesystem (ENOSPC / torch's "unexpected pos" / "No space left") is retried with
    backoff and succeeds once space returns
  * if it never returns, the save is SKIPPED (returns False, warns) -- training goes on with the
    previous checkpoint intact -- instead of raising
  * any other error is raised at once, without retrying
"""

import errno
import os
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import training.common as tc                       # noqa: E402


def _setup(d):
    net = torch.nn.Linear(3, 3)
    p = os.path.join(d, "net.ckpt")
    assert tc.save_ckpt(p, model=net, step=1) is True
    return net, p


def _failing_save(fails, exc):
    """torch.save that writes half a file and raises `exc` the first `fails` times."""
    real, n = torch.save, {"calls": 0}

    def fake(obj, f, *a, **k):
        n["calls"] += 1
        if n["calls"] <= fails:
            with open(f, "wb") as fh:
                fh.write(b"half a checkpoint")
            raise exc
        return real(obj, f, *a, **k)
    return fake, n, real


def _run(fails, exc):
    """A good checkpoint at step 1, then a save of step 2 against a filesystem that fails
    `fails` times with `exc`. -> (result or exception, step on disk, leftovers, waits, calls)."""
    sleeps = []
    with tempfile.TemporaryDirectory() as d:
        net, p = _setup(d)                                           # with the real torch.save
        fake, n, real = _failing_save(fails, exc)
        tc._sleep, real_sleep = sleeps.append, tc._sleep
        torch.save = fake
        try:
            try:
                out = tc.save_ckpt(p, model=net, step=2)
            except Exception as e:                                   # noqa: BLE001
                out = e
        finally:
            torch.save, tc._sleep = real, real_sleep
        step = torch.load(p, map_location="cpu", weights_only=False)["step"]
        leftovers = [f for f in os.listdir(d) if "partial" in f]
    return out, step, leftovers, sleeps, n["calls"]


def test_recovers_when_space_returns():
    out, step, left, sleeps, calls = _run(2, OSError(errno.ENOSPC, "No space left on device"))
    assert out is True and step == 2 and not left, (out, step, left)
    assert sleeps == list(tc.SAVE_RETRY_DELAYS[:2]) and calls == 3      # 2 failures, 1 success


def test_torch_zip_writer_error_counts_as_full():
    out, step, *_ = _run(1, RuntimeError("[enforce fail at inline_container.cc:672] . "
                                         "unexpected pos 81880960 vs 81880912"))
    assert out is True and step == 2


def test_skips_when_space_never_returns():
    n = len(tc.SAVE_RETRY_DELAYS) + 1
    out, step, left, sleeps, calls = _run(n, OSError(errno.ENOSPC, "No space left on device"))
    assert out is False, out                                         # skipped, not raised
    assert step == 1 and not left                                    # previous one intact
    assert sleeps == list(tc.SAVE_RETRY_DELAYS)


def test_other_errors_raise_at_once():
    out, step, left, sleeps, _ = _run(1, ValueError("not a disk problem"))
    assert isinstance(out, ValueError) and step == 1 and not left and sleeps == []


def main():
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                       # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc(limit=4)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
