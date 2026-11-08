"""
`torch.compile` for small PURE pieces of the unrolled nets -- the elementwise glue between the
convs and the custom kernels -- and nothing else.

Why only pieces
---------------
Compiling a whole LGGS layer would meet everything dynamo handles worst at once: Python state
mutated per call (the adjacency cache and its refresh counter, the guide's lazy projections,
one-off self-checks), autograd Functions that launch triton kernels from Python, complex SENSE
algebra inside the layer, and 30 untied layer instances against a recompile limit of 8. The
glue has none of that: a few tensors in, a few out, fixed structure, so one compiled graph
(forward AND backward, through AOTAutograd) serves every layer. What it buys is fusion: the
eager glue is a chain of one-kernel-per-op passes over 64- to 338-channel maps, plus layout
copies, plus the same again in the backward.

Switch: `IMMAP_GUIDED_COMPILE` = 0 (off, default) | 1 / inductor (the real thing, GPU) |
aot_eager (dynamo + AOTAutograd without code generation: what the CPU tests trace). Off, the
callers run their eager code unchanged.
"""

from __future__ import annotations

import os
import warnings

import torch

_MODES = {"0": None, "": None, "off": None, "1": "inductor", "inductor": "inductor",
          "aot_eager": "aot_eager", "eager": "eager"}


def _env_mode():
    v = os.environ.get("IMMAP_GUIDED_COMPILE", "0").strip().lower()
    if v not in _MODES:
        raise ValueError(f"IMMAP_GUIDED_COMPILE must be 0 / 1 / inductor / aot_eager, got {v!r}")
    return _MODES[v]


MODE = _env_mode()
_CACHE = {}


def set_mode(mode):
    """None / "inductor" / "aot_eager" / "eager" for the process -> the previous value."""
    global MODE
    if mode not in (None, "inductor", "aot_eager", "eager"):
        raise ValueError(f"compile mode must be None, 'inductor', 'aot_eager' or 'eager'; "
                         f"got {mode!r}")
    old, MODE = MODE, mode
    return old


def set_from_config(flag):
    """A config's `training.guided_compile` (true / false / absent) -> the mode now in force.

    The environment variable wins when it is set, so a run can be switched either way without
    editing its saved config (as IMMAP_PLANAR_STATE does for `training.planar_state`). Absent
    (None) leaves the default, off. Changes no parameter and, to fp roundoff, no output.
    """
    if os.environ.get("IMMAP_GUIDED_COMPILE", "").strip():
        return MODE
    if flag is not None:
        if not isinstance(flag, bool):
            raise ValueError(f"training.guided_compile must be true or false, got {flag!r}")
        set_mode("inductor" if flag else None)
    return MODE


def enabled():
    return MODE is not None


_FELL_BACK = set()


def compiled(fn):
    """`fn` compiled under the current mode (memoised per function and mode), or `fn`.

    Shapes: dynamo's AUTOMATIC dynamic shapes (dynamic=None). The first call compiles for its
    exact sizes; the first call with a different size recompiles with that dimension symbolic,
    and from then on one graph serves every size. The image grid is NOT fixed per run: fastMRI
    knee volumes come in several widths, and with dynamic=False each width cost a compile until
    dynamo's recompile limit (8) stopped the run (2026-11-08, LGGS knee, latent width 184 then
    160).

    Failure: compiling is a speed-up, never a dependency. If dynamo refuses a piece -- a graph
    break under fullgraph, the recompile limit, a backend failure -- that piece runs its eager
    code (the same map) for the rest of the process, with one warning, instead of the
    exception ending a training run.
    """
    if MODE is None:
        return fn
    key = (fn, MODE)
    if key not in _CACHE:
        import torch._dynamo.exc as dexc
        # What dynamo raises when it refuses: TorchDynamoException covers graph breaks
        # (Unsupported), backend failures and internal errors; the recompile limit under
        # fullgraph raises FailOnRecompileLimitHit, which is NOT one of those (it derives from
        # Exception directly) -- and is exactly what ended the 2026-11-08 run.
        refused = tuple(c for c in (dexc.TorchDynamoException,
                                    getattr(dexc, "FailOnRecompileLimitHit", None)) if c)
        # fullgraph: a graph break in a pure tensor function is a bug -- surface it (once, as
        # the warning below), rather than silently compiling half the function
        cf = torch.compile(fn, backend=MODE, fullgraph=True, dynamic=None)

        def run(*args):
            if key in _FELL_BACK:
                return fn(*args)
            try:
                return cf(*args)
            except refused as err:
                _FELL_BACK.add(key)
                warnings.warn(f"compile_glue: {fn.__name__} could not be compiled "
                              f"({type(err).__name__}: {str(err).splitlines()[0]}); running it "
                              f"eagerly from now on -- same result, without the fusion")
                return fn(*args)

        run.__name__ = fn.__name__
        _CACHE[key] = run
    return _CACHE[key]


def fell_back():
    """Names of the pieces that fell back to eager in this process (for logs and tests)."""
    return sorted(k[0].__name__ for k in _FELL_BACK)
