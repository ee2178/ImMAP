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


def enabled():
    return MODE is not None


def compiled(fn):
    """`fn` compiled under the current mode (memoised per function and mode), or `fn`."""
    if MODE is None:
        return fn
    key = (fn, MODE)
    if key not in _CACHE:
        # fullgraph: these are pure tensor functions, so a graph break is a bug, not a
        # fallback to accept silently. dynamic=False: the shapes are the grid's, fixed per run.
        _CACHE[key] = torch.compile(fn, backend=MODE, fullgraph=True, dynamic=False)
    return _CACHE[key]
