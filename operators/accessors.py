"""
Read / write accessors for composed forward operators.

Port of the "Accessor functions" and "Functional updaters" sections of
Sljiva's `src/operators.jl`.

Writers return a NEW operator rather than mutating one.  That is what lets the
LADMM unroll thread an evolving encoding operator through its layers under
autograd: each layer's refined coil maps become a fresh operator, and the graph
of the previous one stays intact.
"""

from __future__ import annotations

from operators.base import CompositeOperator, Operator
from operators.mask import Mask
from operators.sense import Sense, SoftSense


def _ops(A):
    return A.ops if isinstance(A, CompositeOperator) else [A]


def find_op(A, cls):
    """First sub-operator of type `cls`, or None."""
    for op in _ops(A):
        if isinstance(op, cls):
            return op
    return None


def get_mask(A):
    op = find_op(A, Mask)
    if op is None:
        raise ValueError("no Mask operator found")
    return op.mask


def get_sensitivity_map(A):
    op = find_op(A, (Sense, SoftSense))
    if op is None:
        raise ValueError("no Sense operator found")
    return op.smaps


def _replace(A, cls, factory):
    ops = [factory(op) if isinstance(op, cls) else op for op in _ops(A)]
    return ops[0] if len(ops) == 1 else CompositeOperator(ops)


def set_sensitivity_map(A, smaps):
    """New operator with the coil maps replaced (originals untouched)."""
    return _replace(A, (Sense, SoftSense),
                    lambda op: type(op)(smaps))


def set_mask(A, mask):
    return _replace(A, Mask, lambda op: Mask(mask))
