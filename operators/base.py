from __future__ import annotations


class Operator:
    """
    Base linear operator class.
    """

    def forward(self, x):
        raise NotImplementedError

    def adjoint(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def gram(self, x):
        return self.H(self(x))

    def normal(self, x):
        # Same map; routed through `gram` so a subclass that fuses the normal
        # equations only has to override one method.
        return self.gram(x)

    def transpose(self):
        """
        Return an operator whose forward and adjoint are swapped.

        Memoised.  This used to define a fresh class object on every call, and
        `CompositeOperator.adjoint` calls it once per sub-operator -- an
        unrolled multigrid network evaluates the Gram ~100 times per forward,
        so that was several hundred dynamic class creations per pass, all of
        them identical.
        """
        t = getattr(self, "_transpose_cache", None)
        if t is None:
            t = TransposeOperator(self)
            self._transpose_cache = t
        return t

    @property
    def H(self):
        """
        Convenience property for transpose operator.
        Usage:
            A_H = A.H
        """
        return self.transpose()

    @property
    def T(self):
        """
        Alias for transpose.
        """
        return self.transpose()

    def __matmul__(self, other):
        return CompositeOperator([self, other])


class TransposeOperator(Operator):
    """`parent` with forward and adjoint swapped."""

    def __init__(self, parent):
        self.parent = parent

    def forward(self, x):
        return self.parent.adjoint(x)

    def adjoint(self, x):
        return self.parent.forward(x)

    def transpose(self):
        # (A^T)^T = A
        return self.parent

    def __repr__(self):
        return f"{self.parent!r}.H"


class CompositeOperator(Operator):

    def __init__(self, ops):

        self.ops = []

        for op in ops:
            if isinstance(op, CompositeOperator):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)

        self._fused_gram = _match_sense_gram(self.ops)

    def forward(self, x):

        for op in reversed(self.ops):
            x = op(x)

        return x

    def adjoint(self, x):

        for op in self.ops:
            x = op.H(x)

        return x

    def gram(self, x):
        """`E^H E x`, fused when the composition is a SENSE encoding.

        `Mask(m) @ FFT2D() @ Sense(s)` -- optionally followed by grid-transfer
        operators, which is what `galerkin` appends for the coarse multigrid
        levels -- has a Gram that needs neither of the centring shifts and only
        one mask multiply.  See `FFT2D.sense_gram` for the derivation.  Any
        other composition falls back to the generic `A^H A`.
        """
        if self._fused_gram is None:
            return self.H(self(x))

        mask_op, fft_op, sense_op, tail = self._fused_gram
        for op in reversed(tail):          # `forward` order: innermost first
            x = op(x)
        x = fft_op.sense_gram(x, sense_op.smaps, mask_op.mask)
        for op in tail:                    # `adjoint` order: outermost first
            x = op.H(x)
        return x

    def __repr__(self):

        names = [op.__class__.__name__ for op in self.ops]
        return " @ ".join(names)


def _match_sense_gram(ops):
    """Recognise `Mask @ FFT2D @ Sense @ <grid transfers>`.

    Matched by an explicit `OP_KIND` tag rather than `isinstance`, so this
    module does not have to import its own subclasses (`operators.mask` and
    friends import *this* one).  `SoftSense` deliberately carries no tag: its
    coil reduction is over a different axis, so the fused kernel would be wrong.
    """
    if len(ops) < 3:
        return None
    kinds = [getattr(op, "OP_KIND", None) for op in ops[:3]]
    if kinds != ["kspace_diag", "centered_fft", "sense"]:
        return None
    return ops[0], ops[1], ops[2], ops[3:]
