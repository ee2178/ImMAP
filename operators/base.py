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

    def normal(self, x):
        return self.H(self(x))

    def gram(self, x):
        return self.H(self(x))

    def __matmul__(self, other):
        return CompositeOperator([self, other])

    def transpose(self):
        """
        Return an operator whose forward and adjoint are swapped.
        """
        parent = self

        class TransposeOperator(Operator):
            def forward(self, x):
                return parent.adjoint(x)

            def adjoint(self, x):
                return parent.forward(x)

            def transpose(self):
                # (A^T)^T = A
                return parent

        return TransposeOperator()

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

class CompositeOperator(Operator):

    def __init__(self, ops):

        self.ops = []

        for op in ops:
            if isinstance(op, CompositeOperator):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)

    def forward(self, x):

        for op in reversed(self.ops):
            x = op(x)

        return x

    def adjoint(self, x):

        for op in self.ops:
            x = op.H(x)

        return x

    def __repr__(self):

        names = [op.__class__.__name__ for op in self.ops]
        return " @ ".join(names)
