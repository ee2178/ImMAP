from operators.base import Operator


class Identity(Operator):

    def forward(self, x):
        return x

    def adjoint(self, x):
        return x
