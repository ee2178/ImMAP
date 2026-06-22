import torch
import torch.nn as nn
import numpy as np

from models.components import ComplexConvTranspose2d
from operators.projections import uball_project
from solvers.eigen import power_method


class BaseUnrolledModel(nn.Module):
    """
    Shared initialization + projection logic
    for CDL / LPDS / variants.
    """
    @torch.no_grad()
    def project_filters(self):
        for k in range(self.K):
            self.A[k].weight = uball_project(self.A[k].weight)
            self.B[k].weight = uball_project(self.B[k].weight)

    def init_filters(self, dtype=torch.cfloat):
        W = torch.randn(self.M, self.C, self.P, self.P, dtype=dtype)
        for k in range(self.K):
            self.A[k].weight = W
            self.B[k].weight = W.conj()
    
    def spectral_init(self):
        with torch.no_grad():
            DDt = lambda x: self.B[0](self.A[0](x))
            L = power_method(
                DDt,
                torch.rand(1, self.C, 128, 128, dtype=self.A[0].weight.dtype),
                num_iter=200,
                verbose=False,
            )[0]

            scale = np.sqrt(np.abs(L))

            print(f"Power method returns L = {L}")

            for k in range(self.K):
                self.A[k].weight.data /= scale
                self.B[k].weight.data /= scale
    
