"""E2E-VarNet baseline: fastMRI's model, vendored, behind ImMAP's interface.

`_fastmri/` is upstream code -- verified byte-identical except for varnet.py's
two import lines. `adapter.py` is the only ImMAP-authored file here.
"""
from .adapter import E2EVarNet

__all__ = ["E2EVarNet"]
