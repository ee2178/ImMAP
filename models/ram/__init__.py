"""RAM baseline: the Reconstruct Anything Model, vendored, behind ImMAP's interface.

`_upstream/` is github.com/matthieutrs/ram's model file, byte-identical except
for one moved import. `adapter.py` is the only ImMAP-authored file here.
Needs `deepinv` (pinned in torch/exp13_ram.sbatch's install note).
"""
from .adapter import RAM

__all__ = ["RAM"]
