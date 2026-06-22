# datasets/__init__.py
from datasets.registry import build_loader, register_loader  # re-export

# import each loader module so it self-registers
from datasets.fastmri import loader as _fastmri_loader  # noqa: F401
from datasets.BSD432 import loader as _bsd432_loader     # noqa: F401
from datasets.BraTS import ccl_register, synth_register
