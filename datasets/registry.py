"""
Central registry for dataset loaders.

Each loader registers itself with @register_loader("name").
`build_loader` then looks up the loader by the "name" field in the
config's data block, so train.py never needs to know which datasets exist.
"""

_LOADER_REGISTRY = {}


def register_loader(name):
    def deco(fn):
        if name in _LOADER_REGISTRY:
            raise ValueError(f"Loader '{name}' is already registered")
        _LOADER_REGISTRY[name] = fn
        return fn
    return deco


def build_loader(data_cfg, **overrides):
    """
    data_cfg : dict with a "name" key plus loader kwargs (from cfg["data"][split]).
    overrides: per-split args set in train.py (e.g. shuffle, drop_last).
    """
    cfg = dict(data_cfg)          # copy so we don't mutate the original config

    try:
        name = cfg.pop("name")
    except KeyError:
        raise ValueError("data config is missing a 'name' field")

    if name not in _LOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Registered loaders: {sorted(_LOADER_REGISTRY)}"
        )

    return _LOADER_REGISTRY[name](**cfg, **overrides)
