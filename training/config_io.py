"""
Config serialization. Deliberately free of torch, numpy and every other ImMAP
module.

This lived in `training/common.py`, which imports `models.build_model` (and so
`torch.nn.attention.flex_attention`) and sits behind a `training/__init__.py`
that imports torchvision. Writing a config therefore needed a full training
environment -- so `scripts/make_mg_recon_configs.py` could not write anything
outside the cluster, despite its own docstring promising otherwise and despite
the code below needing nothing but the standard library and PyYAML.

`training.common` re-exports these names, so existing
`from training.common import write_config` imports are unaffected.
"""

import json
import os
import re

import yaml


# Sentinel for a float we serialize ourselves (see _canon_floats). Plain ASCII on purpose:
# a NUL would be legal in a str but puts raw NUL bytes in this source file. write_config
# asserts it is absent from the payload before substituting, so a collision cannot pass.
_RAW = "@@__raw_float_{}__@@"


def _canon_float(v):
    """`1e-06` -> `1.0e-6`, or None to keep json's own repr.

    train.py reads configs with yaml.safe_load, so YAML 1.1 number rules apply, not JSON's: a
    float needs a decimal point in the mantissa AND an explicitly signed exponent. json.dump
    writes `1e-06`, which matches no float production and comes back as the STRING '1e-06' --
    CosineAnnealingLR then does arithmetic on it. json.load is unaffected, which is what makes
    this easy to miss when eyeballing a config.

    `1.0e-6` is the spelling every hand-written config in this repo uses, so machine-written
    configs match them rather than introducing a second convention. (An earlier version wrote
    plain decimals, `0.000001`, on the theory that some PyYAML builds reject the exponent form.
    That was a misdiagnosis: the config that triggered it held `1e-06` as a STRING already, and
    the regex of the day rewrote the characters inside the quotes without changing the fact that
    it was quoted. Both spellings load as floats; this one matches the repo.)

    A mantissa that has no decimal point gets `.0` appended, because `1e+6` -> `1000000` would
    come back as an INT and silently change the type of a value written as a float.

    Returning None means json's repr is already a plain decimal (`0.0001`), which YAML reads as a
    float with no help. write_config validates the round-trip either way, so a spelling that does
    not survive `yaml.safe_load` is refused rather than written.
    """
    r = repr(float(v))
    if "e" not in r and "E" not in r:
        return None                       # already a plain decimal, or inf/nan
    mant, _, exp = r.lower().partition("e")
    if "." not in mant:
        mant += ".0"
    sign = "-" if exp.startswith("-") else "+"
    digits = exp.lstrip("+-").lstrip("0") or "0"
    return f"{mant}e{sign}{digits}"


def _canon_floats(o, raws):
    """Replace every float needing re-spelling with a unique placeholder, collecting the
    replacement text in `raws`.

    This walks the OBJECT rather than regexing the serialized text, which an earlier version did.
    Text substitution cannot tell a JSON number from the same characters inside a string, so a
    value like "run 1e-06 sweep" got silently rewritten to "run 0.000001 sweep". Walking values
    cannot touch strings at all. It also leaves a config whose number arrived as a STRING (a sweep
    script that json.dump'd instead of calling write_config, then had train.py yaml-load it back)
    exactly as it is, so the validation below still catches it and fails loudly.
    """
    if isinstance(o, float):
        p = _canon_float(o)
        if p is None:
            return o
        raws.append(p)
        return _RAW.format(len(raws) - 1)
    if isinstance(o, dict):
        return {k: _canon_floats(v, raws) for k, v in o.items()}
    if isinstance(o, list):
        return [_canon_floats(v, raws) for v in o]
    return o


def write_config(cfg, path):
    """Write a config as JSON that yaml.safe_load will also read correctly.

    train.py reads configs with yaml.safe_load, so YAML 1.1 number rules apply, not JSON's:
    a float needs BOTH a decimal point AND a signed exponent. json.dump writes 1e-06 for
    eta_min, which yaml hands back as the STRING '1e-06' -- CosineAnnealingLR then dies
    doing arithmetic on it. json.load is unaffected, which is what makes this so easy to
    miss. Every config the repo writes goes through here, and the result is verified with
    the same loader train.py uses before it is accepted.
    """
    raws = []
    marked = _canon_floats(cfg, raws)
    text = json.dumps(marked, indent=4)
    for i, plain in enumerate(raws):
        token = json.dumps(_RAW.format(i))          # the quoted form, exactly as json wrote it
        if text.count(token) != 1:
            raise ValueError(
                f"{path}: float placeholder {token} appears {text.count(token)} times -- a config "
                f"string collided with the sentinel. Rename the offending value.")
        text = text.replace(token, plain)

    loaded = yaml.safe_load(text)
    bad = []

    def scan(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                scan(v, f"{p}.{k}" if p else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                scan(v, f"{p}[{i}]")
        elif isinstance(o, str) and re.fullmatch(r"[-+]?[\d._]+(?:[eE][-+]?\d+)?", o):
            bad.append((p, o))          # any number-shaped string, exponent or not

    scan(loaded)
    if bad:
        raise ValueError(f"{path}: numeric values would load as strings under yaml: {bad}")
    if loaded != json.loads(text):
        raise ValueError(f"{path}: yaml and json disagree on the written config")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")     # trailing newline: POSIX, and it keeps a rewritten config from
                                 # showing a spurious "\ No newline at end of file" in every diff
    return path
