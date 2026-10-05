"""Filter-bank visualization for the unrolled models (CDLNet / GroupCDL / LPDSNet / ...).

Works for BOTH complex (`is_complex=True`) and real (`is_complex=False`) dictionaries. The
components expose a `.weight` property that is a complex tensor when the layer is complex and a
real tensor otherwise, so extraction is dtype-agnostic (the old `get_B_complex` path assumed
`conv_imag` existed and crashed on real models).

Rendering per input channel c of a filter tensor W (N, C, P, P):
  * complex weights -> R = real, G = 0, B = imag   (each mapped to [0,1] around 0.5)
  * real weights    -> white-centered diverging map (positive = red, negative = blue, 0 = white)
Multiple input channels are tiled horizontally (up to `max_channels`) into ONE image, so e.g.
the joint dict's D (C=4 = [x_t, FLAIR, T1, T2]) logs as a single 4-panel figure. Every grid is a
(3, H, W) float tensor in [0,1], the format `get_filter_grids` hands to `wandb.Image`.
"""

import os
import re
import numpy as np
import torch
from visualization.wandb_image import wandb_image
from torchvision.utils import make_grid, save_image


# ============================================================
# (1) EXTRACTION LAYER — PURE TENSORS ONLY (real or complex)
# ============================================================

def _weight(layer):
    """Filter weight as a detached cpu tensor. Complex iff the layer is complex (the `.weight`
    property returns torch.complex(real, imag) for complex layers and the real weight otherwise)."""
    return layer.weight.detach().cpu()


def get_B_complex(B_layer):
    """Back-compat helper. Returns the (possibly real) weight of a Conv/ConvTranspose component;
    real when the layer has no imaginary branch (is_complex=False)."""
    if getattr(B_layer, "conv_imag", None) is None:
        return B_layer.conv_real.weight.detach()
    return torch.complex(B_layer.conv_real.weight.detach(),
                         B_layer.conv_imag.weight.detach())


def extract_lpds_filters(net):
    """Extract raw filters (A, B, D) as tensors (real or complex, per the model).

    Returns dict with A (list), B (list), D (tensor), global_max (float over A,B), K (int).
    """
    assert hasattr(net, "A") and hasattr(net, "B"), "net has no A/B filter banks"

    K = net.K
    A_list = [_weight(net.A[k]) for k in range(K)]
    B_list = [_weight(net.B[k]) for k in range(K)]
    D = _weight(net.D) if hasattr(net, "D") else B_list[0]

    global_max = 0.0
    for W in A_list + B_list:
        global_max = max(global_max, float(W.abs().max()))

    return {"A": A_list, "B": B_list, "D": D, "global_max": global_max, "K": K}


# ============================================================
# (2) RENDERING LAYER — PURE VISUALIZATION (NO I/O)
# ============================================================

def _complex_rgb(Wc, scale_each, global_max):
    """(N, 1, P, P) complex -> (N, 3, P, P) in [0,1]: R = real, G = 0, B = imag (centered 0.5)."""
    real, imag = torch.real(Wc), torch.imag(Wc)
    if scale_each:
        rmax = real.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        imax = imag.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    else:
        rmax = imax = global_max + 1e-8
    real = (real / rmax + 1) / 2
    imag = (imag / imax + 1) / 2
    green = torch.zeros_like(real)
    return torch.cat([real, green, imag], dim=1).clamp(0, 1)


def _real_rgb(Wc, scale_each, global_max):
    """(N, 1, P, P) real -> (N, 3, P, P) in [0,1]: white-centered diverging (+red, -blue, 0 white)."""
    if scale_each:
        m = Wc.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    else:
        m = global_max + 1e-8
    g = (Wc / m).clamp(-1, 1)               # (N,1,P,P) in [-1,1]
    p = g.clamp(min=0)                       # positive part
    n = (-g).clamp(min=0)                    # negative part
    rgb = torch.cat([1 - n, 1 - n - p, 1 - p], dim=1)     # R, G, B
    return rgb.clamp(0, 1)


def filter_to_grid(W, nrow=None, scale_each=False, global_max=None, max_channels=4):
    """Render one filter tensor W (N, C, P, P) — real or complex — to a (3, H, W) RGB grid in
    [0,1]. Up to `max_channels` input channels are tiled horizontally."""
    if W.dim() == 3:                                  # (N, P, P) -> (N, 1, P, P)
        W = W.unsqueeze(1)
    N, C, P, _ = W.shape
    is_cplx = W.is_complex()
    if global_max is None:
        global_max = float(W.abs().max())
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(N)))
    pad_val = 0.5 if is_cplx else 1.0                 # neutral (zero) color for the border

    panels = []
    n_ch = C if max_channels is None else min(C, max_channels)
    for c in range(n_ch):
        Wc = W[:, c:c + 1]
        rgb = _complex_rgb(Wc, scale_each, global_max) if is_cplx \
            else _real_rgb(Wc, scale_each, global_max)
        panels.append(make_grid(rgb, nrow=nrow, padding=2, pad_value=pad_val))   # (3, Hg, Wg)

    if len(panels) == 1:
        return panels[0]
    sep = torch.full((3, panels[0].shape[1], 2), pad_val)     # thin separator between channels
    row = []
    for i, p in enumerate(panels):
        row.append(p)
        if i < len(panels) - 1:
            row.append(sep)
    return torch.cat(row, dim=2)


# back-compat alias: old name assumed complex + single channel; the unified renderer covers it.
def complex_to_rgb_grid(W, nrow, scale_each=False, global_max=None):
    return filter_to_grid(W, nrow=nrow, scale_each=scale_each, global_max=global_max)


def render_lpds_filters(filters, scale_each=False, max_channels=4):
    """Render extracted filters into {name: (3, H, W)} grids (A/B per stage + D)."""
    A_list, B_list, D = filters["A"], filters["B"], filters["D"]
    global_max, K = filters["global_max"], filters["K"]
    nrow = int(np.ceil(np.sqrt(A_list[0].shape[0])))

    out = {}
    for k in range(K):
        out[f"A_stage_{k:02d}"] = filter_to_grid(A_list[k], nrow, scale_each, global_max, max_channels)
        out[f"B_stage_{k:02d}"] = filter_to_grid(B_list[k], nrow, scale_each, global_max, max_channels)
    out["D"] = filter_to_grid(D, nrow, scale_each, global_max, max_channels)
    return out


# ============================================================
# (2b) GENERIC PATH — every net built from components.Conv2d / ConvTranspose2d
# ============================================================
# The multilevel / multigrid / wavelet nets have no top-level A/B: their banks
# sit at paths like `net.layers.7.levels.1.analysis`.  The FIRST integer in a
# path is the unrolled iteration k, so replacing it by `#` groups one bank
# across the K iterations (`net.layers.#.levels.1.analysis`, K of them); any
# later integers (a V-cycle's inner stacks, a level index) stay part of the
# bank's identity.
#
# WHICH ITERATION IS SHOWN.  In an LPDS stack the LAST layer's analysis and the
# FIRST layer's synthesis never receive a gradient (the output never reads that
# dual / the cold start makes no primal step), so showing either would make a
# bank look frozen forever.  Images show the middle iteration, and the drift
# numbers take the median over iterations.

def _gauss_convs(net):
    from models.components import _GaussConvNd
    return [(n, m) for n, m in net.named_modules() if isinstance(m, _GaussConvNd)]


def _template(name):
    return re.sub(r"(^|\.)(\d+)(\.|$)", r"\1#\3", name, count=1)


def _short(template):
    """`net.layers.#.levels.1.analysis` -> `levels.1.analysis`; `D` -> `D`."""
    return template.split("#.", 1)[1] if "#." in template else template


def filter_banks(net):
    """{template: [weight_k, ...]}, one list entry per unrolled iteration, in order."""
    banks = {}
    for name, mod in _gauss_convs(net):
        banks.setdefault(_template(name), []).append(_weight(mod))
    return banks


def filter_snapshot(net):
    """Every conv bank's weight, detached on the cpu -- the reference `filter_stats` measures
    drift against.  Take it right after the model is built (or loaded), before training."""
    return {name: _weight(mod).clone() for name, mod in _gauss_convs(net)}


def filter_stats(net, init=None):
    """Per bank: is it learning?  {filters/spread/<bank>, filters/drift/<bank>}.

    spread  median over k of ||W_k - mean_k W|| / ||mean_k W||.  Every stack here starts from
            ONE prototype copied into all K iterations, so spread > 0 means the untied
            iterations have moved apart -- needs no snapshot, so it survives a resume.
    drift   median over k of ||W_k - W_k(init)|| / ||W_k(init)||, against `filter_snapshot`.
            Only when `init` is given; after a resume it is relative to the resume point.
    """
    out = {}
    stacks = {}
    for name, mod in _gauss_convs(net):
        stacks.setdefault(_template(name), []).append((name, _weight(mod)))
    for tmpl, entries in stacks.items():
        key = _short(tmpl)
        W = torch.stack([w for _, w in entries])
        if len(entries) > 1:
            mean = W.mean(0)
            spread = [float((w - mean).norm() / mean.norm().clamp_min(1e-12)) for w in W]
            out[f"filters/spread/{key}"] = float(np.median(spread))
        if init is not None:
            d = [float((w - init[n]).norm() / init[n].norm().clamp_min(1e-12))
                 for n, w in entries if n in init]
            if d:
                out[f"filters/drift/{key}"] = float(np.median(d))
    return out


def _mid(stack):
    return stack[len(stack) // 2]


def _atom_layer(net):
    """The middle-iteration layer whose image-domain atoms can be computed, and how.

    ("wavelet", layer)   WaveletLPDSLayer: its exact K^H (carries and Q included)
    ("levels", layer)    anything with `levels[l].analysis/.synthesis` (ML-LPDS, ML-CDL sweeps,
                         Cascade LPDS): the conv cascade composed level by level
    None                 single-level or multigrid nets -- their raw filters ARE the atoms, or
                         the levels are joined by grid transfers rather than convs
    """
    from models.wavelet_lpds import WaveletLPDSLayer
    wav = [m for m in net.modules() if isinstance(m, WaveletLPDSLayer)]
    if wav:
        return "wavelet", _mid(wav)
    lev = [m for m in net.modules()
           if isinstance(getattr(m, "levels", None), torch.nn.ModuleList) and len(m.levels)
           and all(hasattr(l_, "analysis") and hasattr(l_, "synthesis") for l_ in m.levels)]
    if lev:
        return "levels", _mid(lev)
    return None


def _uses_synthesis_atoms(layer):
    """CDL sweeps are SYNTHESIS models (x = D g): their atoms are the D cascade.  The LPDS
    family is ANALYSIS-form (penalty on A x): its atoms are the analysis filters, A^H e_m."""
    from models.ml_cdlnet import MLSweep, MLSplitSweep
    return isinstance(layer, (MLSweep, MLSplitSweep))


@torch.no_grad()
def _cascade_adjoint(z, convs, conj):
    """Push level-l codes to the image through `convs` (level l first) as transposed convs.
    `conj` uses the conjugated weights -- the adjoint of an ANALYSIS conv."""
    import torch.nn.functional as F
    from models.wavelet_lpds import _cblock, _from_ri, _to_ri
    z = _to_ri(z)
    for c in convs:
        wr, wi = c.conv_real.weight.detach(), c.conv_imag.weight.detach()
        w = _cblock(wr, -wi if conj else wi, transpose=True)
        z = F.conv_transpose2d(z, w, stride=c.stride, padding=c.padding,
                               output_padding=c.stride - 1, groups=c.groups)
    return _from_ri(z)


def _support(convs):
    """Width of a level's image-domain atom: P + (P - 1)(s_1 + s_1 s_2 + ...)."""
    width, scale = 1, 1
    for c in convs:
        P = c.conv_real.weight.shape[-1]
        width += (P - 1) * scale
        scale *= c.stride
    return width, scale


@torch.no_grad()
def effective_atoms(net, max_atoms=512):
    """{level: (N, C, S, S) complex} image-domain atoms of the middle iteration, or {}.

    Atom m of level l is the image-domain filter the net actually applies for deep channel m:
    A_(1,l)^H e_m for the analysis-form nets (what the l1 penalty sees), D_1 ... D_l e_m for
    the CDL sweeps.  Computed by pushing a unit impulse at channel m back to the image.  A
    level-3 conv acting on 64-channel feature maps says nothing by itself; its atom does.
    """
    found = _atom_layer(net)
    if found is None:
        return {}
    kind, layer = found
    out = {}
    if kind == "wavelet":
        from models.wavelet_lpds import _adjoint
        convs = list(layer.analysis)
        _, scale = _support(convs)
        n = 2 * _support(convs)[0] // scale + 2
        ws = [(c.conv_real.weight.detach(), -c.conv_imag.weight.detach()) for c in convs]
        dev = ws[0][0].device
        levels = sorted({lvl for lvl, _ in layer.tags})
        for lvl in levels:
            # a band born at level l is carried below it by fixed unshuffles,
            # so its atom only spans the first l convs
            width = _support(convs[:lvl])[0]
            chans = [m for m in range(layer.M) if layer.tags[m % layer.Cg][0] == lvl][:max_atoms]
            z = torch.zeros(len(chans), layer.M, n, n, dtype=torch.complex64, device=dev)
            z[torch.arange(len(chans)), chans, n // 2, n // 2] = 1
            a = _adjoint(z, ws, layer.QHr, layer.carry)
            out[lvl] = _crop_to_mass(a, width).cpu()
        return out

    synth = _uses_synthesis_atoms(layer)
    for lvl in range(1, len(layer.levels) + 1):
        path = [layer.levels[j] for j in range(lvl - 1, -1, -1)]
        convs = [lv.synthesis if synth else lv.analysis for lv in path]
        width, scale = _support(convs[::-1])
        M = convs[0].conv_real.weight.shape[0]           # level-l channels
        m = min(M, max_atoms)
        n = 2 * width // scale + 2
        dev = convs[0].conv_real.weight.device
        z = torch.zeros(m, M, n, n, dtype=torch.complex64, device=dev)
        z[torch.arange(m), torch.arange(m), n // 2, n // 2] = 1
        a = _cascade_adjoint(z, convs, conj=not synth)
        out[lvl] = _crop(a, width).cpu()
    return out


def _crop_to_mass(a, width):
    """Crop each of (N, C, S, S) atoms to `width` + 4 px around ITS OWN centre of mass.

    A carried band's atom does not sit at the grid centre, and not at one shared
    offset either: the unshuffle phase that carries it shifts it by up to half a
    coarse pixel per level (~6 px for a level-1 band under two carries), and the
    phase differs from atom to atom.  So each atom is centred on itself -- the
    grid then shows every atom whole, aligned on its own centre.
    """
    S = a.shape[-1]
    w = min(S, width + 4)
    e = a.abs().pow(2).sum(1)                                   # (N, S, S)
    idx = torch.arange(S, device=a.device, dtype=e.dtype)
    tot = e.sum((1, 2)).clamp_min(1e-30)
    cy = ((e.sum(2) * idx).sum(1) / tot).round().long()
    cx = ((e.sum(1) * idx).sum(1) / tot).round().long()
    out = []
    for n in range(a.shape[0]):
        lo_y = max(0, min(S - w, int(cy[n]) - w // 2))
        lo_x = max(0, min(S - w, int(cx[n]) - w // 2))
        out.append(a[n, :, lo_y:lo_y + w, lo_x:lo_x + w])
    return torch.stack(out)


def _crop(a, width):
    """Centre crop to the atom's support (+2 px), keeping every channel."""
    S = a.shape[-1]
    w = min(S, width + 4)
    lo = (S - w) // 2
    return a[..., lo:lo + w, lo:lo + w]


def collect_filter_images(net, scale_each=False, max_channels=4, max_filters=256,
                          max_atoms=512):
    """{name: (3, H, W) grid in [0,1]} for any net -- no I/O, so it is testable.

    * the old CDLNet-style A/B/D nets: exactly what `render_lpds_filters` always produced;
    * everything else: the middle iteration's raw bank per `filters/<bank>` (the first
      `max_filters` filters, up to `max_channels` input channels), plus the image-domain atoms
      per level as `atoms/level<l>` where the net has them (`effective_atoms`).
    """
    if hasattr(net, "A") and hasattr(net, "B"):
        filters = extract_lpds_filters(net)
        return render_lpds_filters(filters, scale_each=scale_each, max_channels=max_channels)

    out = {}
    # Group the MODULES, then move one bank per group to the cpu. `filter_banks`
    # transfers every iteration's weight (216 host copies for the mglpds
    # V-cycle) to draw only the middle one of each stack.
    stacks = {}
    for name, mod in _gauss_convs(net):
        stacks.setdefault(_template(name), []).append(mod)
    for tmpl, mods in stacks.items():
        W = _weight(_mid(mods))[:max_filters]
        out[_short(tmpl)] = filter_to_grid(W, scale_each=scale_each, max_channels=max_channels)
    for lvl, a in effective_atoms(net, max_atoms=max_atoms).items():
        out[f"atoms/level{lvl}"] = filter_to_grid(a, scale_each=scale_each, max_channels=1)
    return out


# ============================================================
# (3) IO / LOGGING LAYER — W&B OR DISK
# ============================================================

def get_filter_grids(net, scale_each=False, max_channels=4, init=None, stats=False):
    """W&B-compatible logging dict: filter IMAGES under `filters/...` for ANY net built from
    `components` convs (or the old A/B/D layout).  Returns {} (never raises) for a net with no
    such banks (e.g. E2E-VarNet), so unwrapped callers stay safe.

    Images only by default.  `stats=True` adds the `filter_stats` scalars (spread / drift),
    which walk EVERY iteration's weight; `init` is a `filter_snapshot` taken before training,
    for the drift numbers, and is read only then."""
    grids = collect_filter_images(net, scale_each=scale_each, max_channels=max_channels)
    out = {f"filters/{k}": wandb_image(img.permute(1, 2, 0).numpy())
           for k, img in grids.items()}
    if stats and not (hasattr(net, "A") and hasattr(net, "B")):
        out.update(filter_stats(net, init=init))
    return out


def save_filters(net, save_dir, scale_each=False, max_channels=4):
    """Save filter visualizations to disk as PNGs (any net; see `collect_filter_images`)."""
    os.makedirs(save_dir, exist_ok=True)
    grids = collect_filter_images(net, scale_each=scale_each, max_channels=max_channels)
    for name, img in grids.items():
        path = os.path.join(save_dir, f"{name.replace('/', '_')}.png")
        save_image(img, path)
