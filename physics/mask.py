import torch
import torch.fft as fft
import torch.nn.functional as F
import math
from typing import Union, Tuple

### Acceleration Mask Generation

def resolve_acs_lines(N, acs_lines, center_frac=None):
    """The ACS width a mask will actually have -- the ONE place it is decided.

    `center_frac` wins when given: `Sljiva/src/mask.jl::generate_center_mask`
    takes `Nc = round(N * center_frac)` and forces it ODD so the block is
    symmetric about DC (13 lines at N=320, center_frac=0.04). Anything that needs
    the ACS width -- the mask itself, and the online map estimator calibrating
    from that ACS -- must ask here, or the two disagree: passing the config's
    `acs_lines=20` to the estimator while the mask holds 13 would calibrate from
    columns that are not in the ACS at all.
    """
    if center_frac is None:
        return int(acs_lines)
    nc = int(round(N * float(center_frac)))       # ties-to-even, as Julia's round
    if nc % 2 == 0:
        nc -= 1
    return max(nc, 0)


def make_acc_mask(
    shape,
    accel,
    acs_lines=24,
    dim=1,
    mode="uniform",          # "uniform" or "random"
    variable_density=False,
    offset=0,                # "uniform" only: int phase, or "random"
    seed=None,
    device="cpu",
    center_frac=None,
    adjust_accel=False,
):
    """
    Generate either a Uniform or Random vertical subsampling mask

    `offset` shifts which residue class of lines a uniform mask keeps -- the
    counterpart of `uniform_offset` in Sljiva's `synthmri_closure.yaml`. Pass
    "random" to draw it in [0, accel) per call; the default 0 reproduces the
    original fixed pattern.

    THE ACS IS NOT FREE, AND `adjust_accel` IS WHAT PAYS FOR IT.

    With `adjust_accel=False` (the default, and what every run so far used) the
    outer lines are laid down every `accel` across the WHOLE axis and the ACS
    block is added on top, so the mask samples more than `N / accel` lines and
    the acceleration is nominal, not real. Measured on a 320-line axis with 20
    ACS lines:

        nominal R=4   ->  95 lines  ->  EFFECTIVE R 3.37
        nominal R=8   ->  57 lines  ->  EFFECTIVE R 5.61
        nominal R=16  ->  39 lines  ->  EFFECTIVE R 8.21

    `adjust_accel=True` reproduces `Sljiva/src/mask.jl::generate_uniform_mask`,
    which widens the outer spacing to absorb the ACS,

        adj_accel = round((N - Nc) / (N / accel - Nc))

    so the TOTAL line count is `N / accel` and R means what it says (15.24 at a
    nominal 16, the remainder being integer spacing). `center_frac` is the same
    file's ACS convention -- a fraction of the axis, forced odd -- instead of a
    fixed line count.

    Both default to the old behaviour: flipping them changes the sampling
    pattern, so runs either side are not comparable. `effective_accel()` below
    reports what a mask actually did.
    """
    Ny, Nx = shape
    N = shape[dim]
    mask = torch.zeros((Ny, Nx), dtype=torch.float32, device=device)

    # ACS region
    if center_frac is not None:
        # generate_center_mask, placed at pad = (N - Nc + 1) // 2.
        acs_lines = resolve_acs_lines(N, acs_lines, center_frac)
        acs_start = (N - acs_lines + 1) // 2
        acs_end = acs_start + acs_lines
    else:
        center = N // 2
        half_acs = acs_lines // 2
        acs_start = center - half_acs
        acs_end = center + half_acs

    # `spacing` is what the UNIFORM branch strides by; `accel` stays the
    # requested rate so the random branch can size its budget from it. Without
    # that separation the adjustment is applied twice -- measured: a nominal
    # R=8 random mask came out at an effective 15.2.
    budget = N / float(accel)
    spacing = accel
    if adjust_accel:
        if budget <= acs_lines:
            raise ValueError(
                f"adjust_accel: the ACS alone is {acs_lines} lines but "
                f"R={accel} allows only {budget:.1f} of {N} -- there is no "
                f"budget left for outer lines. Lower acs_lines/center_frac "
                f"(Sljiva uses center_frac=0.04, i.e. 13 lines at N=320) or "
                f"lower R.")
        spacing = max(1, int(round((N - acs_lines) / (budget - acs_lines))))

    acs_idx = torch.arange(acs_start, acs_end, device=device)
    # ---------------------------------------------------------
    # Uniform Cartesian
    # ---------------------------------------------------------
    if mode == "uniform":
        if offset == "random":
            off = int(torch.randint(0, int(spacing), (1,)).item())
        else:
            off = int(offset) % int(spacing)
        outer_idx = torch.arange(off, N, spacing, device=device)
        # Remove overlap with ACS
        outer_idx = outer_idx[
            (outer_idx < acs_start) | (outer_idx >= acs_end)
        ]
        idx_keep = torch.cat([outer_idx, acs_idx]).unique()
    # ---------------------------------------------------------
    # Random Cartesian
    # ---------------------------------------------------------
    elif mode == "random":
        n_outer = N - acs_lines
        # `adjust_accel` here means what it means in generate_random_mask: the
        # Bernoulli rate is (N/accel - Nc) / (N - Nc), i.e. the ACS is counted
        # against the budget rather than added to it.
        n_keep_outer = (math.floor(budget - acs_lines) if adjust_accel
                        else math.floor(n_outer / accel))
        n_keep_outer = max(0, min(n_keep_outer, n_outer))

        # Sampling PDF
        if variable_density:
            x = torch.linspace(-1, 1, N, device=device)
            pdf = torch.exp(-4 * x**2)
        else:
            pdf = torch.ones(N, device=device)

        # Remove ACS from PDF
        pdf[acs_start:acs_end] = 0
        pdf = pdf / pdf.sum()

        if seed is not None:
            torch.manual_seed(seed)
        outer_idx = torch.multinomial(
            pdf,
            n_keep_outer,
            replacement=False,
        )
        idx_keep = torch.cat([outer_idx, acs_idx]).unique()
    else:
        raise ValueError("mode must be 'uniform' or 'random'")
    # Fill mask
    mask.index_fill_(dim, idx_keep, 1.0)
    # Convert to 1 x 1 x H x W
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask

def effective_accel(mask, dim=-1):
    """N / (sampled lines): what a mask actually did, as opposed to its label.

    Report this next to any R. With `adjust_accel=False` the two differ by
    nearly 2x at R=16, which is the difference between a hard problem and a
    moderate one.
    """
    m = mask
    while m.dim() > 1:
        m = m.amax(dim=0) if m.shape[0] > 1 else m[0]
    lines = float((m > 0).sum())
    return float(m.numel()) / max(lines, 1.0)


### SSDU Utils
def mask_uniform_subsample(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]] = 0.5,
) -> torch.Tensor:
    """
    Uniformly subsample a k-space mask.

    Args:
        m_omega: Boolean mask of shape (B, 1, Nx, Ny)
        rho:     Subsampling fraction or (min, max) range for random fraction

    Returns:
        m_lambda: Subsampled boolean mask, same shape as m_omega
    """
    if isinstance(rho, (int, float)):
        rho = (rho, rho)

    B, _, Nx, Ny = m_omega.shape
    m_lambda = torch.zeros_like(m_omega, dtype=torch.bool)

    for b in range(B):
        omega_indices = m_omega[b, 0].nonzero(as_tuple=False)  # (N, 2)
        N = omega_indices.shape[0]

        rho_b = rho[0] + (rho[1] - rho[0]) * torch.rand(1).item()
        N_rho = round(N * rho_b)

        perm = torch.randperm(N)[:N_rho]
        selected = omega_indices[perm]  # (N_rho, 2)

        m_lambda[b, 0, selected[:, 0], selected[:, 1]] = True

    return m_lambda

def mask_uniform_subsample_1D(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]] = 0.5,
    dim: int = 3,
) -> torch.Tensor:
    """
    Uniformly subsample a k-space mask along a single dimension (line subsampling).

    Args:
        m_omega: Boolean mask of shape (B, 1, Nx, Ny)
        rho:     Subsampling fraction or (min, max) range for random fraction
        dim:     Dimension along which to subsample lines (2 for Nx, 3 for Ny).
                 The mask is collapsed across all other spatial dims to find
                 acquired lines, then entire lines are kept or dropped together.

    Returns:
        m_lambda: Subsampled boolean mask, same shape as m_omega
    """
    if isinstance(rho, (int, float)):
        rho = (rho, rho)

    assert dim in (2, 3), f"dim must be 2 (Nx) or 3 (Ny), got {dim}"

    B = m_omega.shape[0]
    collapse_dim = 3 if dim == 2 else 2  # the spatial dim we reduce over

    m_lambda = torch.zeros_like(m_omega, dtype=torch.bool)

    for b in range(B):
        # A line is "acquired" if any point along the orthogonal dimension is acquired
        acquired_lines = m_omega[b, 0].any(dim=collapse_dim - 2)  # (Nx,) or (Ny,)
        omega_line_indices = acquired_lines.nonzero(as_tuple=False).squeeze(1)  # (N,)
        N = omega_line_indices.shape[0]

        rho_b = rho[0] + (rho[1] - rho[0]) * torch.rand(1).item()
        N_rho = round(N * rho_b)

        perm = torch.randperm(N)[:N_rho]
        selected = omega_line_indices[perm]  # (N_rho,)

        if dim == 2:
            m_lambda[b, 0, selected, :] = True
        else:
            m_lambda[b, 0, :, selected] = True

        # Mask back down to only acquired points in Ω
        m_lambda[b] = m_lambda[b] & m_omega[b]

    return m_lambda


def mask_gaussian_subsample(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]] = 0.5,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Gaussian-weighted subsample a k-space mask, biased toward k-space center.

    Args:
        m_omega: Boolean mask of shape (B, 1, Nx, Ny)
        rho:     Subsampling fraction or (min, max) range for random fraction
        sigma:   Std dev of Gaussian as a fraction of half-FOV

    Returns:
        m_lambda: Subsampled boolean mask, same shape as m_omega
    """
    if isinstance(rho, (int, float)):
        rho = (rho, rho)

    B, _, Nx, Ny = m_omega.shape
    Cx, Cy = Nx // 2, Ny // 2
    device = m_omega.device

    m_lambda = torch.zeros_like(m_omega, dtype=torch.bool)

    for b in range(B):
        rho_b = rho[0] + (rho[1] - rho[0]) * torch.rand(1).item()
        N = m_omega[b, 0].sum().item()
        N_rho = round(N * rho_b)

        count = 0
        while count < N_rho:
            n_candidates = max((N_rho - count) * 4, 64)
            xs = (Cx + sigma * Cx * torch.randn(n_candidates, device=device)).round().long()
            ys = (Cy + sigma * Cy * torch.randn(n_candidates, device=device)).round().long()

            for x, y in zip(xs.tolist(), ys.tolist()):
                if count >= N_rho:
                    break
                if (
                    0 <= x < Nx
                    and 0 <= y < Ny
                    and m_omega[b, 0, x, y]
                    and not m_lambda[b, 0, x, y]
                ):
                    m_lambda[b, 0, x, y] = True
                    count += 1

    return m_lambda


def mask_subsample(
    m: torch.Tensor,
    rho: Union[float, Tuple[float, float]],
    type: str = "gaussian",
    **kwargs,
) -> torch.Tensor:
    """
    Subsample a k-space mask using the specified strategy.

    Args:
        m:    Boolean mask of shape (B, 1, Nx, Ny)
        rho:  Subsampling fraction or (min, max) range
        type: "gaussian" or "uniform"

    Returns:
        Subsampled boolean mask
    """
    if type == "uniform":
        return mask_uniform_subsample(m, rho)
    elif type == "gaussian":
        return mask_gaussian_subsample(m, rho, **kwargs)
    elif type == "uniform_1D":
        return mask_uniform_subsample_1D(m, rho, **kwargs)
    else:
        raise ValueError(f'mask_subsample type "{type}" not implemented.')


def ssdu_mask_subsample(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]],
    acs_size: int = 0,
    type: str = "gaussian",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split an acquired k-space mask (Ω) into input (Λ) and loss (Ξ) masks for SSDU.

    Ω = Λ ∪ Ξ  (approximately, before optional ACS inclusion)

    Args:
        m_omega:  Boolean mask of shape (B, 1, Nx, Ny)
        rho:      Fraction of Ω points assigned to Λ (input mask)
        acs_size: Size of auto-calibration signal (ACS) region to include in both masks
        type:     Subsampling strategy — "gaussian" or "uniform"

    Returns:
        (m_lambda, m_xi): Input and loss boolean masks, same shape as m_omega
    """
    m_lambda = mask_subsample(m_omega, rho, type=type, **kwargs)
    m_xi = ~m_lambda & m_omega

    if acs_size > 0:
        _, _, Nx, Ny = m_omega.shape
        Cx, Cy = Nx // 2, Ny // 2
        acs_mask = torch.zeros_like(m_omega, dtype=torch.bool)

        if Cx > 0 and Cy > 0:
            acs_mask[:, :, Cx - acs_size // 2:Cx + acs_size // 2,
                          Cy - acs_size // 2:Cy + acs_size // 2] = True
        elif Cx > 0:
            acs_mask[:, :, Cx - acs_size // 2:Cx + acs_size // 2, :] = True
        elif Cy > 0:
            acs_mask[:, :, :, Cy - acs_size // 2:Cy + acs_size // 2] = True

        m_lambda = m_lambda | acs_mask
        m_xi = m_xi | acs_mask

    return m_lambda, m_xi

# Wrapper function for ssdu_mask as used in ImMAP2.5 
def gen_ssdu_mask(shape, base_acs, ssdu_base_accel, ssdu_acs, ssdu_rho, device = 'cpu'):
    # Generate a base mask 
    ssdu_base_mask = make_acc_mask(
        shape,
        ssdu_base_accel,
        base_acs
    )
    # Cast to bool so we can apply bitwise operations internally
    ssdu_base_mask = ssdu_base_mask.bool()
    # Subsample on top of this mask
    _, ssdu_mask = ssdu_mask_subsample(
        ssdu_base_mask,
        rho = ssdu_rho, # Discard rho % of lines
        acs_size = ssdu_acs,
        type = "uniform_1D"
    )
    # Push to GPU
    ssdu_mask = ssdu_mask.to(device)
    return ssdu_mask

### Mask Caching (Useful in training)
_mask_cache = {}

def get_mask_cached(smaps, R, acs_lines, mode, offset=0, center_frac=None,
                    adjust_accel=False):
    """Memoised `make_acc_mask`, keyed on everything that changes the pattern.

    The cache is what makes a per-step mask cheap, but it also means a cached
    mask is reused for the WHOLE run. That is exactly right for a uniform mask
    at a fixed offset (the pattern is deterministic anyway) and wrong for any
    mode that is supposed to vary -- so `mode="random"` and `offset="random"`
    bypass it and draw a fresh pattern each call.
    """
    Ny, Nx = smaps.shape[-2], smaps.shape[-1]

    def build():
        return make_acc_mask(
            shape=(Ny, Nx),
            accel=R,
            acs_lines=acs_lines,
            mode=mode,
            offset=offset,
            center_frac=center_frac,
            adjust_accel=adjust_accel,
        ).to(smaps.device, non_blocking=True)

    if mode == "random" or offset == "random":
        return build()

    # center_frac and adjust_accel change the PATTERN, so they belong in the
    # key -- a cache hit across a change of either would hand back a mask for a
    # different acceleration.
    key = (Ny, Nx, R, acs_lines, mode, offset, center_frac, adjust_accel,
           smaps.device)

    if key not in _mask_cache:
        _mask_cache[key] = build()

    return _mask_cache[key]
