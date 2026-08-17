"""
Noise models.

`awgn` noises an image directly (the denoising task).  `mri_awgn` builds a
*synthetic* MRI measurement from a coil-combined ground-truth image and a set of
coil sensitivities -- the PyTorch counterpart of
`genobs(clo::SyntheticMRIReco, ...)` in Sljiva's `src/closures/mrireco.jl`.

Both draw sigma through `sample_sigma`, so a noise level means the same thing in
the denoising and reconstruction paths and a config can be moved between them.
"""

import math

import torch

from operators.fourier import fftc


def _draw(shape, dtype, device, generator, fn=torch.randn):
    """`fn(shape, ...)` that tolerates a generator on the *other* device.

    `torch.Generator()` is CPU-only, so passing one alongside a CUDA target raises
    "Expected a 'cuda' device type for generator but found 'cpu'".  Draw on the
    generator's own device and move instead: never raises, and a CPU-seeded run
    produces bit-identical noise on CPU and GPU.
    """
    if generator is None:
        return fn(shape, dtype=dtype, device=device)
    gdev = generator.device
    tgt = torch.device(device) if device is not None else gdev
    if tgt.type != gdev.type:
        return fn(shape, dtype=dtype, device=gdev, generator=generator).to(tgt)
    return fn(shape, dtype=dtype, device=tgt, generator=generator)


def _randn_like(x, generator=None):
    """`torch.randn_like` that accepts a generator (which `randn_like` does not).

    For a complex dtype torch draws the real and imaginary parts at variance 1/2
    each, so `E|z|^2 = 1` and `sigma * randn` has total variance `sigma^2` --
    the same convention `awgn` has always used.  `tests/test_synthetic_kspace.py`
    pins it rather than trusting it.
    """
    if generator is None:
        return torch.randn_like(x)
    return _draw(x.shape, x.dtype, x.device, generator)


def sample_sigma(n, noise_std, dist="uniform", ndim=4, device=None, k=1,
                 eps=1e-8, generator=None):
    """One noise level per batch element, singleton elsewhere.

    Returns a scalar tensor-free float when `noise_std` is a bare number, and a
    `(n, 1, ..., 1)` tensor with `ndim` axes otherwise, so the result broadcasts
    against `(B, C, H, W)` and against `(B, H, W)` alike.

    `dist`:
      * `uniform` -- U[a, b]
      * `log`     -- log-uniform on [a, b], warped by `k` (k > 1 pushes mass
                     toward a, k < 1 toward b; endpoints unchanged)
      * `cosine`  -- sigma = ((cos X + 1)/2)^2 with X uniform on the arccos
                     preimage of [a, b]
    """
    if not isinstance(noise_std, (list, tuple)):
        return noise_std

    shape = (int(n),) + (1,) * (int(ndim) - 1)
    a, b = float(noise_std[0]), float(noise_std[1])

    def _rand():
        return _draw(shape, None, device, generator, fn=torch.rand)

    if dist == "uniform":
        return a + (b - a) * _rand()

    if dist == "log":
        # log-uniform on [a, b] with a shape/"temperature" knob k
        #   k = 1 -> plain log-uniform
        #   k > 1 -> mass pushed toward low sigma (a)
        #   k < 1 -> mass pushed toward high sigma (b)
        lo = a + eps
        return lo * (b / lo) ** (_rand() ** k)

    if dist == "cosine":
        # The lower bound comes from b, not a: arccos is monotonically decreasing.
        hi, lo_x = math.acos(2 * a ** 0.5 - 1), math.acos(2 * b ** 0.5 - 1)
        x = lo_x + (hi - lo_x) * _rand()
        return ((torch.cos(x) + 1) / 2) ** 2

    raise ValueError(
        f"noise_dist must be 'uniform', 'log' or 'cosine'; got {dist!r}")


def awgn(input, noise_std, dist="uniform", k=1, eps=1e-8, generator=None):
    """Additive white Gaussian noise.

    input: clean image, any shape with the batch on dim 0.
    noise_std: a number, or a (lo, hi) range sampled per batch element.
    Returns (noisy, sigma).
    """
    sigma = sample_sigma(input.shape[0], noise_std, dist=dist, ndim=input.dim(),
                         device=input.device, k=k, eps=eps, generator=generator)
    return input + _randn_like(input, generator) * sigma, sigma


def rss(smaps, dim=1, eps=1e-8):
    """Root-sum-of-squares over the coil axis, keeping the axis."""
    return smaps.abs().pow(2).sum(dim=dim, keepdim=True).sqrt() + eps


def mri_awgn(image, acceleration_map, smaps, noise_std, noise_dist="uniform",
             normalize_smaps=False, generator=None, k=1, eps=1e-8):
    """Simulate an accelerated, noisy multicoil measurement from a clean image.

        x_mc  <- s . x                           Sense forward, (B, C, H, W)
        y     <- mask . F (x_mc + sigma . n)

    Noise is added in the COIL-IMAGE domain before the transform, matching
    `acgn(slice_rng, xmc, L)` in the Julia closure.  `fftc` is `norm="ortho"`,
    so that is distributionally the same as adding it in k-space.

    ASSUMES UNIT-RSS MAPS, which is what this repo's preprocessing produces.
    That is what makes `sigma` the noise std of the coil-combined adjoint:

        E^H y = sum_c conj(s_c)(s_c x + n_c) = x (sum_c |s_c|^2) + sum_c conj(s_c) n_c

    is `x + noise` with variance `sigma^2 sum_c |s_c|^2 = sigma^2` exactly when
    `sum_c |s_c|^2 = 1`.  That is the quantity the noise-adaptive thresholds are
    calibrated against.  `mrireco.jl:220` renormalizes because its maps come
    straight off a downsample; ours do not need it, and `normalize_smaps=True`
    is left as an escape hatch for maps of unknown provenance.

    Parameters
    ----------
    image : (B, 1, H, W) complex   coil-combined ground truth
    acceleration_map : (B, 1, H, W) or (1, 1, H, W)   sampling mask
    smaps : (B, C, H, W) complex   coil sensitivities, assumed unit-RSS
    noise_std : number or (lo, hi)  sampled per batch element

    Returns
    -------
    (y, sigma, smaps) -- masked k-space (B, C, H, W), sigma (B, 1, 1, 1), and
    the maps the simulation used.  The maps come back so the caller builds its
    encoding operator from exactly what `y` is consistent with; that matters
    when `normalize_smaps` is on and is a pass-through otherwise.
    """
    if normalize_smaps:
        smaps = smaps / rss(smaps, eps=eps)

    x_coils = smaps * image                      # Sense(smaps) applied to x

    sigma = sample_sigma(x_coils.shape[0], noise_std, dist=noise_dist,
                         ndim=x_coils.dim(), device=x_coils.device, k=k,
                         eps=eps, generator=generator)

    y_coils = fftc(x_coils + _randn_like(x_coils, generator) * sigma)
    return acceleration_map * y_coils, sigma, smaps
