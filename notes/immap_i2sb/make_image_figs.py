"""Natural-image figures for the I2SB talk -> figs/*.png.

Source: scikit-image's "chelsea" photo (CC0), a 288 x 288 crop, values in [0, 1]. No network is
involved: every image below is an exact construction, not a model output.

  side information  y = L x         L: keep only the low spatial frequencies (|f| < 1/16 cycles
                                    per pixel, about an 8x lower resolution); a projection.
  consistent images x_k = y + (n_k - L n_k),  n_k = x shifted by a few pixels.
                    L x_k = L x exactly: each has the same low frequencies, different fine
                    detail (the operator's null space).
  their average     xbar = mean_k x_k  -- what a regression returns: blurry. Used below as the
                    stand-in for an early guess E[x0 | y].
  noise strip       x + sigma eps
  bridge strip      z_t = (1 - t) x + t y + sqrt(V t (1 - t)) eps
  re-noise panels   one I2SB step t -> s toward the guess xbar, errors against the TRUE bridge
                    mean (1 - s) x + s y:
                      a  a true bridge sample at s:   white noise, std sigma_sb(s)
                      b  walking ALL the way:         the guess itself, error xbar - x: purely
                                                      structured, no noise
                      c  walking part way:            (s/t)-shrunk noise + (1 - s/t)(xbar - x)
                      d  part way, then re-noised:    c + fresh noise, total noise std sigma_sb(s)
  warm start        xbar + sigma0 eps for a small and a large sigma0, errors against x

Errors are shown in grey (mid-grey = 0, black/white = -/+ ERR_RANGE), averaged over colour.

    python make_image_figs.py         # rewrites figs/ next to this file
"""
import os

import imageio.v3 as iio
import numpy as np
from skimage import data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
CUTOFF = 1.0 / 16           # side information keeps |f| < CUTOFF (cycles / pixel)
V = 0.16                    # bridge variance scale: sigma_sb(1/2) = sqrt(V)/2 = 0.2
ERR_RANGE = 0.45
SHIFTS = [(dy, dx) for dy in (-4, 0, 4) for dx in (-4, 0, 4) if (dy, dx) != (0, 0)]
SEED = 0


def L(x):
    """Orthogonal projection onto the low spatial frequencies, per colour channel."""
    fy = np.fft.fftfreq(x.shape[0])[:, None]
    fx = np.fft.fftfreq(x.shape[1])[None, :]
    mask = ((np.abs(fy) < CUTOFF) & (np.abs(fx) < CUTOFF)).astype(float)[..., None]
    return np.real(np.fft.ifft2(np.fft.fft2(x, axes=(0, 1)) * mask, axes=(0, 1)))


def save(name, img):
    iio.imwrite(os.path.join(OUT, name + ".png"),
                (np.clip(img, 0, 1) * 255).round().astype(np.uint8))


def save_err(name, e):
    g = 0.5 + e.mean(axis=-1) / (2 * ERR_RANGE)
    save(name, np.repeat(g[..., None], 3, axis=-1))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = data.chelsea()[6:294, 80:368].astype(np.float64) / 255.0
    y = L(x)

    # images consistent with y, and their average
    def consistent(dy, dx):
        n = np.roll(x, (dy, dx), axis=(0, 1))
        return y + (n - L(n))
    alts = [consistent(*sh) for sh in SHIFTS]
    for k, a in enumerate(alts):
        assert np.allclose(L(a), L(x))
    xbar = np.mean(alts + [x], axis=0)
    save("x0", x)
    save("y", y)
    save("alt1", alts[0])
    save("alt2", alts[5])
    save("avg", xbar)

    # adding noise to an image
    eps = rng.standard_normal(x.shape)
    for k, sig in enumerate((0.1, 0.3, 1.0, 3.0)):
        save(f"noise{k}", x + sig * eps)

    # bridge samples from x (t = 0) to y (t = 1)
    eps_b = rng.standard_normal(x.shape)
    for t in (0.25, 0.5, 0.75):
        z = (1 - t) * x + t * y + np.sqrt(V * t * (1 - t)) * eps_b
        save(f"bridge{int(round(100 * t))}", z)

    # one I2SB step t -> s toward the stand-in guess xbar
    t, s = 0.75, 0.5
    sig = lambda u: np.sqrt(V * u * (1 - u))                       # noqa: E731
    mean_s = (1 - s) * x + s * y
    e1, e2 = rng.standard_normal(x.shape), rng.standard_normal(x.shape)
    z_t = (1 - t) * x + t * y + sig(t) * e1                        # a true bridge sample at t
    walked = (s / t) * z_t + (1 - s / t) * xbar
    fresh = np.sqrt(V * s * (t - s) / t)
    renoised = walked + fresh * e2
    on_bridge = mean_s + sig(s) * rng.standard_normal(x.shape)
    assert np.isclose((s / t) ** 2 * sig(t) ** 2 + fresh ** 2, sig(s) ** 2)
    for name, z in (("re_a", on_bridge), ("re_b", xbar), ("re_c", walked), ("re_d", renoised)):
        save(name + "_img", z)
        save_err(name + "_err", z - mean_s if name != "re_b" else z - x)

    # warm start: the blurry guess plus too little / enough noise
    e3 = rng.standard_normal(x.shape)
    for name, s0 in (("warm_lo", 0.04), ("warm_hi", 0.25)):
        save(name + "_img", xbar + s0 * e3)
        save_err(name + "_err", xbar + s0 * e3 - x)

    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))                 # noqa: E731
    print("rms(xbar - x) =", round(rms(xbar - x), 4),
          " structured part of the walk =", round(rms((1 - s / t) * (xbar - x)), 4),
          " sigma_sb(s) =", round(float(sig(s)), 4),
          " walked noise =", round(float(s / t * sig(t)), 4))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
