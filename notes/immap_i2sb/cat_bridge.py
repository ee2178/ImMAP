"""Two small, real networks for the talk's figures -> figs/real_*.png.

Task: recover an image x0 from its low spatial frequencies y = L(x0) (L as in make_image_figs.py:
keep |f| < 1/16 cycles per pixel). Pixel values in [0, 1].

  bridge     I2SB: z_t = (1 - t) x0 + t y + sqrt(V t (1 - t)) eps, V = 0.16. Predicts x0 from
             (z_t, y, t): E[x0 | z_t, y].
  diffusion  guided diffusion: x_s = x0 + sigma eps, sigma log-uniform in [0.02, 3]. Predicts x0
             from (x_s, y, sigma): E[x0 | x_s, y].

Training data: random 64 x 64 crops (with flips) of eight scikit-image photos, INCLUDING the cat
shown in the talk, drawn half of the time (an illustration, not a generalisation test: an
earlier run without the cat gave visibly poor samples). Side information is computed on each full
photo, then cropped with the same window. The networks are fully convolutional and run on the
whole 288 x 288 crop at test time.

    python cat_bridge.py train --kind bridge|diffusion [--iters N]   # -> ckpt/cat_<kind>.pt
    python cat_bridge.py figs                                         # -> figs/real_*.png
"""
import argparse
import copy
import math
import os
import time

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import data

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt", "cat_{}.pt")
FIGS = os.path.join(HERE, "figs")
CUTOFF, V, PATCH = 1.0 / 16, 0.16, 64
TRAIN = ("astronaut", "coffee", "rocket", "stereo_motorcycle", "brick", "grass", "gravel")
P_CAT = 0.5                 # share of crops taken from the cat (chelsea)
SMIN, SMAX, SDATA = 0.02, 3.0, 0.25   # diffusion noise range; input scaling


def L(x):
    """Low spatial frequencies of a (..., H, W) array (numpy or torch)."""
    t = torch.as_tensor(x, dtype=torch.float64)
    fy = torch.fft.fftfreq(t.shape[-2], dtype=torch.float64)[:, None]
    fx = torch.fft.fftfreq(t.shape[-1], dtype=torch.float64)[None, :]
    mask = ((fy.abs() < CUTOFF) & (fx.abs() < CUTOFF)).to(torch.float64)
    return torch.fft.ifft2(torch.fft.fft2(t) * mask).real


def chw(img):
    img = img[0] if isinstance(img, tuple) else img
    img = img.astype(np.float64) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    return torch.from_numpy(img[..., :3]).permute(2, 0, 1)


def cat():
    return chw(data.chelsea()[6:294, 80:368])


# ----------------------------------------------------------------------------- the network
def _emb(t, dim=64):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = (1000.0 * t)[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Block(nn.Module):
    def __init__(self, cin, cout, emb=64):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.e(e)[:, :, None, None].chunk(2, dim=1)
        return self.skip(x) + self.c2(F.silu(self.n2(h) * (1 + scale) + shift))


class Net(nn.Module):
    """Fully convolutional UNet, 3 levels (x1, x1/2, x1/4), no attention. Input: z_t and y."""

    def __init__(self, ch=32):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.inp = nn.Conv2d(6, ch, 3, padding=1)
        self.d1, self.d2, self.d3 = Block(ch, ch), Block(ch, 2 * ch), Block(2 * ch, 4 * ch)
        self.m = Block(4 * ch, 4 * ch)
        self.u3, self.u2 = Block(8 * ch, 2 * ch), Block(4 * ch, ch)
        self.u1 = Block(2 * ch, ch)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, z, y, t):
        e = self.emb(_emb(t))
        h1 = self.d1(self.inp(torch.cat([z, y], 1)), e)
        h2 = self.d2(F.avg_pool2d(h1, 2), e)
        h3 = self.d3(F.avg_pool2d(h2, 2), e)
        h = self.m(h3, e)
        h = self.u3(torch.cat([h, h3], 1), e)
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2.0), h2], 1), e)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2.0), h1], 1), e)
        return y + self.out(h)                      # predict x0 as a correction to y


def sig_emb(sigma):
    """Diffusion noise level -> the [0, 1] "time" the network is conditioned on."""
    return (torch.log(torch.as_tensor(sigma)) - math.log(SMIN)) / (math.log(SMAX) - math.log(SMIN))


def diff_in(xs, sigma):
    """Scale a noisy image to unit-ish variance (EDM's c_in)."""
    s = torch.as_tensor(sigma, device=xs.device).reshape(-1, 1, 1, 1)
    return xs / (s ** 2 + SDATA ** 2).sqrt()


# ----------------------------------------------------------------------------- training
def train(iters, kind="bridge", batch=24, lr=1e-3, dev="cpu"):
    torch.manual_seed(0)
    pairs = []
    for name in TRAIN:
        x = chw(getattr(data, name)())
        pairs.append((x.float(), L(x).float()))
    xc = chw(data.chelsea())
    cat_pair = (xc.float(), L(xc).float())
    net = Net().to(dev)
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr / 20)
    g = torch.Generator().manual_seed(0)
    print(f"[{kind}] params {sum(p.numel() for p in net.parameters()) / 1e3:.0f}k, "
          f"training on chelsea ({P_CAT:.0%}) + {', '.join(TRAIN)}", flush=True)
    t0, run = time.time(), None
    for it in range(1, iters + 1):
        xs, ys = [], []
        for _ in range(batch):
            if torch.rand(1, generator=g) < P_CAT:
                x, y = cat_pair
            else:
                x, y = pairs[int(torch.randint(len(pairs), (1,), generator=g))]
            i = int(torch.randint(x.shape[1] - PATCH + 1, (1,), generator=g))
            j = int(torch.randint(x.shape[2] - PATCH + 1, (1,), generator=g))
            cx, cy = x[:, i:i + PATCH, j:j + PATCH], y[:, i:i + PATCH, j:j + PATCH]
            if torch.rand(1, generator=g) < 0.5:
                cx, cy = cx.flip(2), cy.flip(2)
            xs.append(cx)
            ys.append(cy)
        x0, y = torch.stack(xs), torch.stack(ys)
        eps = torch.randn(x0.shape, generator=g)
        if kind == "bridge":
            t = torch.rand(batch, generator=g).clamp(1e-3, 1.0)
            tt = t[:, None, None, None]
            inp = (1 - tt) * x0 + tt * y + (V * tt * (1 - tt)).sqrt() * eps
        else:
            sigma = torch.exp(math.log(SMIN) + torch.rand(batch, generator=g)
                              * (math.log(SMAX) - math.log(SMIN)))
            inp = diff_in(x0 + sigma[:, None, None, None] * eps, sigma)
            t = sig_emb(sigma)
        x0, y, inp, t = x0.to(dev), y.to(dev), inp.to(dev), t.to(dev)
        loss = F.mse_loss(net(inp, y, t), x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pm, 1 - min(0.999, (1 + it) / (10 + it)))
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if it % 250 == 0 or it == iters:
            el = time.time() - t0
            print(f"[{kind}] it {it}/{iters} loss {run:.5f} {it / el:.2f} it/s "
                  f"eta {(iters - it) / (it / el) / 60:.1f} min", flush=True)
    path = CKPT.format(kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in ema.state_dict().items()}, path)
    print("saved", path)


# ----------------------------------------------------------------------------- I2SB
PAD = 24


def guess(net, z, y, t):
    """The network's E[x0 | z_t, y] on a whole image. Trained on interior crops only, so the
    image is reflect-padded for the call and cropped back (no border artifacts)."""
    zp = F.pad(z, (PAD,) * 4, mode="reflect")
    yp = F.pad(y, (PAD,) * 4, mode="reflect")
    tt = torch.full((y.shape[0],), float(t), device=y.device)
    return net(zp, yp, tt)[..., PAD:-PAD, PAD:-PAD]


@torch.no_grad()
def i2sb(net, y, n, gen, keep=()):
    """Equispaced t = 1, 1 - 1/n, ..., 0. Returns the sample and {t: (z_t, guess)} for t in keep."""
    ts = np.linspace(1.0, 0.0, n + 1)
    z, snaps = y.clone(), {}
    for t, s in zip(ts[:-1], ts[1:]):
        xh = guess(net, z, y, t)
        if any(abs(t - k) < 1e-9 for k in keep):
            snaps[round(float(t), 4)] = (z.clone(), xh.clone())
        if s == 0.0:
            return xh, snaps
        noise = torch.randn(y.shape, generator=gen).to(y.device)
        z = (s / t) * z + (1 - s / t) * xh + math.sqrt(V * s * (t - s) / t) * noise
    return z, snaps


def save(name, img):
    a = img.detach().cpu().double().clamp(0, 1).permute(1, 2, 0).numpy()
    iio.imwrite(os.path.join(FIGS, name + ".png"), (a * 255).round().astype(np.uint8))


def save_err(name, e, rng=0.45):
    g = (0.5 + e.mean(0) / (2 * rng)).clamp(0, 1)
    save(name, g[None].repeat(3, 1, 1))


def figs(steps=20, n_samples=8):
    os.makedirs(FIGS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(dev)
    net.load_state_dict(torch.load(CKPT.format("bridge"), map_location=dev))
    net.eval()
    x0 = cat().float()[None].to(dev)
    y = L(x0.cpu()).float().to(dev)
    gen = torch.Generator().manual_seed(0)
    psnr = lambda a, b: float(10 * torch.log10(1 / F.mse_loss(a.clamp(0, 1), b)))  # noqa: E731

    # one run, snapshots at t = 1, 0.75, 0.5, 0.25 (+ the sample at 0)
    keep = (1.0, 0.75, 0.5, 0.25)
    sample, snaps = i2sb(net, y, steps, gen, keep=keep)
    for t in keep:
        z, xh = snaps[t]
        tag = f"{int(round(100 * t)):03d}"
        save(f"real_z{tag}", z[0])
        save(f"real_g{tag}", xh[0])
    save("real_sample0", sample[0])
    reg = snaps[1.0][1]                                   # the first guess: the regression E[x0|y]
    save("real_reg", reg[0])
    save("real_y", y[0])
    save("real_x0", x0[0])

    # more samples: mean and per-pixel spread
    samples = [sample] + [i2sb(net, y, steps, gen)[0] for _ in range(n_samples - 1)]
    for k, s_ in enumerate(samples[1:3], start=1):
        save(f"real_sample{k}", s_[0])
    S = torch.cat(samples)
    mean, std = S.mean(0), S.std(0).mean(0)
    save("real_mean", mean)
    save("real_std", (std / std.quantile(0.995)).clamp(0, 1)[None].repeat(3, 1, 1))

    # one step t = 0.75 -> 0.5 with the REAL guess: structured vs unstructured error
    t, s = 0.75, 0.5
    rn = lambda: torch.randn(x0.shape, generator=gen).to(dev)       # noqa: E731
    z_t = (1 - t) * x0 + t * y + math.sqrt(V * t * (1 - t)) * rn()
    with torch.no_grad():
        g = guess(net, z_t, y, t)
    mean_s = (1 - s) * x0 + s * y
    walked = (s / t) * z_t + (1 - s / t) * g
    renoised = walked + math.sqrt(V * s * (t - s) / t) * rn()
    on_bridge = mean_s + math.sqrt(V * s * (1 - s)) * rn()
    for name, z, ref in (("a", on_bridge, mean_s), ("b", g, x0), ("c", walked, mean_s),
                         ("d", renoised, mean_s)):
        save(f"real_re_{name}_img", z[0])
        save_err(f"real_re_{name}_err", (z - ref)[0])

    # the two expectations, inputs chosen to look typical (NOT matched levels): a noisy image
    # x0 + sigma eps with sigma = 0.6, and a bridge sample at t = 0.75 (mostly the blurry guide,
    # mild noise), each with its own noise draw
    dnet = Net().to(dev)
    dnet.load_state_dict(torch.load(CKPT.format("diffusion"), map_location=dev))
    dnet.eval()
    t, sig = 0.75, 0.6
    z_b = (1 - t) * x0 + t * y + math.sqrt(V * t * (1 - t)) * rn()
    u = x0 + sig * rn()
    with torch.no_grad():
        g_b = guess(net, z_b, y, t)
        g_d = guess(dnet, diff_in(u, sig), y, float(sig_emb(sig)))
    save("real_cmp_bridge_in", z_b[0])
    save("real_cmp_bridge_out", g_b[0])
    save("real_cmp_diff_in", u[0])
    save("real_cmp_diff_out", g_d[0])
    print(f"bridge t={t}: guess {psnr(g_b, x0):.2f} dB; noisy image sigma={sig}: "
          f"guess {psnr(g_d, x0):.2f} dB")

    print(f"PSNR vs the true cat: side info {psnr(y, x0):.2f} dB, regression {psnr(reg, x0):.2f} dB, "
          f"one sample {psnr(sample, x0):.2f} dB, mean of {n_samples} {psnr(mean[None], x0):.2f} dB")
    print(f"guess at t=0.75: rms error {float((g - x0).pow(2).mean().sqrt()):.4f}; "
          f"sigma_sb(0.5) = {math.sqrt(V * 0.25):.3f}")
    print("wrote", FIGS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "figs"))
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--kind", choices=("bridge", "diffusion"), default="bridge")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    train(a.iters, kind=a.kind, dev=a.device) if a.cmd == "train" else figs()
