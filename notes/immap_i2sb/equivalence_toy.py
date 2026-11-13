"""Small-scale check that I2SB is a conditional VE diffusion (i2sb_as_warm_start_diffusion.tex).

Task: 4x super-resolution on CIFAR-10 (32x32 colour, [-1, 1]).
- x0 is the image.
- x1 = bilinear-upsample(avgpool4(x0)) is the bridge start and the conditioning.
- The measurement operator for the consistency metric is avgpool4.

Five networks share one architecture, optimiser, budget and data. Each is conditioned on x1
(concatenated) unless noted:

  bridge_a       I2SB: x0-prediction on bridge states x_t, t = k/1000 uniform, tau_a
  bridge_a_s1    the same with another seed            (seed-to-seed baseline)
  bridge_b       I2SB with tau_b = 3 tau_a             (does tau change the target?)
  edm            conditional VE diffusion, EDM preconditioning, log-normal sigma
  bridge_a_nox1  I2SB with tau_a but NOT given x1      (control: the caveat in the note)

Every network is turned into a denoiser D(u, sigma; x1) of the debiased coordinate
u = x0 + sigma * eps:
- a bridge net is fed x_t = x1 + c(u - x1), with c = V/(sigma^2 + V) and t = sigma^2/(sigma^2 + V);
- the EDM net uses its own c_skip / c_out / c_in.

All sampling uses ONE VE ancestral step, u_s = r^2 u + (1-r^2) D + sigma_s sqrt(1-r^2) z,
on either of two ladders:
- the I2SB ladder: uniform in t, starting at sigma = inf;
- the Karras ladder: starting from noise at sigma_max = 80.

The script measures two things:
1. Denoiser agreement. On held-out images with shared noise, ||D_i - D_j||^2 is compared
   across levels, normalised by the MMSE estimate ||D - x0||^2.
2. Posterior agreement. For each (network, ladder) pair it draws K samples per image and
   compares their mean and spread with native I2SB, against a split-half Monte-Carlo
   baseline.

Usage (any CUDA python with torch, numpy and matplotlib):
    python notes/immap_i2sb/equivalence_toy.py --stage all
"""
import argparse
import copy
import importlib
import io
import json
import math
import os
import pickle
import tarfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CIFAR_TAR = os.path.join(os.path.expanduser("~"), "Downloads", "cifar-10-python.tar.gz")
MODELS = ("bridge_a", "bridge_a_s1", "bridge_b", "edm", "bridge_a_nox1")
SIGMA_MAX, SIGMA_MIN, SIGMA_DATA = 80.0, 0.002, 0.5
UNET_CH = 32  # base width; set by --ch


# ----------------------------------------------------------------------------- data
class _SafeUnpickler(pickle.Unpickler):
    """CIFAR's python pickles need only numpy array reconstruction; refuse anything else."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
               ("numpy", "dtype"), ("numpy.core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        if module.startswith("numpy.core") and hasattr(np, "_core"):
            module = module.replace("numpy.core", "numpy._core", 1)
        return getattr(importlib.import_module(module), name)


def load_cifar(path=CIFAR_TAR):
    out = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in ("data_batch_1", "test_batch"):
            f = tar.extractfile(f"cifar-10-batches-py/{member}")
            d = _SafeUnpickler(io.BytesIO(f.read()), encoding="bytes").load()
            x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
            out[member] = torch.from_numpy(x).float().div(127.5).sub(1.0)
    return out["data_batch_1"], out["test_batch"]


def A(x):
    """Measurement: 4x average pooling."""
    return F.avg_pool2d(x, 4)


def degrade(x):
    """Bridge start / condition: the low-resolution image, bilinearly upsampled."""
    return F.interpolate(A(x), scale_factor=4, mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------- network
def _emb(s, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device) / half)
    a = (1000.0 * s)[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.e(e)[:, :, None, None].chunk(2, dim=1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, c, h * w).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        return x + self.o((v @ a.transpose(1, 2)).reshape(b, c, h, w))


class UNet(nn.Module):
    """32 -> 16 -> 8, two res blocks per level, attention at 8x8."""

    def __init__(self, cin, cout=3, ch=64, mult=(1, 2, 2), emb=256):
        super().__init__()
        self.emb_dim = emb
        self.emb = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(cin, ch, 3, padding=1)
        chans, c = [ch], ch
        self.down = nn.ModuleList()
        for i, m in enumerate(mult):
            for _ in range(2):
                self.down.append(ResBlock(c, ch * m, emb))
                c = ch * m
                chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                chans.append(c)
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, emb), Attn(c), ResBlock(c, c, emb)
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(3):
                self.up.append(ResBlock(c + chans.pop(), ch * m, emb))
                c = ch * m
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, cout, 3, padding=1))

    def forward(self, x, s):
        e = self.emb(_emb(s, self.emb_dim))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = blk(h, e) if isinstance(blk, ResBlock) else blk(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for blk in self.up:
            if isinstance(blk, ResBlock):
                h = blk(torch.cat([h, hs.pop()], dim=1), e)
            else:
                h = blk(h)
        return self.out(h)


# ----------------------------------------------------------------------------- the five models
class Bridge(nn.Module):
    """I2SB with the constant Brownian bridge, x0-prediction, t on the 1000-step grid."""

    def __init__(self, tau, cond_x1=True):
        super().__init__()
        self.V, self.cond_x1 = 4.0 * tau ** 2, cond_x1
        self.net = UNet(6 if cond_x1 else 3, ch=UNET_CH)

    def _net(self, xt, t, x1):
        return self.net(torch.cat([xt, x1], 1) if self.cond_x1 else xt, t)

    def loss(self, x0, x1):
        t = torch.randint(1, 1001, (x0.shape[0],), device=x0.device).float() / 1000.0
        tt = t[:, None, None, None]
        xt = (1 - tt) * x0 + tt * x1 + (self.V * tt * (1 - tt)).sqrt() * torch.randn_like(x0)
        return F.mse_loss(self._net(xt, t, x1), x0)

    def denoise(self, u, sigma, x1):
        """E[x0 | u, x1] at noise level `sigma` (a float; inf allowed)."""
        b = u.shape[0]
        if torch.is_tensor(sigma):  # one finite level per sample, shape (B,)
            s2 = sigma.reshape(-1) ** 2
            c = (self.V / (s2 + self.V))[:, None, None, None]
            return self._net(x1 + c * (u - x1), s2 / (s2 + self.V), x1)
        if math.isinf(sigma):
            return self._net(x1, torch.ones(b, device=u.device), x1)
        c, t = self.V / (sigma ** 2 + self.V), sigma ** 2 / (sigma ** 2 + self.V)
        return self._net(x1 + c * (u - x1), torch.full((b,), t, device=u.device), x1)


class EDM(nn.Module):
    """Conditional VE diffusion with EDM preconditioning and its log-normal sigma training."""

    def __init__(self, p_mean=-1.2, p_std=1.2):
        super().__init__()
        self.p_mean, self.p_std = p_mean, p_std
        self.net = UNet(6, ch=UNET_CH)

    @staticmethod
    def _s(sigma):
        return (torch.log(sigma) - math.log(SIGMA_MIN)) / (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))

    def _D(self, u, sigma, x1):
        sd2, s2 = SIGMA_DATA ** 2, sigma ** 2
        c_skip, c_out = sd2 / (s2 + sd2), sigma * SIGMA_DATA / (s2 + sd2).sqrt()
        c_in = 1.0 / (s2 + sd2).sqrt()
        return c_skip * u + c_out * self.net(torch.cat([c_in * u, x1], 1), self._s(sigma.flatten()))

    def loss(self, x0, x1):
        sigma = (self.p_mean + self.p_std * torch.randn(x0.shape[0], device=x0.device)).exp()
        s = sigma[:, None, None, None]
        w = (s ** 2 + SIGMA_DATA ** 2) / (s * SIGMA_DATA) ** 2
        return (w * (self._D(x0 + s * torch.randn_like(x0), s, x1) - x0) ** 2).mean()

    def denoise(self, u, sigma, x1):
        sigma = min(sigma, SIGMA_MAX)  # never trained above sigma_max; inf -> sigma_max
        s = torch.full((u.shape[0], 1, 1, 1), sigma, device=u.device)
        return self._D(u, s, x1)


def build(name, tau_a):
    torch.manual_seed(1 if name.endswith("_s1") else 0)
    if name == "edm":
        return EDM()
    tau = 3.0 * tau_a if name == "bridge_b" else tau_a
    return Bridge(tau, cond_x1=not name.endswith("nox1"))


# ----------------------------------------------------------------------------- training
def train(name, model, xtr, args, dev):
    path = os.path.join(args.out, "ckpt", f"{name}.pt")
    if os.path.exists(path) and not args.retrain:
        model.load_state_dict(torch.load(path, map_location=dev))
        print(f"[{name}] loaded {path}", flush=True)
        return model
    model.to(dev).train()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(1 if name.endswith("_s1") else 0)
    torch.manual_seed(1 if name.endswith("_s1") else 0)
    t0, run = time.time(), 0.0
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, xtr.shape[0], (args.batch,), generator=g)
        x0 = xtr[idx.to(xtr.device)].to(dev)
        flip = torch.rand(args.batch, device=dev) < 0.5
        x0 = torch.where(flip[:, None, None, None], x0.flip(3), x0)
        lr = args.lr * min(1.0, it / 500)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = model.loss(x0, degrade(x0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        run = loss.item() if it == 1 else 0.98 * run + 0.02 * loss.item()
        if it % 1000 == 0 or it == args.iters:
            el = time.time() - t0
            print(f"[{name}] it {it}/{args.iters} loss {run:.4f} {it / el:.1f} it/s "
                  f"eta {(args.iters - it) / (it / el) / 60:.1f} min", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ema.state_dict(), path)
    return ema


# ----------------------------------------------------------------------------- sampling
def ladder(kind, n, V):
    if kind == "i2sb":  # uniform in bridge time t, from t = 1 (sigma = inf) to t = 0
        t = np.linspace(1.0, 0.0, n + 1)
        with np.errstate(divide="ignore"):
            return [math.inf if ti == 1.0 else math.sqrt(V * ti / (1 - ti)) for ti in t]
    i = np.arange(n)  # Karras rho = 7, then a final step to 0
    rho = 7.0
    s = (SIGMA_MAX ** (1 / rho) + i / (n - 1) * (SIGMA_MIN ** (1 / rho) - SIGMA_MAX ** (1 / rho))) ** rho
    return list(s) + [0.0]


@torch.no_grad()
def sample(model, x1, sigmas, gen):
    """VE ancestral sampler in u. sigma = inf at the top means 'start from the regression'."""
    top = sigmas[0]
    if isinstance(model, EDM) and math.isinf(top):
        top = SIGMA_MAX  # the EDM net cannot be evaluated at inf: start from noise at sigma_max
        sigmas = [top] + list(sigmas[1:])
    u = torch.zeros_like(x1) if math.isinf(top) else top * torch.randn(x1.shape, generator=gen, device=x1.device)
    for st, ss in zip(sigmas[:-1], sigmas[1:]):
        d = model.denoise(u, st, x1)
        if ss == 0.0:
            return d
        r2 = 0.0 if math.isinf(st) else (ss / st) ** 2
        z = torch.randn(x1.shape, generator=gen, device=x1.device)
        u = r2 * u + (1 - r2) * d + ss * math.sqrt(1 - r2) * z
    return u


def start_ladder(V, nfe, sigma0, steps):
    """Open-loop ladder: regress at inf, re-noise to sigma0, then geometric targets down to the
    I2SB ladder's last nonzero rung at the same NFE, then 0. `steps` network calls after the
    regression."""
    s_end = math.sqrt(V / (nfe - 1))  # I2SB rung t = 1/nfe
    if steps == 1:
        return [math.inf, sigma0, 0.0]
    g = [sigma0 * (s_end / sigma0) ** (j / (steps - 1)) for j in range(steps)]
    return [math.inf] + g + [0.0]


@torch.no_grad()
def sample_selfpaced(model, x1, nfe, sigma0, corrected, gen, diag=None):
    """Closed-loop ascent on the bridge network.

    1. Regress at sigma = inf, then re-noise to sigma0.
    2. Each later step re-estimates the level actually present from the residual:
       - sigma_hat^2 = ||D - u||^2 / n                 (ImMAP2's estimate, biased low), or
       - sigma_hat^2 = ||D - u||^2 / (n - div D)       (corrected; Hutchinson, +1 call).
    3. Step from sigma_hat to the next target level with h = 1 - r^2, r = target / sigma_hat.
    4. Tell the network the new level.

    Targets are those of start_ladder(); the NFE budget includes the divergence calls.
    """
    steps = (nfe - 1) // 2 if corrected else nfe - 1
    if steps < 1:
        return None
    targets = start_ladder(model.V, nfe, sigma0, steps)[2:-1]  # levels after sigma0, before 0
    b, n = x1.shape[0], x1[0].numel()
    v = model.denoise(x1, math.inf, x1)
    u = v + sigma0 * torch.randn(x1.shape, generator=gen, device=x1.device)
    s = torch.full((b,), sigma0, device=x1.device)
    for j in range(steps):
        v = model.denoise(u, s, x1)
        if j == steps - 1:
            return v
        res2 = (v - u).pow(2).flatten(1).sum(1)
        if corrected:
            z = torch.randn(x1.shape, generator=gen, device=x1.device)
            dlt = 1e-2 * s[:, None, None, None]
            div = (z * (model.denoise(u + dlt * z, s, x1) - v)).flatten(1).sum(1) / dlt.flatten()
            sig_hat = (res2 / (n - div).clamp_min(0.05 * n)).sqrt()
        else:
            sig_hat = (res2 / n).sqrt()
        if diag is not None and j == 0:
            diag.append((sig_hat / s).mean().item())
        tgt = targets[j]
        r = (tgt / sig_hat).clamp(max=1.0)[:, None, None, None]
        z = torch.randn(x1.shape, generator=gen, device=x1.device)
        u = r ** 2 * u + (1 - r ** 2) * v + tgt * (1 - r ** 2).sqrt() * z
        s = torch.where(r.flatten() < 1.0, torch.full_like(s, tgt), sig_hat)
    return v


def psnr(a, b):
    """Mean per-image PSNR for images in [-1, 1] (peak-to-peak 2)."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return (10 * torch.log10(4.0 / mse)).mean().item()


# ----------------------------------------------------------------------------- experiments
@torch.no_grad()
def denoiser_agreement(models, xte, args, dev):
    sigmas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 80.0]
    gen = torch.Generator(device=dev).manual_seed(123)
    names = list(models)
    mse = {n: [] for n in names}
    dis = {}
    for s in sigmas:
        sums = {n: 0.0 for n in names}
        pair = {}
        for i in range(0, xte.shape[0], 250):
            x0 = xte[i:i + 250].to(dev)
            x1 = degrade(x0)
            u = x0 + s * torch.randn(x0.shape, generator=gen, device=dev)
            outs = {n: models[n].denoise(u, s, x1) for n in names}
            for n in names:
                sums[n] += ((outs[n] - x0) ** 2).sum().item()
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    k = (names[a], names[b])
                    pair[k] = pair.get(k, 0.0) + ((outs[names[a]] - outs[names[b]]) ** 2).sum().item()
        n_el = xte.numel()
        for n in names:
            mse[n].append(sums[n] / n_el)
        for k, v in pair.items():
            dis.setdefault(f"{k[0]}|{k[1]}", []).append(v / n_el)
    return {"sigmas": sigmas, "mse": mse, "disagreement": dis}


@torch.no_grad()
def posterior_agreement(models, xte, V_a, args, dev):
    configs = [  # (label, model, ladder)
        ("I2SB (bridge_a, I2SB ladder)", "bridge_a", "i2sb"),
        ("bridge_a, Karras from noise", "bridge_a", "karras"),
        ("EDM, Karras from noise", "edm", "karras"),
        ("EDM, I2SB ladder", "edm", "i2sb"),
        ("bridge_b (3x tau), I2SB ladder", "bridge_b", "i2sb"),
        ("bridge_a seed 1, I2SB ladder", "bridge_a_s1", "i2sb"),
        ("bridge_a no x1, I2SB ladder", "bridge_a_nox1", "i2sb"),
        ("bridge_a no x1, Karras from noise", "bridge_a_nox1", "karras"),
    ]
    x0 = xte[:args.n_post].to(dev)
    x1 = degrade(x0)
    K = args.k_post
    res, samples = {}, {}
    for label, name, lad in configs:
        if name not in models:
            continue
        V = models[name].V if hasattr(models[name], "V") else V_a
        sig = ladder(lad, args.nfe, V if lad == "i2sb" else None)
        gen = torch.Generator(device=dev).manual_seed(7)
        xs = torch.stack([torch.cat([sample(models[name], x1[i:i + 256], sig, gen)
                                     for i in range(0, x1.shape[0], 256)]) for _ in range(K)])
        samples[label] = xs
        mean = xs.mean(0)
        res[label] = {
            "nfe": len(sig) - 1,
            "psnr_sample": float(np.mean([psnr(x, x0) for x in xs])),
            "psnr_mean": psnr(mean, x0),
            "consistency_rmse": ((A(xs.flatten(0, 1)).unflatten(0, (K, -1)) - A(x0)[None]) ** 2)
                                .mean().sqrt().item(),
            "posterior_std": xs.std(0).pow(2).mean().sqrt().item(),
        }
    ref = configs[0][0]
    xr = samples[ref]
    # split-half baseline: two K/2 means of the SAME config differ by sqrt(4/K) * std;
    # two K-sample means from configs with the same law differ by sqrt(2/K) * std.
    half = (xr[:K // 2].mean(0) - xr[K // 2:].mean(0)).pow(2).mean().sqrt().item()
    for label, xs in samples.items():
        d = (xs.mean(0) - xr.mean(0)).pow(2).mean().sqrt().item()
        res[label]["mean_rms_vs_i2sb"] = d
        res[label]["mean_rms_vs_i2sb_over_mc"] = d / (half / math.sqrt(2))
        res[label]["std_ratio_vs_i2sb"] = res[label]["posterior_std"] / res[ref]["posterior_std"]
    return res, samples, x0, x1


def figures(agree, post, samples, x0, x1, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = agree["sigmas"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n, v in agree["mse"].items():
        ax[0].loglog(s, v, marker="o", ms=3, label=n)
    ax[0].set(xlabel=r"$\sigma$", ylabel=r"MSE$(D,\ x_0)$", title="denoiser error (held-out)")
    ax[0].legend(fontsize=8)
    ref = np.array(agree["mse"]["bridge_a"])
    show = [("bridge_a|bridge_a_s1", "seed (baseline)"), ("bridge_a|bridge_b", r"$\tau$ x3"),
            ("bridge_a|edm", "EDM"), ("bridge_a|bridge_a_nox1", "no $x_1$ (control)")]
    for k, lab in show:
        if k in agree["disagreement"]:
            ax[1].loglog(s, np.array(agree["disagreement"][k]) / ref, marker="o", ms=3, label=lab)
    ax[1].set(xlabel=r"$\sigma$", ylabel=r"$\|D_{\rm bridge\_a}-D\|^2 / \mathrm{MSE}(D_{\rm bridge\_a},x_0)$",
              title="disagreement with bridge_a")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "denoiser_agreement.png"), dpi=140)

    labels = list(samples)
    n_img = 6
    rows = 2 + 2 * len(labels)
    fig, axes = plt.subplots(rows, n_img, figsize=(1.3 * n_img, 1.3 * rows))
    to_img = lambda t: ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    names = ["$x_0$", "$x_1$"]
    for j in range(n_img):
        axes[0, j].imshow(to_img(x0[j]))
        axes[1, j].imshow(to_img(x1[j]))
    for r, lab in enumerate(labels):
        xs = samples[lab]
        std = xs.std(0).mean(1).cpu().numpy()
        for j in range(n_img):
            axes[2 + 2 * r, j].imshow(to_img(xs[0, j]))
            axes[3 + 2 * r, j].imshow(std[j], cmap="magma", vmin=0, vmax=0.3)
        names += [lab.replace(", ", "\n"), "pixel std"]
    for i in range(rows):
        axes[i, 0].set_ylabel(names[i], fontsize=6, rotation=0, ha="right", va="center")
        for j in range(n_img):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out, "posterior_samples.png"), dpi=140)


def sweep_configs(V):
    """(label, model, kind, sigma0). Kinds:
    - i2sb / karras: the two fixed ladders;
    - open: regress, re-noise to sigma0, then a fixed geometric ladder (the control);
    - sp / spc: the same start and targets, self-paced with ImMAP2's / the corrected estimate.
    """
    s99 = math.sqrt(V * 0.99 / 0.01)  # sigma(t = 0.99)
    cfg = [("I2SB: bridge, I2SB ladder", "bridge_a", "i2sb", None),
           ("bridge, Karras from noise", "bridge_a", "karras", None),
           ("EDM: EDM, Karras from noise", "edm", "karras", None),
           ("EDM, I2SB ladder", "edm", "i2sb", None)]
    for s0, tag in ((s99, "t=0.99"), (0.5, "sigma0=0.5")):
        cfg += [(f"bridge, open loop, {tag}", "bridge_a", "open", s0),
                (f"bridge, self-paced, {tag}", "bridge_a", "sp", s0),
                (f"bridge, self-paced corrected, {tag}", "bridge_a", "spc", s0)]
    return cfg


def _detail(x):
    """Energy of the band the measurement cannot see: x minus its own low-pass bridge start."""
    return (x - degrade(x)).pow(2).flatten(1).mean(1)


@torch.no_grad()
def nfe_sweep(models, xte, V_a, args, dev):
    """Every (network, ladder) pairing at each step count, against a converged reference.

    The reference is native I2SB at --ref-nfe steps. Reported per run:
    - single-sample PSNR, and PSNR of the K-sample mean;
    - consistency RMSE of the 4x-pooled sample;
    - posterior spread;
    - detail ratio: detail-band energy of the samples / that of x0
      (about 1 = right amount of detail; below 1 = blurry; above 1 = noisy);
    - distance of the posterior mean to the reference's, in units of Monte-Carlo noise.
    """
    x0 = xte[:args.n_post].to(dev)
    x1 = degrade(x0)
    K = args.k_post

    def draw(name, kind, nfe, sigma0=None, diag=None):
        gen = torch.Generator(device=dev).manual_seed(7)
        m = models[name]
        if kind in ("i2sb", "karras"):
            sig = ladder(kind, nfe, V_a if kind == "i2sb" else None)
            fn = lambda xb: sample(m, xb, sig, gen)
        elif kind == "open":
            sig = start_ladder(m.V, nfe, sigma0, nfe - 1)
            fn = lambda xb: sample(m, xb, sig, gen)
        else:
            if (nfe - 1) // (2 if kind == "spc" else 1) < 1:
                return None
            fn = lambda xb: sample_selfpaced(m, xb, nfe, sigma0, kind == "spc", gen, diag)
        return torch.stack([torch.cat([fn(x1[i:i + 256]) for i in range(0, x1.shape[0], 256)])
                            for _ in range(K)])

    ref = draw("bridge_a", "i2sb", args.ref_nfe)
    ref_mean = ref.mean(0)
    mc = (ref[:K // 2].mean(0) - ref[K // 2:].mean(0)).pow(2).mean().sqrt().item() / math.sqrt(2)
    d0 = _detail(x0).mean().item()
    regress = models["bridge_a"].denoise(x1, math.inf, x1)  # 1 NFE: E[x0 | x1]
    res = {"nfe": list(args.nfe_list), "reference_nfe": args.ref_nfe, "mc_noise": mc,
           "regression_psnr": psnr(regress, x0),
           "regression_detail": _detail(regress).mean().item() / d0, "runs": {}}
    for label, name, kind, sigma0 in sweep_configs(V_a):
        rows = []
        for nfe in args.nfe_list:
            t0 = time.time()
            diag = []
            xs = draw(name, kind, nfe, sigma0, diag)
            if xs is None:
                rows.append(None)
                continue
            rows.append({
                "psnr_sample": float(np.mean([psnr(x, x0) for x in xs])),
                "psnr_mean": psnr(xs.mean(0), x0),
                "consistency_rmse": ((A(xs.flatten(0, 1)).unflatten(0, (K, -1)) - A(x0)[None]) ** 2)
                                    .mean().sqrt().item(),
                "posterior_std": xs.std(0).pow(2).mean().sqrt().item(),
                "detail_ratio": _detail(xs.flatten(0, 1)).mean().item() / d0,
                "mean_vs_ref_over_mc": (xs.mean(0) - ref_mean).pow(2).mean().sqrt().item() / mc,
                "first_sigma_hat_over_sigma0": float(np.mean(diag)) if diag else None,
            })
            print(f"  {label:<40} nfe {nfe:>3}  psnr_s {rows[-1]['psnr_sample']:.2f}  "
                  f"detail {rows[-1]['detail_ratio']:.3f}  ({time.time() - t0:.1f}s)", flush=True)
        res["runs"][label] = rows
    return res


def sweep_figure(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nfe = res["nfe"]
    panels = [("psnr_sample", "single-sample PSNR (dB)"), ("consistency_rmse", "consistency RMSE"),
              ("detail_ratio", "detail energy / that of $x_0$"), ("posterior_std", "posterior std"),
              ("mean_vs_ref_over_mc", "posterior-mean distance to ref (MC units)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    def style(label):
        if label.startswith(("I2SB", "bridge, Karras")):
            return "C0", "-" if label.startswith("I2SB") else "--"
        if "EDM" in label:
            return "C2", "-" if label.startswith("EDM:") else "--"
        c = {"open loop": "C7", "self-paced,": "C1", "corrected": "C3"}
        col = next(v for k, v in c.items() if k in label)
        return col, "-" if "t=0.99" in label else ":"
    for ax, (key, title) in zip(axes, panels):
        for label, rows in res["runs"].items():
            c, ls = style(label)
            pts = [(n, r[key]) for n, r in zip(nfe, rows) if r is not None]
            ax.plot(*zip(*pts), color=c, ls=ls, marker="o", ms=3, label=label)
        if key == "detail_ratio":
            ax.axhline(1.0, color="gray", lw=0.8)
            ax.axhline(res["regression_detail"], color="gray", lw=0.8, ls=":")
        if key == "mean_vs_ref_over_mc":
            ax.axhline(1.0, color="gray", lw=0.8)
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks(nfe)
        ax.set_xticklabels([str(n) for n in nfe])
        ax.set(xlabel="NFE", title=title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "nfe_sweep.png"), dpi=140)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("train", "eval", "all", "sweep"), default="all")
    p.add_argument("--nfe-list", type=int, nargs="+", default=[2, 3, 5, 10, 20, 50])
    p.add_argument("--ref-nfe", type=int, default=100)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--n-post", type=int, default=512)
    p.add_argument("--k-post", type=int, default=8)
    p.add_argument("--n-agree", type=int, default=1000)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "equivalence_toy"))
    args = p.parse_args()
    global UNET_CH
    UNET_CH = args.ch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    xtr, xte = load_cifar()
    spread = (xtr[:2000] - degrade(xtr[:2000])).std().item()
    tau_a = spread / 2.0  # sqrt(V) = 2 tau = the actual per-pixel spread of x0 - x1
    print(f"train {tuple(xtr.shape)}  test {tuple(xte.shape)}  std(x0 - x1) = {spread:.3f}  "
          f"tau_a = {tau_a:.3f}  tau_b = {3 * tau_a:.3f}  device {dev}", flush=True)
    xtr = xtr.to(dev)

    models = {}
    for name in args.models:
        m = build(name, tau_a)
        print(f"[{name}] {sum(q.numel() for q in m.parameters()) / 1e6:.2f}M params", flush=True)
        if args.stage in ("train", "all"):
            m = train(name, m, xtr, args, dev)
        else:
            m.load_state_dict(torch.load(os.path.join(args.out, "ckpt", f"{name}.pt"), map_location=dev))
        models[name] = m.to(dev).eval()
    if args.stage == "train":
        return
    if args.stage == "sweep":
        res = nfe_sweep(models, xte, 4 * tau_a ** 2, args, dev)
        with open(os.path.join(args.out, "nfe_sweep.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nregression E[x0|x1] (1 NFE): psnr {res['regression_psnr']:.2f}  "
              f"detail {res['regression_detail']:.3f};  MC noise {res['mc_noise']:.4f}")
        cols = ("psnr_sample", "psnr_mean", "consistency_rmse", "posterior_std", "detail_ratio",
                "mean_vs_ref_over_mc")
        for label, rows in res["runs"].items():
            print(f"\n{label}\n  nfe  " + "  ".join(f"{c[:14]:>14}" for c in cols))
            for n, r in zip(res["nfe"], rows):
                if r is None:
                    continue
                extra = (f"   sigma_hat/sigma0 {r['first_sigma_hat_over_sigma0']:.3f}"
                         if r["first_sigma_hat_over_sigma0"] is not None else "")
                print(f"  {n:>3}  " + "  ".join(f"{r[c]:14.4f}" for c in cols) + extra)
        sweep_figure(res, args.out)
        return

    agree = denoiser_agreement(models, xte[:args.n_agree], args, dev)
    post, samples, x0, x1 = posterior_agreement(models, xte, 4 * tau_a ** 2, args, dev)
    summary = {"std_x0_minus_x1": spread, "tau_a": tau_a, "tau_b": 3 * tau_a,
               "args": vars(args), "denoiser": agree, "posterior": post}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)

    ref = np.array(agree["mse"]["bridge_a"])
    print("\nsigma            " + " ".join(f"{s:>7g}" for s in agree["sigmas"]))
    print("MSE bridge_a     " + " ".join(f"{v:7.4f}" for v in ref))
    for k, v in agree["disagreement"].items():
        if k.startswith("bridge_a|"):
            print(f"{k.split('|')[1]:<17}" + " ".join(f"{d / m:7.3f}" for d, m in zip(v, ref)))
    print(f"\n{'config':<36} psnr_s  psnr_mean  consist  post_std  mean-vs-I2SB/MC  std-ratio")
    for lab, r in post.items():
        print(f"{lab:<36} {r['psnr_sample']:6.2f}  {r['psnr_mean']:8.2f}  {r['consistency_rmse']:7.4f}"
              f"  {r['posterior_std']:7.4f}  {r['mean_rms_vs_i2sb_over_mc']:14.2f}  {r['std_ratio_vs_i2sb']:8.3f}")
    figures(agree, post, samples, x0, x1, args.out)


if __name__ == "__main__":
    main()
