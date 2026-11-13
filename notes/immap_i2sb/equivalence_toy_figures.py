"""Visual examples for immap_guided_ascent.tex, Section 8.3, from the trained CIFAR bridge network.

Usage: python notes/immap_i2sb/equivalence_toy_figures.py   (after equivalence_toy.py trained bridge_a)

1. trajectory.png: one I2SB run in both coordinates, with the same noise.
   - row 1: the bridge state x the network sees;
   - row 2: the ascent state u = (x - t x1)/(1 - t);
   - row 3: the shared estimate.
2. samples.png: target, start, the one-shot regression E[x0 | x1], then samples and per-pixel
   spread from I2SB (start x1, sigma = inf) and from the same ascent started at pure noise
   (sigma = 80, Karras ladder).
"""
import json
import math
import os
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import equivalence_toy as E  # noqa: E402

OUT = os.path.join(HERE, "equivalence_toy")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E.UNET_CH = 32
_, xte = E.load_cifar()
tau = json.load(open(os.path.join(OUT, "results.json")))["std_x0_minus_x1"] / 2
m = E.build("bridge_a", tau)
m.load_state_dict(torch.load(os.path.join(OUT, "ckpt", "bridge_a.pt"), map_location=dev,
                             weights_only=True))
m = m.to(dev).eval()
V = m.V
to_img = lambda t: ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


# ----------------------------------------------------------------------------- trajectory
@torch.no_grad()
def trajectory(idx=3, nfe=10, seed=1):
    x0 = xte[idx:idx + 1].to(dev)
    x1 = E.degrade(x0)
    ts = [1.0 - k / nfe for k in range(nfe + 1)]
    sig = lambda t: math.inf if t == 1.0 else math.sqrt(V * t / (1 - t))
    g = torch.Generator(device=dev).manual_seed(seed)
    u = torch.zeros_like(x1)
    rec = []
    for k in range(nfe):
        t, s = ts[k], ts[k + 1]
        st, ss = sig(t), sig(s)
        x = x1 if math.isinf(st) else (1 - t) * u + t * x1
        v = m.denoise(u, st, x1)
        rec.append((t, st, x, None if math.isinf(st) else u, v))
        r2 = 0.0 if math.isinf(st) else (ss / st) ** 2
        u = r2 * u + (1 - r2) * v + ss * math.sqrt(1 - r2) * torch.randn(x1.shape, generator=g,
                                                                          device=dev)
    show = [0, 1, 2, 4, 6, 9]
    fig, ax = plt.subplots(3, len(show) + 1, figsize=(1.45 * (len(show) + 1), 4.6))
    for j, k in enumerate(show):
        t, st, x, uu, v = rec[k]
        ax[0, j].imshow(to_img(x[0]))
        if uu is None:
            ax[1, j].set_facecolor("0.85")
            ax[1, j].text(0.5, 0.5, "any\n($\\sigma=\\infty$)", ha="center", va="center",
                          transform=ax[1, j].transAxes, fontsize=12)
        else:
            ax[1, j].imshow(to_img(uu[0]))
        ax[2, j].imshow(to_img(v[0]))
        lab = "$\\infty$" if math.isinf(st) else f"{st:.2f}"
        ax[0, j].set_title(f"$t={t:.1f}$\n$\\sigma=${lab}", fontsize=12)
    ax[0, -1].imshow(to_img(x0[0]))
    ax[0, -1].set_title("target $x_0$", fontsize=12)
    ax[1, -1].imshow(to_img(x1[0]))
    ax[1, -1].set_title("start $x_1$", fontsize=12)
    ax[2, -1].axis("off")
    for i, lab in enumerate(["I$^2$SB state $x_t$\n(network input)",
                             "ascent state $u$\n$=x_0+\\sigma\\varepsilon$",
                             "estimate\n$\\hat x_0 = v$"]):
        ax[i, 0].set_ylabel(lab, fontsize=12)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout(pad=0.3, h_pad=1.2)
    fig.savefig(os.path.join(OUT, "trajectory.png"), dpi=170)


# ----------------------------------------------------------------------------- samples
@torch.no_grad()
def samples(idx=(0, 1, 7, 12, 21, 29), nfe=50, K=8, seed=7):
    x0 = xte[list(idx)].to(dev)
    x1 = E.degrade(x0)
    reg = m.denoise(x1, math.inf, x1)
    runs = {}
    for name, lad in (("I$^2$SB", "i2sb"), ("from noise", "karras")):
        sg = E.ladder(lad, nfe, V if lad == "i2sb" else None)
        g = torch.Generator(device=dev).manual_seed(seed)
        runs[name] = torch.stack([E.sample(m, x1, sg, g) for _ in range(K)])
    cols = (["target $x_0$", "start $x_1$", "$E[x_0\\mid x_1]$"]
            + [f"I$^2$SB #{i + 1}" for i in range(2)] + ["I$^2$SB std"]
            + [f"from noise #{i + 1}" for i in range(2)] + ["from noise std"])
    n = len(idx)
    fig, ax = plt.subplots(n, len(cols), figsize=(1.25 * len(cols), 1.3 * n + 0.3))
    for i in range(n):
        ims = [to_img(x0[i]), to_img(x1[i]), to_img(reg[i])]
        for name in runs:
            xs = runs[name][:, i]
            ims += [to_img(xs[0]), to_img(xs[1]), xs.std(0).mean(0).cpu().numpy()]
        for j, im in enumerate(ims):
            if im.ndim == 2:
                ax[i, j].imshow(im, cmap="magma", vmin=0, vmax=0.3)
            else:
                ax[i, j].imshow(im)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "samples.png"), dpi=170)


if __name__ == "__main__":
    trajectory()
    samples()
    print("wrote", os.path.join(OUT, "trajectory.png"), os.path.join(OUT, "samples.png"))
