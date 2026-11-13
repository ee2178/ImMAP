"""Usage: python notes/immap_i2sb/equivalence_toy_iterates.py  (after equivalence_toy.py has trained bridge_a).

Run I2SB in x (DDPM bridge posterior) and the ascent in u side by side on the trained CIFAR
bridge network, with the same noise draws, and compare every iterate after mapping u -> x."""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equivalence_toy as E  # noqa: E402

dev = torch.device("cuda")
E.UNET_CH = 32
_, xte = E.load_cifar()
xtr_spread = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equivalence_toy", "results.json")))["std_x0_minus_x1"]
tau = xtr_spread / 2
m = E.build("bridge_a", tau)
m.load_state_dict(torch.load(os.path.join(E.HERE, "equivalence_toy", "ckpt", "bridge_a.pt"),
                             map_location=dev, weights_only=True))
m = m.to(dev).eval()
V = m.V
x0 = xte[:256].to(dev)
x1 = E.degrade(x0)
out = {}
for nfe in (5, 50):
    t_grid = [1.0 - k / nfe for k in range(nfe + 1)]          # 1 -> 0
    sig = lambda t: math.inf if t == 1.0 else math.sqrt(V * t / (1 - t))
    g = torch.Generator(device=dev).manual_seed(0)
    zs = [torch.randn(x0.shape, generator=g, device=dev) for _ in range(nfe)]
    with torch.no_grad():
        # I2SB in x
        x = x1.clone()
        xs = []
        for k in range(nfe):
            t, s = t_grid[k], t_grid[k + 1]
            xh = m._net(x, torch.full((x.shape[0],), t, device=dev), x1)
            if s == 0.0:
                x = xh
            else:
                x = (s / t) * x + (1 - s / t) * xh + math.sqrt(V * s * (t - s) / t) * zs[k]
            xs.append(x)
        # ascent in u
        u = torch.zeros_like(x1)
        us = []
        for k in range(nfe):
            t, s = t_grid[k], t_grid[k + 1]
            st, ss = sig(t), sig(s)
            v = m.denoise(u, st, x1)
            if s == 0.0:
                u = v
            else:
                r2 = 0.0 if math.isinf(st) else (ss / st) ** 2
                u = r2 * u + (1 - r2) * v + ss * math.sqrt(1 - r2) * zs[k]
            us.append((1 - s) * u + s * x1)                    # mapped to x
    rel = [((a - b).norm() / b.norm()).item() for a, b in zip(us, xs)]
    out[nfe] = {"max_rel": max(rel), "final_rel": rel[-1],
                "final_psnr_x": E.psnr(xs[-1], x0), "final_psnr_u": E.psnr(us[-1], x0)}
    print(nfe, out[nfe])
json.dump(out, open(os.path.join(E.HERE, "equivalence_toy", "iterate_check.json"), "w"), indent=1)
