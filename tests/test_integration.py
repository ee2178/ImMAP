"""
End-to-end checks: the ported networks actually train, and every construction
variant builds and runs.

Run with:  python -m tests.test_integration
"""

import sys

import torch

from models import build_model
from models.multigrid import MGCDLNet
from operators import Identity
from operators.noise import awgn

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
def test_variants():
    """Every switch in the port must construct and run a forward+backward."""
    base = dict(M=4, C=1, P=5, s=1, is_complex=False)
    variants = {
        "3-level V-cycle":        dict(K=[2, [4, 4, 2]], **base),
        "widen=2":                dict(K=[1, [2, 2]], widen=2, **base),
        "group prox (W=5)":       dict(K=[1, [2, 2]], W=5, Mh=2, **base),
        "dual / Fenchel prox":    dict(K=[1, [2, 2]], dual=True, **base),
        "simple subgradient":     dict(K=[1, [2, 2]], W=5, Mh=2,
                                       subgrad_mode="simple", **base),
        "moreau subgradient":     dict(K=[1, [2, 2]], W=5, Mh=2,
                                       subgrad_mode="moreau", **base),
        "alpha as a scale":       dict(K=[1, [2, 2]], alpha_conv=False, **base),
        "julia_compat restrict":  dict(K=[1, [2, 2]], julia_compat=True, **base),
        "noise-adaptive (deg 1)": dict(K=[1, [2, 2]], degrees=1, eta_degrees=1,
                                       **base),
        "identity preproc":       dict(K=[1, [2, 2]], preproc="identity", **base),
        "stride 2":               dict(K=[1, [2, 2]], **{**base, "s": 2}),
        "complex":               dict(K=[1, [2, 2]], **{**base, "is_complex": True}),
        "plain (no multigrid)":   dict(K=3, **base),
    }
    y = torch.randn(2, 1, 32, 32)
    yc = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    for name, kws in variants.items():
        try:
            net = MGCDLNet(**kws)
            inp = yc if kws.get("is_complex", False) else y
            out, z = net(inp, E=Identity(), sigma=0.05)
            out.abs().sum().backward()
            net.project()
            ok = out.shape == inp.shape and torch.isfinite(out.abs()).all().item()
            check(name, ok, str(tuple(out.shape)))
        except Exception as exc:                                  # noqa: BLE001
            check(name, False, f"{type(exc).__name__}: {exc}")


def test_odd_iters_rejected():
    try:
        MGCDLNet(K=[1, [3, 2]], M=4, C=1, P=5, is_complex=False)
        check("odd non-coarsest iters are rejected", False)
    except AssertionError:
        check("odd non-coarsest iters are rejected", True)


def test_training_reduces_loss():
    """A few Adam steps on a toy denoising problem must reduce the loss."""
    torch.manual_seed(1)
    net = MGCDLNet(K=[2, [4, 4, 2]], M=8, C=1, P=5, s=1, is_complex=False)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    # a fixed toy target: piecewise-constant blobs
    gt = torch.zeros(4, 1, 32, 32)
    gt[:, :, 8:24, 8:24] = 1.0
    gt = gt + 0.1 * torch.randn_like(gt)

    losses = []
    for step in range(25):
        noisy, sigma = awgn(gt, [0.1, 0.1])
        recon, _ = net(noisy, E=Identity(), sigma=sigma)
        loss = torch.mean((recon - gt) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net.project()
        losses.append(loss.item())

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    check("training reduces the loss", last < first,
          f"{first:.4e} -> {last:.4e}")
    check("no NaNs during training", all(l == l for l in losses))
    check("constraints hold after training",
          all((p >= 0).all() for n, p in net.named_parameters()
              if n.endswith("tau.weight")))


def test_ladmm_trains():
    from models.ladmm import AltSplitCDLNet
    from operators import FFT2D, Mask, Sense

    torch.manual_seed(2)
    net = AltSplitCDLNet(admm_iters=2, cg_maxit=4, implicit_cg=True,
                         denoiser_kws=dict(K=[1, [2, 2]], M=4, C=1, P=5, s=1,
                                           is_complex=True))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    smaps = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    smaps = smaps / smaps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = (torch.rand(1, 1, 16, 16) > 0.4).to(torch.complex64)
    E = Mask(mask) @ FFT2D() @ Sense(smaps)
    gt = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    y = E(gt)

    losses = []
    for _ in range(15):
        recon, _ = net(y, E=E, sigma=torch.tensor([0.01]))
        loss = torch.mean((recon - gt).abs() ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net.project()
        losses.append(loss.item())
    check("LADMM training reduces the loss", losses[-1] < losses[0],
          f"{losses[0]:.4e} -> {losses[-1]:.4e}")
    check("LADMM training stays finite", all(l == l for l in losses))


def test_configs_build():
    """Every checked-in multigrid config must build, and be shaped for its task.

    The generated fastMRI grid (`scripts/make_mg_recon_configs.py`) is globbed
    rather than listed: a new cell is covered the moment it is generated.
    """
    import glob
    import json

    paths = (["config/bsd432/mg_denoiser.json", "config/knee/ladmm.json"]
             + sorted(glob.glob("config/*/mg/*.json")))

    for path in paths:
        cfg = json.load(open(path))
        try:
            net = build_model(cfg)
        except Exception as exc:                                  # noqa: BLE001
            check(f"build {path}", False, f"{type(exc).__name__}: {exc}")
            continue

        n = sum(p.numel() for p in net.parameters())
        check(f"build {path} -> {cfg['model']['type']}", n > 0,
              f"{n / 1e6:.2f}M params")

        # preproc='image' pads y~ while E's mask and maps stay put, and its
        # plain mean subtraction is the wrong DC model once E is not identity.
        if cfg.get("task") == "recon" and cfg["model"]["type"] != "AltSplitCDLNet":
            check(f"{path} uses preproc='kspace' or 'identity'",
                  cfg["model"]["params"].get("preproc") in ("kspace", "identity"),
                  str(cfg["model"]["params"].get("preproc")))

        # train.py indexes these blocks unconditionally per task.
        required = {"recon": ("data", "training", "mri", "optimizer",
                              "scheduler", "paths", "wandb"),
                    "denoiser": ("data", "training", "optimizer", "scheduler",
                                 "paths", "wandb")}.get(cfg.get("task"), ())
        missing = [k for k in required if k not in cfg]
        check(f"{path} has every block train.py reads", not missing, str(missing))


if __name__ == "__main__":
    for fn in (test_variants, test_odd_iters_rejected, test_training_reduces_loss,
               test_ladmm_trains, test_configs_build):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
