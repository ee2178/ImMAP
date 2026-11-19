# -*- coding: utf-8 -*-
"""scripts/diagnose_guide_use.py: the attention-mass hook and the guide ablation, on small LGGS
nets (gather, and the materialised-triton construction through its CPU reference), with the
run-loading helpers stubbed so no trained run or data is needed."""
import sys
import types

import pytest
import torch

sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))

import tests.test_guided_recon as tgr                              # noqa: E402,F401  (its stubs)
import models.guided_prox as gp                                    # noqa: E402
import models.prox as prox_mod                                     # noqa: E402
import scripts.diagnose_guide_use as dg                            # noqa: E402
from models.guided_lpds import LGGSNet                             # noqa: E402

D = torch.float64


def cplx(*shape):
    return torch.complex(torch.randn(*shape, dtype=D), torch.randn(*shape, dtype=D))


def net(backend, K=5, dK=2):
    torch.manual_seed(0)
    return LGGSNet(K=K, M=12, C=1, P=5, s=2, Mh=4, window=1, guide_window=5, dK=dK, lam0=0.05,
                   preproc="image", attn_backend=backend, triton_materialize=True,
                   spectral_init=False).double().eval()


@pytest.fixture
def cpu_triton(monkeypatch):
    monkeypatch.setattr(prox_mod, "HAVE_TRITON", True)
    monkeypatch.setattr(gp, "_FUSED_JOINT_CHECKED", type("All", (set,), {
        "__contains__": lambda self, k: True})())


@pytest.mark.parametrize("backend,expect", [("triton", 5), ("gather", 3)])
def test_guide_mass_hook_records_and_restores(backend, expect, cpu_triton):
    """Fused: one record per layer (merge_joint runs every layer). Gather: one per adjacency
    build (layers 0, 2, 4 at dK=2). Every record is a share in [0, 1]; the hooks come off."""
    n = net(backend)
    y, g = cplx(1, 1, 16, 14), [cplx(1, 1, 16, 14)]
    sig = torch.full((1, 1, 1, 1), 0.05, dtype=D)
    orig = (gp.merge_joint, gp.GuidedGroupThreshold._gather_adjacencies)
    with torch.no_grad(), dg.GuideMass() as gm:
        out_hooked, _ = n(y, guide=g, sigma=sig)
    assert (gp.merge_joint, gp.GuidedGroupThreshold._gather_adjacencies) == orig
    assert len(gm.calls) == expect, gm.calls
    assert all(0.0 <= c <= 1.0 for c in gm.calls)
    with torch.no_grad():
        out_plain, _ = n(y, guide=g, sigma=sig)
    assert torch.equal(out_hooked, out_plain)                       # observing changes nothing


def test_ablations():
    g = cplx(1, 1, 6, 5)
    assert dg._ablate(g, "true") is g
    assert torch.equal(dg._ablate(g, "rotated"), torch.flip(g, dims=(-2, -1)))
    assert not dg._ablate(g, "zero").any()
    assert torch.equal(dg._ablate(g, "x0.35"), g * 0.35)
    g5 = cplx(1, 3, 1, 6, 5)                                        # N_g guide slices
    assert torch.equal(dg._ablate(g5, "rotated")[:, 1], torch.flip(g5[:, 1], dims=(-2, -1)))


def test_diagnose_end_to_end_with_stubbed_run(monkeypatch, cpu_triton):
    """The whole flow on a stub run: the three guides are scored, the rotated and zero guides
    change the recon, and the guide mass is reported per layer."""
    n = net("triton", K=4, dK=1).float()                 # fp32, as trained runs (and SSIM's window)
    torch.manual_seed(1)
    c64 = lambda: cplx(1, 1, 16, 14).to(torch.complex64)            # noqa: E731
    items = [(c64(), c64()) for _ in range(3)]
    run = dict(cfg={"training": {"metric_range": "gt_max"}}, net=n, device="cpu", name="stub")
    ident = types.SimpleNamespace(forward=lambda x: x)
    monkeypatch.setattr(dg, "load_recon_run", lambda rd, device=None: run)
    monkeypatch.setattr(dg, "val_subset", lambda cfg: items)
    monkeypatch.setattr(dg, "_wants_guide", lambda net: True)
    monkeypatch.setattr(dg, "measure", lambda run, item, noise=None, seed=0: dict(
        y=item[0], E=None, T=ident, sigma=torch.full((1, 1, 1, 1), 0.05),
        guide=item[1], gt=item[0]))
    recons = []
    real = dg.reconstruct
    monkeypatch.setattr(dg, "reconstruct", lambda r, m: recons.append(real(r, m)) or recons[-1])
    res = dg.diagnose("stub", n=3, noise=None, device="cpu")
    assert set(res) >= {"true", "rotated", "zero", "x0.35", "x0.1", "guide_mass_per_layer"}
    assert "x0.35_minus_true_psnr" in res
    assert len(res["guide_mass_per_layer"]) == 4                    # one per layer
    assert len(res["rotated_minus_true_psnr"]) == 3
    assert not torch.equal(recons[0], recons[1]) and not torch.equal(recons[0], recons[2])


def test_unguided_lggs_is_the_same_net_without_its_guide():
    """LGGSNet(guided=False), the lggs_noguide control: the trainer sends no guide, and a guide
    handed to it anyway changes nothing -- every prox runs its self-only form."""
    from training.recon import _wants_guide
    torch.manual_seed(0)
    kw = dict(K=4, M=12, C=1, P=5, s=2, Mh=4, window=1, guide_window=5, dK=2, lam0=0.05,
              preproc="image", attn_backend="gather", spectral_init=False)
    off = LGGSNet(guided=False, **kw).double()
    on = LGGSNet(**kw).double()
    on.load_state_dict(off.state_dict())                            # same weights
    assert not _wants_guide(off) and _wants_guide(on)
    y, g = cplx(1, 1, 16, 14), [cplx(1, 1, 16, 14)]
    sig = torch.full((1, 1, 1, 1), 0.05, dtype=D)
    a, _ = off(y, guide=g, sigma=sig)
    b, _ = off(y, guide=None, sigma=sig)
    c, _ = on(y, guide=None, sigma=sig)
    d, _ = on(y, guide=g, sigma=sig)
    assert torch.equal(a, b) and torch.equal(b, c)                  # guide ignored = no guide
    assert not torch.allclose(c, d)                                 # ... and the guide matters


def test_diff_runs_reports_configs_tying_and_degree(tmp_path, capsys):
    """Two fake run dirs: a tied LGGS (noise-adaptive threshold) and the same net's state dict
    with layer 1's W's changed (an untied, pre-2026-11-09 run) under another run tag."""
    import json
    import scripts.diff_runs as dr
    torch.manual_seed(0)
    tied = LGGSNet(K=3, M=12, C=1, P=5, s=2, Mh=4, window=1, guide_window=5, preproc="image",
                   attn_backend="gather", spectral_init=False, degrees=0, prox_degrees=1)
    sd_tied = tied.state_dict()
    sd_untied = {k: v.clone() for k, v in sd_tied.items()}
    key = next(k for k in sd_untied if k.startswith("layers.1.") and ".Walpha." in k)
    sd_untied[key] += 1.0
    for name, sd, tag in (("a", sd_tied, "s4"), ("b", sd_untied, "s3")):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"paths": {"save_dir": f"trained_nets/{tag}/x"}, "mri": {"guide_R": 1 if name == "a" else 8},
             "training": {"clip_grad": 0.05}}))
        torch.save({"model_state_dict": sd, "step": 100}, d / "net.ckpt")
    dr.main([str(tmp_path / "a"), str(tmp_path / "b")])
    out = capsys.readouterr().out
    assert "paths.save_dir" in out and "mri.guide_R" in out
    assert "training.clip_grad" not in out                          # equal keys are not listed
    assert "NO (trained untied" in out and "yes" in out
    assert "prox tau degree" in out
