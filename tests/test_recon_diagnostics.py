"""Tests of scripts/smaps_check.py and scripts/inspect_tgvn_trust.py on synthetic paired data.

    python tests/test_recon_diagnostics.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_guided_recon as tg            # noqa: E402  (stubs + paired fixtures)

import torch                              # noqa: E402

sys.path.insert(0, os.path.join(tg.REPO, "scripts"))
import smaps_check as sc                  # noqa: E402
import inspect_tgvn_trust as it           # noqa: E402

MRI = {"R": 4, "acs_lines": 4, "mask_dist": "uniform", "mask_offset": 0,
       "online_smaps": "espirit", "online_smaps_kws": {}}
TRAIN = {"metric_range": 1.0}


def _slice(root):
    csv, _ = tg.make_data(root)
    maps = tg.make_maps(root, os.path.join(root, "maps"))
    ds = tg.PairedKneeDataset(csv, root, scale_fac=1.0, target="sense", smap_root=maps,
                              start_slice=1, end_slice=2)
    k, s, img, *_ = ds[0]
    return k[None], s[None], img[None]


def _with_online(maps_fn, fn):
    import physics.online_smaps as osm
    real = osm.online_smaps
    osm.online_smaps = maps_fn
    try:
        return fn()
    finally:
        osm.online_smaps = real


def test_smaps_check_identical_maps_score_identically():
    """Online maps = the offline ones: full/online is full/offline, both rebuild the reference,
    and the map error is 0."""
    with tempfile.TemporaryDirectory() as root:
        k, s, img = _slice(root)
        res, ext = _with_online(lambda y, mask, **kw: s.clone(),
                                lambda: sc.evaluate_slice(k, s, img, MRI, TRAIN, lams=(1e-3,),
                                                          cg_iters=30))
    a, b = res["full/offline"]["support"], res["full/online"]["support"]
    assert abs(a["psnr"] - b["psnr"]) < 1e-6 and a["psnr"] > 80, (a, b)
    assert ext["map_error_support"] < 1e-5
    assert abs(res["cg0.001/online"]["fov"]["psnr"] - res["cg0.001/offline"]["fov"]["psnr"]) < 1e-6
    assert res["cg0.001/offline"]["fov"]["psnr"] > res["zf/online"]["fov"]["psnr"]


def test_smaps_check_sees_wrong_maps_and_ignores_the_espirit_phase():
    """A per-pixel common phase on the online maps (ESPIRiT's ambiguity) costs nothing in
    magnitude and reads as zero map error; perturbed maps cost PSNR and read as error."""
    with tempfile.TemporaryDirectory() as root:
        k, s, img = _slice(root)
        H, W = s.shape[-2:]
        ph = torch.exp(1j * torch.linspace(0, 3, H * W).reshape(1, 1, H, W)).to(s.dtype)
        res_ph, ext_ph = _with_online(lambda y, mask, **kw: s * ph,
                                      lambda: sc.evaluate_slice(k, s, img, MRI, TRAIN, lams=()))
        torch.manual_seed(0)
        bad = s + 0.3 * torch.randn_like(s) * (s.abs() > 0)
        bad = bad / bad.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
        res_bad, ext_bad = _with_online(lambda y, mask, **kw: bad,
                                        lambda: sc.evaluate_slice(k, s, img, MRI, TRAIN, lams=()))
    assert ext_ph["map_error_support"] < 1e-5
    assert res_ph["full/online"]["support"]["psnr"] > 80
    assert ext_bad["map_error_support"] > 0.05
    assert res_bad["full/online"]["support"]["psnr"] < res_ph["full/online"]["support"]["psnr"] - 10


def _tiny_tgvn():
    from models.tgvn import TGVN
    torch.manual_seed(0)
    return TGVN(guided=True, cond_sigma=True, use_smaps=True, **tg.TINY_TGVN).eval()


def _measure_like(k, s, img):
    from operators import FFT2D, Mask, Sense
    from physics.mask import get_mask_cached
    mask = get_mask_cached(img, R=4, acs_lines=4, mode="uniform", offset=0)
    side = k * 0.7 + 0.05 * torch.randn_like(k)          # a stand-in guide scan
    return dict(y=mask * k, E=Mask(mask) @ FFT2D() @ Sense(s), sigma=torch.full((1,), 0.01),
                guide=side)


def test_trace_reproduces_tgvn_and_records_every_cascade():
    net = _tiny_tgvn()
    with tempfile.TemporaryDirectory() as root:
        k, s, img = _slice(root)
    m = _measure_like(k, s, img)
    with torch.no_grad():
        ref, _ = net(m["y"], E=m["E"], sigma=m["sigma"], guide=m["guide"])
        with it.TraceTGVN(net) as tr:
            out, _ = net(m["y"], E=m["E"], sigma=m["sigma"], guide=m["guide"])
    assert torch.equal(ref, out)                                   # the trace changes nothing
    assert tr.max_rel_diff < 1e-5, tr.max_rel_diff
    assert len(tr.records) == tg.TINY_TGVN["num_cascades"]
    assert not any("forward" in b.__dict__ for b in tr.blocks)       # unpatched afterwards
    r = tr.records[0]
    x_next = r["x_in"] - r["soft_dc"] - r["soft_asc"] - r["model"]
    assert torch.allclose(x_next, r["x_out"], atol=1e-5)
    # the guide as the net combines it: S^H F^-1 w with the operator's maps
    from operators.fourier import ifftc
    g = (s.conj() * ifftc(m["guide"])).sum(1)
    assert torch.allclose(r["guide"], g, atol=1e-4)
    stats = it.cascade_stats(tr.records, img[:, 0], img[:, 0].abs() > 0)
    assert len(stats) == len(tr.records)
    assert all(0.0 <= s_["asc_measured"] <= 1.0 for s_ in stats)
    assert all(-1.0 <= s_["corr_H_target"] <= 1.0 for s_ in stats)


def test_figure_is_written():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("    (skipped: no matplotlib)")
        return
    net = _tiny_tgvn()
    with tempfile.TemporaryDirectory() as root:
        k, s, img = _slice(root)
        m = _measure_like(k, s, img)
        with torch.no_grad(), it.TraceTGVN(net) as tr:
            out, _ = net(m["y"], E=m["E"], sigma=m["sigma"], guide=m["guide"])
        p = os.path.join(root, "f.png")
        it.save_figure(p, tr.records, img[0, 0], out[0, 0])
        assert os.path.getsize(p) > 1000


def main():
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                             # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc(limit=6)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
