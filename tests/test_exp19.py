"""
exp19's three cells and the diagnostic that reads them.

Run with `python -m tests.test_exp19`. CPU, about a minute.

1. Folded mask: exact without coils (levels 1, 2), closer to Galerkin than the crop
   with coils, also on an embedded operator, and Hermitian with a true adjoint.
2. `alpha_z_zero`: same output as a free net with alpha_z = 0, planar too; alpha_z
   gets no gradient, and a free checkpoint loads.
3. The generated configs build, take a training step on an embedded operator, and
   save and load. The default grid's cell order is untouched.
4. scripts/coarse_diag.py: the spectrum bins, hooks that knock out exactly what they
   say and come off afterwards, and a whole pass through `run_diag` + `combine` on a
   synthetic slice with the real adapter and metrics.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

import torch

from models import build_model
from models.mg_lpds import MGLPDSNet, PDVCycle
from operators import FFT2D, Mask, Sense
from operators.coarse import coarse_sense, coarsen, rediscretize
from operators.resample import galerkin
from operators.truncate import embed_operator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import coarse_diag as cd                                   # noqa: E402
import make_mg_recon_configs as gen                        # noqa: E402

FAIL = []
F = "sinc_vertex"
CDT = torch.complex128


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def coils(h, w, c=8, seed=0, dtype=CDT):
    g = torch.Generator().manual_seed(seed)
    yy = (torch.arange(h)[:, None] - h / 2).double()
    xx = (torch.arange(w)[None, :] - w / 2).double()
    sm = []
    for i in range(c):
        a = 2 * math.pi * i / c
        cy, cx = 0.55 * h * math.sin(a), 0.55 * w * math.cos(a)
        mag = torch.exp(-((yy - cy) ** 2 / (2 * (0.35 * h) ** 2) + (xx - cx) ** 2 / (2 * (0.35 * w) ** 2)))
        ph = 2 * math.pi * torch.rand(1, generator=g, dtype=torch.float64) + 0.01 * (
            yy * math.cos(a) - xx * math.sin(a))
        sm.append(mag * torch.exp(1j * ph))
    sm = torch.stack(sm)
    return (sm / sm.abs().pow(2).sum(0, keepdim=True).sqrt()).to(dtype)[None]


def line_mask(h, w, R=8, acs=13, dtype=torch.float64):
    m = torch.zeros(w, dtype=dtype)
    m[::R] = 1
    m[w // 2 - acs // 2: w // 2 + acs // 2 + 1] = 1
    return m[None, None, None, :]             # broadcast along readout, as the grid's


def noise(shape, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g, dtype=torch.float64),
                         torch.randn(shape, generator=g, dtype=torch.float64))


def structure(shape):
    h, w = shape[-2:]
    yy = (torch.arange(h)[:, None] - h / 2).double()
    xx = (torch.arange(w)[None, :] - w / 2).double()
    head = ((yy / (0.42 * h)) ** 2 + (xx / (0.36 * w)) ** 2 < 1).double()
    return (head * (1 + 0.3 * torch.cos(xx / 3) * torch.sin(yy / 5))).to(CDT)[None, None]


def test_folded():
    N = 160
    S1 = torch.ones(1, 1, N, N, dtype=CDT)
    E = Mask(line_mask(N, N)) @ FFT2D() @ Sense(S1)
    gal, fold = E, E
    for level in (1, 2):
        gal, fold = galerkin(gal, filter=F), coarsen(fold, "folded", filter=F)
        x = noise((1, 1, N >> level, N >> level))
        err = rel(fold.gram(x), gal.gram(x))
        check(f"no coils, level {level}: the folded Gram IS Galerkin's", err < 1e-9, f"{err:.1e}")

    S = coils(N, N)
    E = Mask(line_mask(N, N)) @ FFT2D() @ Sense(S)
    gal, fold, crop = E, E, E
    for level in (1, 2):
        gal = galerkin(gal, filter=F)
        fold, crop = coarsen(fold, "folded", filter=F), coarsen(crop, "rediscretize", filter=F)
        grid = (1, 1, N >> level, N >> level)
        for name, x in (("noise", noise(grid)), ("structure", structure(grid))):
            ef, ec = rel(fold.gram(x), gal.gram(x)), rel(crop.gram(x), gal.gram(x))
            check(f"coils, level {level}, {name}: folded beats the crop", ef < ec / 2,
                  f"folded {ef:.4f}, crop {ec:.4f}")

    # knee's size class: measured 92 wide, embedded to 96, going 92 -> 46 -> 23
    h, w = 160, 92
    E_t, T = embed_operator(Mask(line_mask(h, w)) @ FFT2D() @ Sense(coils(h, w)), (h, w), 8)
    gal, fold, crop = E_t, E_t, E_t
    for level in (1, 2):
        gal = galerkin(gal, filter=F)
        fold = rediscretize(fold, filter=F, folded=True)
        crop = rediscretize(crop, filter=F)
        check(f"embedded, level {level}: expressible", fold is not None)
        grid = (1, 1, h >> level, T.big[1] >> level)
        for name, x in (("noise", noise(grid)), ("structure", structure(grid))):
            ef, ec = rel(fold.gram(x), gal.gram(x)), rel(crop.gram(x), gal.gram(x))
            check(f"embedded, level {level}, {name}: folded beats the crop", ef < ec,
                  f"folded {ef:.4f}, crop {ec:.4f}")
    a, b = noise(grid, 5), noise(grid, 6)
    herm = abs(complex(torch.sum(fold.gram(a).conj() * b) - torch.sum(a.conj() * fold.gram(b))))
    herm /= abs(complex(torch.sum(a.conj() * fold.gram(b))))
    check("embedded, level 2: Hermitian on the odd measured grid", herm < 1e-10, f"{herm:.1e}")

    full = torch.ones(1, 1, N, N, dtype=torch.float64)
    _, m_c, _ = coarse_sense(full, coils(N, N), filter=F, folded=True)
    # per axis |H(pi/2)|^2 + |H(-pi/2)|^2 = 1/2, so sqrt(D_c) is 1/2 at the Nyquist corner
    check("full sampling: sqrt(D_c) is 1 at DC and 1/2 at the coarse Nyquist corner",
          abs(float(m_c[0, 0, N // 4, N // 4]) - 1) < 1e-6 and abs(float(m_c.min()) - 0.5) < 0.01,
          f"min {float(m_c.min()):.3f}")
    bool_m = line_mask(N, N, dtype=torch.float32) > 0
    _, m_b, _ = coarse_sense(bool_m, coils(N, N), filter=F, folded=True)
    check("a boolean mask folds too", m_b.is_floating_point() and m_b.shape[-2:] == (N // 2, N // 2))


def _net(**kw):
    torch.manual_seed(4)
    return MGLPDSNet(K=[2, [2, 2, 2]], M=4, C=1, P=5, s=2, is_complex=True, preproc="identity",
                     lam0=0.05, alpha0=0.3, transfer_filter=F, fas_dual="fixed_point",
                     fas_copies="untied", fas_parent_pi=True, **kw)


Y = torch.randn(1, 1, 32, 32, dtype=torch.complex64, generator=torch.Generator().manual_seed(1))
SIG = torch.tensor(0.015)


def _alpha_z(net):
    return [m.alpha_z for m in net.modules() if isinstance(m, PDVCycle)]


def test_alpha_z_zero():
    def out(net):
        with torch.no_grad():
            return net(Y, sigma=SIG)[0]

    zero, ref = _net(alpha_z_zero=True), _net()
    with torch.no_grad():
        for m in _alpha_z(ref):
            m.weight.zero_()
    check("alpha_z_zero == a free net with alpha_z = 0", torch.equal(out(zero), out(ref)))
    check("...and differs from the free net", not torch.equal(out(zero), out(_net())))
    prev = MGLPDSNet.PLANAR_STATE
    try:
        MGLPDSNet.PLANAR_STATE = True
        planar = _net(alpha_z_zero=True)
        gap = float((out(planar) - out(ref).detach()).abs().max())
        check("planar state: same output", planar.planar_state_active() and gap < 1e-4, f"{gap:.1e}")
    finally:
        MGLPDSNet.PLANAR_STATE = prev
    net = _net(alpha_z_zero=True)
    net(Y, sigma=SIG)[0].abs().pow(2).mean().backward()
    check("alpha_z: no gradient, alpha_x has one",
          all(m.weight.grad is None and not m.weight.requires_grad for m in _alpha_z(net))
          and all(m.alpha_x.weight.grad is not None for m in net.modules() if isinstance(m, PDVCycle)))
    free = _net()
    z2 = _net(alpha_z_zero=True)
    z2.load_state_dict(free.state_dict())
    z2.project()
    check("a free checkpoint loads, and project() puts alpha_z back at 0",
          all(float(m.weight.abs().max()) == 0 for m in _alpha_z(z2)))
    check("no alpha_z series in the logs", not z2.param_logs())
    try:
        _net(alpha_z_zero=True, widen=2)
        check("refused with widen 2", False)
    except ValueError:
        check("refused with widen 2", True)


def _config(cell, tmp, K=(2, (2, 2, 2)), M=9, P=5):
    """The cell's config as the launcher generates it, on a smaller net."""
    out = os.path.join(tmp, "config")
    if not os.path.isdir(out):
        subprocess.run([sys.executable, "scripts/make_mg_recon_configs.py", "--out", out,
                        "--anatomy", "knee", "--attn", "flex", "--organ-mask", "--planar-state"],
                       check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(out, "knee", "mg", f"{cell}_R8.json")) as f:
        cfg = json.load(f)
    cfg["model"]["params"].update(K=[K[0], list(K[1])], M=M, P=P)
    return cfg


def test_cells(tmp):
    want = dict(mg81v6vtxAnestUgalcf32=("galerkin", False), mg81v6vtxAnestUfoldcf32=("folded", False),
                mg81v6vtxAnestUaz0cf32=("rediscretize", True))
    base = gen.MODELS["mg81v6vtxAnestUcf32"]
    names = {}
    for cell, (op, az0) in want.items():
        spec = gen.MODELS[cell]
        diff = {k for k in set(spec["params"]) | set(base["params"])
                if spec["params"].get(k) != base["params"].get(k)}
        check(f"{cell}: differs from the baseline in one key",
              len(diff) == 1 and spec["center_frac_times_r"] == 0.32 and spec["adjust_accel"] is False,
              str(diff))
        check(f"{cell}: not in the default grid", cell in gen.OPT_IN)
        cfg = _config(cell, tmp)
        names[cell] = cfg["experiment"]["name"]
        p = cfg["model"]["params"]
        check(f"{cell}: config says what it should",
              p["coarse_op"] == op and bool(p.get("alpha_z_zero")) == az0
              and p["transfer_filter"] == F and cfg["mri"]["center_frac"] == 0.04
              and not cfg["mri"].get("adjust_accel"))

        net = build_model(cfg)
        h, w = 64, 44                      # embedded to 64 x 48, like knee's 372 -> 376
        s = coils(h, w, c=4, dtype=torch.complex64)
        E, T = embed_operator(Mask(line_mask(h, w, dtype=torch.float32)) @ FFT2D() @ Sense(s), (h, w), 8)
        x = structure((1, 1, h, w)).to(torch.complex64) * 0.1
        y = (Mask(line_mask(h, w, dtype=torch.float32)) @ FFT2D() @ Sense(s))(x)
        out = net(y, E=E, sigma=torch.tensor(0.015))[0]
        loss = (T.forward(out) - x).abs().pow(2).mean()
        loss.backward()
        ok = math.isfinite(float(loss)) and all(
            q.grad is None or bool(torch.isfinite(q.grad.real).all()) for q in net.parameters())
        opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=1e-4)
        opt.step()
        net.project()
        check(f"{cell}: a training step on an embedded operator", ok, f"loss {float(loss):.4f}")
    check("three wandb names, none the baseline's",
          len(set(names.values())) == 3
          and _config("mg81v6vtxAnestUcf32", tmp)["experiment"]["name"] not in names.values(),
          str(sorted(names.values())))


def test_spectrum():
    H, W = 32, 48
    yy = torch.arange(H)[:, None].double()
    low = torch.exp(2j * math.pi * 2 * yy / H).expand(H, W).to(torch.complex64)[None, None]
    xx = torch.arange(W)[None, :].double()
    hi = torch.exp(2j * math.pi * 18 * xx / W).expand(H, W).to(torch.complex64)[None, None]
    e_low, e_hi = cd.band_energy(low), cd.band_energy(hi)
    check("a pi/8 wave is in bin 1, a 3pi/4 wave in bin 6",
          e_low.argmax() == 1 and e_hi.argmax() == 6, f"{e_low.argmax()}, {e_hi.argmax()}")
    r = cd.spectrum_row(hi, low)
    check("hf = 1 for the high wave, rel_norm against the state",
          abs(r["hf"] - 1) < 1e-6 and abs(r["rel_norm"] - 1) < 1e-5)
    planar = torch.cat((hi.real, hi.imag), dim=1)
    check("a planar tensor reads the same",
          abs(float(cd.band_energy(planar).sum() / e_hi.sum()) - 1) < 1e-5)


def test_hooks():
    net = _net()
    with torch.no_grad():
        full = net(Y, sigma=SIG)[0]
    before = {k: v.clone() for k, v in net.state_dict().items()}
    outs, spectra = cd.run_knockouts(net, Y, None, SIG)
    check("full == the unhooked net", torch.equal(outs["full"], full))
    ref = _net()
    with torch.no_grad():
        for m in _alpha_z(ref):
            m.weight.zero_()
        check("az0 == alpha_z weights at zero", torch.equal(outs["az0"], ref(Y, sigma=SIG)[0]))
        ref = _net()
        for V in ref.modules():
            if isinstance(V, PDVCycle):
                V.alpha_x.weight.zero_()
        check("ax0 == alpha_x weights at zero", torch.equal(outs["ax0"], ref(Y, sigma=SIG)[0]))
        ref = _net()
        for k, level, V in ref.vcycles():
            if level == 1:
                V.alpha_z.weight.zero_()
        check("az0_L1 touches only L2 -> L1", torch.equal(outs["az0_L1"], ref(Y, sigma=SIG)[0]))
    check("each knockout changes the output",
          all(not torch.equal(outs[k], full) for k in cd.KNOCKOUTS if k != "full"))
    check("hooks removed and weights untouched",
          torch.equal(net(Y, sigma=SIG)[0].detach(), full)
          and all(torch.equal(v, before[k]) for k, v in net.state_dict().items()))
    n = {p: sum(r["path"] == p for r in spectra) for p in ("state", "x", "z")}
    check("one row per V-cycle, level and path (2 x 2)", n == dict(state=4, x=4, z=4), str(n))
    ref = _net()
    a = [r for r in spectra if r["path"] == "x" and r["outer"] == 0 and r["level"] == 0][0]
    check("spectra rows are finite", all(math.isfinite(r["hf"]) and math.isfinite(r["rel_norm"])
                                         for r in spectra), str(a["hf"]))
    z_net = _net(alpha_z_zero=True)
    _, sp = cd.run_knockouts(z_net, Y, None, SIG, names=("full",))
    check("alpha_z_zero: no z rows", not [r for r in sp if r["path"] == "z"]
          and len([r for r in sp if r["path"] == "x"]) == 4)


def test_rho():
    """A ghost in the residual reads as high rho, an uncorrelated residual does not, and
    a residual too weak against the full net's scale reads as -1 (alias_check's meaning)."""
    H, W = 160, 128
    g = torch.Generator().manual_seed(2)
    yy = torch.arange(H)[:, None].double() - H / 2
    xx = torch.arange(W)[None, :].double() - W / 2
    ref = 0.5 + 0.4 * ((yy / 50) ** 2 + (xx / 8) ** 2 < 1).double()
    noise_ = lambda s: s * torch.randn(H, W, generator=g, dtype=torch.float64)      # noqa: E731
    ghost = torch.roll(((yy / 50) ** 2 + (xx / 8) ** 2 < 1).double(), W // 4, dims=1)
    img = lambda t: t.to(torch.complex64)[None, None]                               # noqa: E731
    gt, zf = img(ref), img(ref + 0.5 * ghost + noise_(0.005))
    outs = dict(full=img(ref + noise_(0.01)), az0=img(ref + 0.2 * ghost + noise_(0.01)),
                ax0=img(ref + noise_(0.2)), ax0az0=img(ref + noise_(0.01)))
    organ = torch.ones(1, 1, H, W)
    rows = {r["knockout"]: r for r in cd.score_slice(gt, outs, zf, organ, cd.build_metrics(cd.METRICS),
                                                     cd.parser().parse_args(["--run", "x"]))}
    check("rho: a ghost is high, unrelated noise is not",
          rows["az0"]["rho"] > 0.8 and rows["az0"]["rho"] > rows["ax0"]["rho"] + 0.3,
          f"ghost {rows['az0']['rho']:.2f}, noise {rows['ax0']['rho']:.2f}")
    check("rho: the extra error shows as strength on the full net's scale",
          rows["ax0"]["strength"] > 5 * max(rows["full"]["strength"], 1e-3)
          or rows["full"]["rho"] == -1, f"{rows['ax0']['strength']:.1f} vs {rows['full']['strength']:.1f}")
    check("PSNR orders the same way", rows["ax0"]["psnr"] < rows["az0"]["psnr"] < rows["full"]["psnr"])


def test_ghost_fraction():
    """beta is exactly 1 on the zero-filled image and 0 on the reference, tells the two apart
    in between, does not count a blurred edge, and shrugs off a lattice tone."""
    H, W = 160, 128
    g = torch.Generator().manual_seed(3)
    yy = torch.arange(H)[:, None].double() - H / 2
    xx = torch.arange(W)[None, :].double() - W / 2
    disc = ((yy / 50) ** 2 + (xx / 8) ** 2 < 1).double()
    ref = (0.5 + 0.4 * disc).numpy()
    ghost = torch.roll(disc, W // 4, dims=1).numpy()
    zf = 0.6 * ref + 0.3 * ghost + 0.002 * torch.randn(H, W, generator=g, dtype=torch.float64).numpy()
    G = cd.ghost_template(ref, zf)
    all_ = (ref > 0)
    b = lambda im: cd.ghost_fraction(im, ref, G, all_)                       # noqa: E731
    check("beta: zero-filled ~ 1, reference = 0", abs(b(zf) - 1) < 0.02 and abs(b(ref)) < 1e-9,
          f"{b(zf):.4f}, {b(ref):.4f}")
    half = ref + 0.5 * (zf - 0.6 * ref)
    check("beta: half the ghost reads about 1/2 of it", 0.3 < b(half) < 0.7, f"{b(half):.3f}")
    blur = ref - 0.1 * cd.hp_pe(ref)
    check("beta: an edge error does not read as ghost", abs(b(blur)) < 0.05, f"{b(blur):.3f}")
    n_y, n_x = yy.numpy(), xx.numpy()
    for name, t in (("pi along PE", (-1.0) ** (n_x % 2)), ("pi along readout", (-1.0) ** (n_y % 2)),
                    ("checkerboard", (-1.0) ** ((n_x + n_y) % 2)),
                    ("pi/2 along PE", torch.cos(torch.pi / 2 * xx).numpy())):
        tone = ref + 0.2 * t + 0 * n_y
        check(f"beta: a lattice tone does not read as ghost ({name})", abs(b(tone)) < 0.02,
              f"{b(tone):.4f}")


def test_trajectory():
    net = _net()
    with torch.no_grad():
        full = net(Y, sigma=SIG)[0]
    out, rows = cd.run_trajectory(net, Y, None, SIG)
    check("trajectory: the forward is unchanged, and the last post is the output",
          torch.equal(out, full) and torch.equal(rows[-1][2], full))
    check("trajectory: four stages per fine V-cycle", [r[1] for r in rows[:4]] == ["pre", "+x", "+z", "post"]
          and len(rows) == 8 and {r[0] for r in rows} == {0, 1}, str([r[1] for r in rows[:4]]))
    az0 = _net(alpha_z_zero=True)
    _, rows = cd.run_trajectory(az0, Y, None, SIG)
    check("trajectory: alpha_z_zero has no +z", not [r for r in rows if r[1] == "+z"] and len(rows) == 6)


def test_whole_pass(tmp):
    """`run_diag` + `combine` with the real adapter, measurement and metrics; only the
    loader is synthetic, there being no fastMRI here."""
    root = os.path.join(tmp, "run")
    os.makedirs(root, exist_ok=True)
    h, w = 64, 44
    cfg = _config("mg81v6vtxAnestUcf32", tmp)
    cfg["mri"].update(online_smaps=None, online_smaps_kws=None, kspace_type="simulated")
    cfg["data"]["val"].update(end_slice=2, start_slice=0)
    cfg["data"]["train"].update(end_slice=2, start_slice=0)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.save({"model_state_dict": build_model(cfg).state_dict(), "step": 50},
               os.path.join(root, "net.ckpt"))

    s = coils(h, w, c=4, dtype=torch.complex64)
    img = structure((1, 1, h, w)).to(torch.complex64) * 0.2
    organ = (s.abs().sum(1, keepdim=True) > 0).float() * (img.abs() > 0).float()
    kspace = FFT2D()(Sense(s)(img))
    batch = (kspace, s, img, organ, torch.tensor([[64, 48]]))

    real = dict(load=cd.load_batch, pick=cd.pick_slices)
    cd.load_batch = lambda c, v, sl: batch
    cd.pick_slices = lambda c, a: [("volA", 0, "random"), ("volB", 1, "flagged")]
    try:
        out = os.path.join(tmp, "diag")
        cd.main(["--run", root, "--out", out, "--combine", "--device", "cpu", "--sigma", "0.015",
                 "--val-kspace-type", "simulated"])
    finally:
        cd.load_batch, cd.pick_slices = real["load"], real["pick"]
    with open(os.path.join(out, "summary.txt")) as f:
        text = f.read()
    print("\n" + text)
    check("summary: every knockout, both slice sources, per-V spectra, ghost by stage",
          all(k in text for k in cd.KNOCKOUTS) and "flagged" in text and "L2->L1" in text
          and "flagged, beta (whole organ)" in text)
    import csv
    with open(os.path.join(out, "knockouts.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    check("knockouts.csv: 2 slices x 8 knockouts", len(rows) == 16, str(len(rows)))
    check("no temp files left", not [n for n in os.listdir(out) if "partial" in n])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (test_folded, test_alpha_z_zero, test_spectrum, test_hooks, test_rho,
                   test_ghost_fraction, test_trajectory):
            print(f"\n--- {fn.__name__}")
            fn()
        for fn in (test_cells, test_whole_pass):
            print(f"\n--- {fn.__name__}")
            fn(tmp)
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
