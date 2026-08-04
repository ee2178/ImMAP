"""
Correctness + speed check for the Triton phase-invariant attention kernel.

Two independent layers, because they fail for different reasons:

  PART A -- the ALGEBRA, on CPU, no triton needed.
      `_ref_forward` / `_ref_backward` re-express exactly what the kernels
      compute (the offset loop, the streaming softmax, the hand-derived
      dq/dk/dv/dbias expressions) in plain PyTorch, and check them against
      autograd. If the derivation is wrong this catches it anywhere, including
      a laptop. It is also the readable statement of what the kernel does.

  PART B -- the KERNEL, on CUDA, triton needed.
      The compiled kernel against the GATHER path (`circulant_similarity.py`
      builds the dense window with torch.roll, `circulant_attention.Circulant`
      applies it). Slow and memory-hungry -- which is why the kernel exists --
      but exact, and affordable on small grids. A failure here that Part A
      passed is a Triton problem: indexing, masking, or the API.

Then a benchmark, which is the number that decides whether any of it was worth
it.

    python -m tests.test_triton_attention            # both parts + bench
    python -m tests.test_triton_attention --bench-only
"""

import argparse
import itertools
import sys

import torch

# Part B's imports are deferred: `models/__init__` pulls in FlexAttention, which
# needs torch 2.x, and Part A must stay runnable on any box.

EPS = 1e-8
PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def rel(a, b, eps=1e-12):
    return ((a - b).abs().max() / (b.abs().max() + eps)).item()


def close(name, a, b, tol=1e-5):
    err = rel(a, b)
    check(name, err < tol, f"rel={err:.2e}")


# ===========================================================================
#  PART A -- the algebra, in plain PyTorch (CPU, no triton)
# ===========================================================================
def _offsets(win):
    p = (win - 1) // 2
    return list(itertools.product(range(-p, p + 1), repeat=2))


def _shift(x, o):
    """`xr[..., r] == x[..., r + o]`, circular -- the kernel's wraparound."""
    return torch.roll(x, shifts=(-o[0], -o[1]), dims=(2, 3))


def _ref_forward(qr, qi, kr, ki, v, win, use_knorm=True, bias=None):
    """What `_fwd` computes: offset loop + streaming (online) softmax."""
    B, _, H, W = qr.shape
    knorm = (kr * kr).sum(1, keepdim=True)
    if ki is not None:
        knorm = knorm + (ki * ki).sum(1, keepdim=True)

    m = torch.full((B, 1, H, W), -1e30, device=qr.device)
    l = torch.zeros((B, 1, H, W), device=qr.device)
    acc = torch.zeros_like(v)

    for o in _offsets(win):
        kr_o = _shift(kr, o)
        re = (qr * kr_o).sum(1, keepdim=True)
        if ki is not None:
            ki_o = _shift(ki, o)
            re = re + (qi * ki_o).sum(1, keepdim=True)
            im = (qi * kr_o).sum(1, keepdim=True) - (qr * ki_o).sum(1, keepdim=True)
            s = torch.sqrt(re * re + im * im + EPS)
        else:
            s = torch.sqrt(re * re + EPS)
        if use_knorm:
            s = s - 0.5 * _shift(knorm, o)
        if bias is not None:
            s = s + _shift(bias, o)

        m_new = torch.maximum(m, s)
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = l * alpha + p
        acc = acc * alpha + p * _shift(v, o)
        m = m_new

    return acc / l, m + torch.log(l)


def _ref_backward(qr, qi, kr, ki, v, win, dout, lse, out, dlse=None,
                  use_knorm=True, bias=None):
    """What `_bwd_dq` / `_bwd_dkdv` compute, expression for expression.

    Per-key quantities are shifted BACK by -o, which is the PyTorch spelling of
    "key k is seen by query j = k - o" -- the symmetry that lets the kernel do
    the dk/dv pass without atomics.
    """
    B, _, H, W = qr.shape
    knorm = (kr * kr).sum(1, keepdim=True)
    if ki is not None:
        knorm = knorm + (ki * ki).sum(1, keepdim=True)

    delta = (dout * out).sum(1, keepdim=True)
    if dlse is not None:
        delta = delta - dlse

    dqr = torch.zeros_like(qr)
    dqi = torch.zeros_like(qr) if qi is not None else None
    dkr = torch.zeros_like(kr)
    dki = torch.zeros_like(kr) if ki is not None else None
    dv = torch.zeros_like(v)
    dperkey = torch.zeros((B, 1, H, W), device=qr.device)

    for o in _offsets(win):
        back = (-o[0], -o[1])
        kr_o, v_o = _shift(kr, o), _shift(v, o)
        re = (qr * kr_o).sum(1, keepdim=True)
        if ki is not None:
            ki_o = _shift(ki, o)
            re = re + (qi * ki_o).sum(1, keepdim=True)
            im = (qi * kr_o).sum(1, keepdim=True) - (qr * ki_o).sum(1, keepdim=True)
        else:
            ki_o = kr_o
            im = torch.zeros_like(re)
        A = torch.sqrt(re * re + im * im + EPS)

        s = A
        if use_knorm:
            s = s - 0.5 * _shift(knorm, o)
        if bias is not None:
            s = s + _shift(bias, o)

        gamma = torch.exp(s - lse)
        ds = gamma * ((dout * v_o).sum(1, keepdim=True) - delta)
        c_re, c_im = ds * re / A, ds * im / A

        dqr += c_re * kr_o - c_im * ki_o
        if qi is not None:
            dqi += c_re * ki_o + c_im * kr_o

        dkr += _shift(c_re * qr + (c_im * qi if qi is not None else 0.0), back)
        if ki is not None:
            dki += _shift(c_re * qi - c_im * qr, back)
        dv += _shift(gamma * dout, back)
        dperkey += _shift(ds, back)

    if use_knorm:                       # knorm_k enters as -1/2 knorm_k
        dkr -= dperkey * kr
        if ki is not None:
            dki -= dperkey * ki

    return dqr, dqi, dkr, dki, dv, dperkey


def _autograd_forward(qr, qi, kr, ki, v, win, use_knorm=True, bias=None):
    """The same objective written so autograd can differentiate it."""
    knorm = (kr * kr).sum(1, keepdim=True)
    if ki is not None:
        knorm = knorm + (ki * ki).sum(1, keepdim=True)

    scores, vals = [], []
    for o in _offsets(win):
        kr_o = _shift(kr, o)
        re = (qr * kr_o).sum(1, keepdim=True)
        if ki is not None:
            ki_o = _shift(ki, o)
            re = re + (qi * ki_o).sum(1, keepdim=True)
            im = (qi * kr_o).sum(1, keepdim=True) - (qr * ki_o).sum(1, keepdim=True)
            s = torch.sqrt(re * re + im * im + EPS)
        else:
            s = torch.sqrt(re * re + EPS)
        if use_knorm:
            s = s - 0.5 * _shift(knorm, o)
        if bias is not None:
            s = s + _shift(bias, o)
        scores.append(s)
        vals.append(_shift(v, o))

    S = torch.cat(scores, dim=1)                          # (B, K, H, W)
    G = torch.softmax(S, dim=1)
    V = torch.stack(vals, dim=1)                          # (B, K, C, H, W)
    return (G.unsqueeze(2) * V).sum(1), torch.logsumexp(S, dim=1, keepdim=True)


def test_kernel_algebra():
    """The kernel's forward and hand-derived backward, versus autograd."""
    torch.manual_seed(0)
    B, C, DV, H, W, WIN = 2, 5, 4, 9, 7, 5

    for complex_q, use_knorm, use_bias, use_dlse in [
            (True, True, False, False),
            (True, True, True, False),      # per-key bias (the transpose path)
            (True, True, False, True),      # d/d lse (also the transpose path)
            (True, False, False, False),    # pidot
            (False, True, False, False)]:   # real features
        tag = (f"cplx={int(complex_q)} knorm={int(use_knorm)} "
               f"bias={int(use_bias)} dlse={int(use_dlse)}")

        qr = torch.randn(B, C, H, W, requires_grad=True)
        kr = torch.randn(B, C, H, W, requires_grad=True)
        qi = torch.randn(B, C, H, W, requires_grad=True) if complex_q else None
        ki = torch.randn(B, C, H, W, requires_grad=True) if complex_q else None
        v = torch.randn(B, DV, H, W, requires_grad=True)
        bias = torch.randn(B, 1, H, W, requires_grad=True) if use_bias else None

        with torch.no_grad():
            out_k, lse_k = _ref_forward(qr, qi, kr, ki, v, WIN, use_knorm, bias)
        out_a, lse_a = _autograd_forward(qr, qi, kr, ki, v, WIN, use_knorm, bias)
        close(f"[{tag}] streaming softmax == plain softmax", out_k, out_a.detach())
        close(f"[{tag}] lse", lse_k, lse_a.detach())

        dout = torch.randn_like(out_a)
        dlse = torch.randn_like(lse_a) if use_dlse else None
        leaves = [t for t in (qr, qi, kr, ki, v, bias) if t is not None]
        grads = torch.autograd.grad(
            (out_a * dout).sum() + ((lse_a * dlse).sum() if use_dlse else 0.0),
            leaves)

        with torch.no_grad():
            mine = _ref_backward(qr, qi, kr, ki, v, WIN, dout, lse_a.detach(),
                                 out_a.detach(), dlse, use_knorm, bias)
        dqr, dqi, dkr, dki, dv, dbias = mine
        got = [dqr] + ([dqi] if complex_q else []) + [dkr] + \
              ([dki] if complex_q else []) + [dv] + ([dbias] if use_bias else [])
        names = ["dqr"] + (["dqi"] if complex_q else []) + ["dkr"] + \
                (["dki"] if complex_q else []) + ["dv"] + \
                (["dbias"] if use_bias else [])
        for nm, a, b in zip(names, got, grads):
            close(f"[{tag}] backward {nm}", a, b, tol=2e-4)


def _dense_gamma(qr, qi, kr, ki, win, use_knorm):
    """The explicit (B, N, N) adjacency -- ground truth for the transpose."""
    B, _, H, W = qr.shape
    N = H * W
    knorm = ((kr * kr).sum(1) + ((ki * ki).sum(1) if ki is not None else 0)
             ).reshape(B, N)
    S = torch.full((B, N, N), -1e30)
    p = (win - 1) // 2
    for jr, jc in itertools.product(range(H), range(W)):
        for dr, dc in itertools.product(range(-p, p + 1), repeat=2):
            r2, c2 = (jr + dr) % H, (jc + dc) % W
            qv, kv = qr[:, :, jr, jc], kr[:, :, r2, c2]
            re = (qv * kv).sum(1)
            if ki is not None:
                qvi, kvi = qi[:, :, jr, jc], ki[:, :, r2, c2]
                re = re + (qvi * kvi).sum(1)
                im = (qvi * kv).sum(1) - (qv * kvi).sum(1)
                s = torch.sqrt(re * re + im * im + EPS)
            else:
                s = torch.sqrt(re * re + EPS)
            if use_knorm:
                s = s - 0.5 * knorm[:, r2 * W + c2]
            S[:, jr * W + jc, r2 * W + c2] = s
    return torch.softmax(S, dim=2)


def test_transpose_recipe():
    """`TritonAdjacency`'s transpose bookkeeping, against a dense Gamma.

    The kernel is stubbed out by `_ref_forward` here; what is under test is

        (Gamma^T h)_k = colsum_k * fwd(q<->k, bias=-lse)_k,
        colsum_k      = exp(lse_t - 1/2||k||^2) = sum_j Gamma_jk

    A sign error or a dropped term would surface only in the transposed apply,
    i.e. only in the rigorous subgradient -- late, and hard to attribute.
    """
    torch.manual_seed(1)
    B, C, DV, H, W = 2, 4, 3, 5, 6

    for use_knorm, complex_q in [(True, True), (False, True), (True, False)]:
        tag = f"knorm={int(use_knorm)} cplx={int(complex_q)}"
        qr, kr = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
        qi = torch.randn(B, C, H, W) if complex_q else None
        ki = torch.randn(B, C, H, W) if complex_q else None
        v = torch.randn(B, DV, H, W)
        vflat = v.reshape(B, DV, H * W)

        for win in (3, 5):
            G = _dense_gamma(qr, qi, kr, ki, win, use_knorm)
            ksq = (kr * kr).sum(1, keepdim=True)
            if ki is not None:
                ksq = ksq + (ki * ki).sum(1, keepdim=True)

            got_f, lse = _ref_forward(qr, qi, kr, ki, v, win, use_knorm, None)
            want_f = torch.einsum("bjk,bck->bcj", G, vflat).reshape(B, DV, H, W)
            close(f"[{tag} win={win}] forward == dense Gamma v", got_f, want_f)

            out, lse_t = _ref_forward(kr, ki, qr, qi, v, win, use_knorm=False,
                                      bias=-lse)
            colsum = torch.exp(lse_t + (-0.5 * ksq if use_knorm else 0.0))
            close(f"[{tag} win={win}] colsum == column sums of Gamma",
                  colsum.reshape(B, -1), G.sum(dim=1), tol=1e-4)

            want_t = torch.einsum("bjk,bcj->bck", G, vflat).reshape(B, DV, H, W)
            close(f"[{tag} win={win}] transpose recipe == dense Gamma^T v",
                  out * colsum, want_t, tol=1e-4)


def _load_circulant_triton():
    """Import `models.circulant_triton` without running `models/__init__`.

    That `__init__` pulls in FlexAttention, which needs torch 2.x. This module
    needs neither it nor triton to be *imported*, and Part A is supposed to run
    on any box -- including the one where these layout helpers get edited.
    """
    try:
        import models.circulant_triton as m
        return m
    except ImportError:
        import importlib.util
        import os
        import types
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if "models" not in sys.modules:
            pkg = types.ModuleType("models")
            pkg.__path__ = [os.path.join(root, "models")]
            sys.modules["models"] = pkg
        spec = importlib.util.spec_from_file_location(
            "models.circulant_triton",
            os.path.join(root, "models", "circulant_triton.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["models.circulant_triton"] = m
        spec.loader.exec_module(m)
        return m


def test_layout_roundtrip():
    """img -> seq -> img must be exact, and match GroupThreshold._to_heads.

    This is where the first GPU run died: a permute without the matching
    flatten left a 4-D tensor, so `B, HW, C = qr.shape` unpacked four values.
    """
    ct = _load_circulant_triton()
    _img_to_seq, _seq_to_img = ct._img_to_seq, ct._seq_to_img
    B, C, H, W = 3, 12, 5, 7
    x = torch.randn(B, C, H, W)
    for heads in (1, 2, 3, 4):
        s = _img_to_seq(x, heads)
        check(f"seq shape heads={heads}",
              tuple(s.shape) == (B * heads, H * W, C // heads),
              str(tuple(s.shape)))
        check(f"roundtrip heads={heads}",
              torch.equal(_seq_to_img(s, B, heads, H, W), x))
        check(f"head split matches _to_heads heads={heads}",
              torch.equal(s.transpose(1, 2).reshape(B * heads, C // heads, H, W),
                          x.reshape(B * heads, C // heads, H, W)))


# ---------------------------------------------------------------------------
#  reference: the gather path, in the kernel's layout
# ---------------------------------------------------------------------------
def ref_apply(q_img, k_img, v_img, win, sim, transpose=False):
    """`Gamma x` via the materialised Circulant. q/k may be complex."""
    from models.circulant_attention import circ_adjacency
    G = circ_adjacency(sim, q_img, k_img, win)
    return G.apply(v_img, transpose=transpose)


def ref_lse(q_img, k_img, win, sim):
    """log-sum-exp of the window scores, per query -- the kernel's `lse`.

    The kernel drops `-1/2||q||^2` (constant across k, cancels in the softmax),
    so the reference has to be shifted by the same amount to compare.
    """
    from models.circulant_similarity import circulant_similarity_window
    vals, _, _ = circulant_similarity_window(sim, q_img, k_img, win)   # (B,N,K)
    # the kernel drops the -1/2||q||^2 term (constant in k, cancels in softmax)
    if sim == "pidistance":
        qsq = q_img.abs().pow(2).sum(1).reshape(vals.shape[0], -1)
        vals = vals + 0.5 * qsq[:, :, None]
    return torch.logsumexp(vals, dim=2)


def make_inputs(B=2, C=8, DV=8, H=12, W=12, complex_q=True, seed=0,
                device="cuda", requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape):
        t = torch.randn(*shape, generator=g).to(device=device, dtype=torch.float32)
        t.requires_grad_(requires_grad)
        return t

    if complex_q:
        qr, qi, kr, ki = r(B, C, H, W), r(B, C, H, W), r(B, C, H, W), r(B, C, H, W)
        q_img = torch.complex(qr, qi)
        k_img = torch.complex(kr, ki)
    else:
        qr, kr = r(B, C, H, W), r(B, C, H, W)
        qi = ki = None
        q_img, k_img = qr, kr
    v_img = r(B, DV, H, W)
    return dict(qr=qr, qi=qi, kr=kr, ki=ki, v=v_img, q_img=q_img, k_img=k_img)


# ---------------------------------------------------------------------------
def test_forward(device):
    from models.circulant_triton import TritonAdjacency
    for sim in ("pidistance", "pidot"):
        for complex_q in (True, False):
            d = make_inputs(complex_q=complex_q, device=device, seed=1)
            H = W = 12
            tag = f"{sim}/{'complex' if complex_q else 'real'}"

            adj = TritonAdjacency(d["q_img"], d["k_img"], win=5, sim=sim,
                                  heads=1, block_m=32)
            got = adj.apply(d["v"])
            want = ref_apply(d["q_img"], d["k_img"], d["v"], 5, sim)
            check(f"forward {tag}", rel(got, want) < 2e-5,
                  f"rel={rel(got, want):.2e}")

            want_lse = ref_lse(d["q_img"], d["k_img"], 5, sim)
            got_lse = adj._lse.reshape(want_lse.shape)
            check(f"forward lse {tag}", rel(got_lse, want_lse) < 2e-5,
                  f"rel={rel(got_lse, want_lse):.2e}")


def test_transpose(device):
    from models.circulant_triton import TritonAdjacency
    for sim in ("pidistance", "pidot"):
        d = make_inputs(device=device, seed=2)
        adj = TritonAdjacency(d["q_img"], d["k_img"], win=5, sim=sim,
                              heads=1, block_m=32)
        adj.apply(d["v"])                                  # populate lse
        got = adj.apply(d["v"], transpose=True)
        want = ref_apply(d["q_img"], d["k_img"], d["v"], 5, sim, transpose=True)
        check(f"transposed apply {sim}", rel(got, want) < 5e-5,
              f"rel={rel(got, want):.2e}")

        # cold transpose: no forward first, so the class has to build lse itself
        adj2 = TritonAdjacency(d["q_img"], d["k_img"], win=5, sim=sim,
                               heads=1, block_m=32)
        got2 = adj2.apply(d["v"], transpose=True)
        check(f"transposed apply {sim} (cold)", rel(got2, want) < 5e-5,
              f"rel={rel(got2, want):.2e}")


def test_heads(device):
    from models.circulant_triton import TritonAdjacency
    d = make_inputs(B=2, C=8, DV=8, device=device, seed=3)
    for heads in (1, 2, 4):
        adj = TritonAdjacency(d["q_img"], d["k_img"], win=5, heads=heads,
                              block_m=32)
        got = adj.apply(d["v"])
        # reference folds heads into the batch axis the same way
        B, C, H, W = d["q_img"].shape
        qh = d["q_img"].reshape(B * heads, C // heads, H, W)
        kh = d["k_img"].reshape(B * heads, C // heads, H, W)
        vh = d["v"].reshape(B * heads, -1, H, W)
        want = ref_apply(qh, kh, vh, 5, "pidistance").reshape(B, -1, H, W)
        check(f"heads={heads}", rel(got, want) < 2e-5, f"rel={rel(got, want):.2e}")


def test_gradients(device):
    from models.circulant_triton import TritonAdjacency
    """dq / dk / dv against autograd through the gather path."""
    for sim in ("pidistance", "pidot"):
        for complex_q in (True, False):
            tag = f"{sim}/{'complex' if complex_q else 'real'}"
            d = make_inputs(H=10, W=10, C=8, DV=8, device=device, seed=4,
                            complex_q=complex_q, requires_grad=True)
            leaves = [t for t in (d["qr"], d["qi"], d["kr"], d["ki"], d["v"])
                      if t is not None]

            # --- kernel ---
            # retain_graph: both paths hang off the SAME torch.complex(qr, qi)
            # node built in make_inputs, so freeing after the first grad call
            # takes the reference's graph with it.
            adj = TritonAdjacency(d["q_img"], d["k_img"], win=5, sim=sim,
                                  heads=1, block_m=32)
            out = adj.apply(d["v"])
            loss = (out * out).sum()
            g_tri = torch.autograd.grad(loss, leaves, retain_graph=True)

            # --- reference ---
            out_r = ref_apply(d["q_img"], d["k_img"], d["v"], 5, sim)
            loss_r = (out_r * out_r).sum()
            g_ref = torch.autograd.grad(loss_r, leaves)

            names = ["dqr"] + (["dqi"] if complex_q else []) + ["dkr"] + \
                    (["dki"] if complex_q else []) + ["dv"]
            for nm, a, b in zip(names, g_tri, g_ref):
                check(f"grad {nm} {tag}", rel(a, b) < 2e-4, f"rel={rel(a, b):.2e}")


def test_transpose_gradients(device):
    """The transposed apply must be differentiable THROUGH the cached lse.

    `Gamma^T` is built from the forward's log-sum-exp, so if the kernel's `lse`
    output were not differentiable the q/k gradients would be silently wrong --
    and only here, not in the forward.
    """
    from models.circulant_triton import TritonAdjacency
    d = make_inputs(H=10, W=10, C=8, DV=8, device=device, seed=5,
                    requires_grad=True)
    leaves = [d["qr"], d["qi"], d["kr"], d["ki"], d["v"]]

    adj = TritonAdjacency(d["q_img"], d["k_img"], win=5, heads=1, block_m=32)
    out = adj.apply(d["v"], transpose=True)
    # retain_graph: shared torch.complex node, as in test_gradients
    g_tri = torch.autograd.grad((out * out).sum(), leaves, retain_graph=True)

    out_r = ref_apply(d["q_img"], d["k_img"], d["v"], 5, "pidistance",
                      transpose=True)
    g_ref = torch.autograd.grad((out_r * out_r).sum(), leaves)

    for nm, a, b in zip(["dqr", "dqi", "dkr", "dki", "dv"], g_tri, g_ref):
        check(f"transpose grad {nm}", rel(a, b) < 5e-4, f"rel={rel(a, b):.2e}")


def test_group_threshold_end_to_end(device):
    """The whole prox, forward and every subgradient mode, triton vs gather."""
    from models.prox import GroupThreshold
    torch.manual_seed(6)
    kw = dict(M=8, Mh=4, window=5, tau0=0.05, nheads=1, sim_fun="pidistance")
    gat = GroupThreshold(attn_backend="gather", **kw).to(device)
    tri = GroupThreshold(attn_backend="triton", **kw).to(device)
    tri.load_state_dict(gat.state_dict())

    z = torch.randn(2, 8, 12, 12, dtype=torch.complex64, device=device)

    with torch.no_grad():
        a, _ = gat(z, 0.02, {})
        b, _ = tri(z, 0.02, {})
    check("GroupThreshold forward", rel(b, a) < 5e-5, f"rel={rel(b, a):.2e}")

    for mode in ("moreau", "simple", "rigorous"):
        with torch.no_grad():
            ga, _ = gat.subgradient(z, 0.02, {}, mode=mode)
            gb, _ = tri.subgradient(z, 0.02, {}, mode=mode)
        check(f"GroupThreshold dg[{mode}]", rel(gb, ga) < 5e-4,
              f"rel={rel(gb, ga):.2e}")

    # and it trains: one step must move the weights without NaNs
    zt = z.clone().requires_grad_(True)
    out, _ = tri(zt, 0.02, {})
    out.abs().pow(2).sum().backward()
    grads = [p.grad for p in tri.parameters() if p.grad is not None]
    check("GroupThreshold backward is finite",
          bool(grads) and all(torch.isfinite(g).all() for g in grads),
          f"{len(grads)} parameter grads")


def test_numerical_floor(device):
    """How much of the gather-vs-triton gap is just fp32 reduction order?

    `block_m` changes only the tiling and the order of the softmax reduction --
    never the result in exact arithmetic. So running the SAME kernel at two
    block sizes measures the floor. A gather-vs-triton difference of the same
    magnitude carries no information about correctness; one far above it does.

    This exists because the prox modes that clamp (`relu(1 - tau/xi)`) or
    divide by `xi` amplify a ~1e-7 difference in Gamma, and it is worth knowing
    that rather than assuming it.
    """
    from models.prox import GroupThreshold
    torch.manual_seed(9)
    kw = dict(M=8, Mh=4, window=5, tau0=0.05, nheads=1, sim_fun="pidistance")
    gat = GroupThreshold(attn_backend="gather", **kw).to(device)
    t32 = GroupThreshold(attn_backend="triton", triton_block_m=32, **kw).to(device)
    t64 = GroupThreshold(attn_backend="triton", triton_block_m=64, **kw).to(device)
    t32.load_state_dict(gat.state_dict())
    t64.load_state_dict(gat.state_dict())

    z = torch.randn(2, 8, 12, 12, dtype=torch.complex64, device=device)

    def three(fn):
        with torch.no_grad():
            return fn(gat), fn(t32), fn(t64)

    cases = {"forward": lambda m: m(z, 0.02, {})[0]}
    for mode in ("moreau", "simple", "rigorous"):
        cases[f"dg[{mode}]"] = (
            lambda m, _mode=mode: m.subgradient(z, 0.02, {}, mode=_mode)[0])

    for name, fn in cases.items():
        a, b32, b64 = three(fn)
        floor = rel(b32, b64)
        gap = rel(b32, a)
        print(f"       {name:<14} tiling floor {floor:.2e}   "
              f"gather gap {gap:.2e}   ratio {gap / max(floor, 1e-12):6.1f}x")
        # Loose: catches a broken kernel (gap would be O(1)) without asserting
        # a relationship between two noisy measurements.
        check(f"{name} gap is numerical, not structural",
              gap < max(50 * floor, 1e-3), f"gap={gap:.2e} floor={floor:.2e}")


def test_wraparound(device):
    from models.circulant_triton import TritonAdjacency
    """The window is CIRCULAR: an edge pixel must see the opposite edge."""
    d = make_inputs(B=1, C=8, DV=8, H=8, W=8, device=device, seed=7)
    for win in (3, 5, 7):
        adj = TritonAdjacency(d["q_img"], d["k_img"], win=win, heads=1, block_m=32)
        got = adj.apply(d["v"])
        want = ref_apply(d["q_img"], d["k_img"], d["v"], win, "pidistance")
        check(f"circular wrap win={win}", rel(got, want) < 2e-5,
              f"rel={rel(got, want):.2e}")


def test_shapes(device):
    from models.circulant_triton import TritonAdjacency
    """Non-square grids, sizes that do not divide BLOCK_M, odd channel counts."""
    for (H, W, C, DV, bm) in [(7, 11, 6, 6, 32), (16, 5, 12, 4, 64),
                              (13, 13, 3, 9, 16)]:
        d = make_inputs(B=1, C=C, DV=DV, H=H, W=W, device=device, seed=8)
        adj = TritonAdjacency(d["q_img"], d["k_img"], win=3, heads=1, block_m=bm)
        got = adj.apply(d["v"])
        want = ref_apply(d["q_img"], d["k_img"], d["v"], 3, "pidistance")
        check(f"shape {H}x{W} C={C} DV={DV} block_m={bm}",
              rel(got, want) < 2e-5, f"rel={rel(got, want):.2e}")


# ---------------------------------------------------------------------------
def benchmark(device, H=160, W=160, Mh=64, win=9, B=1):
    """The number that decides whether any of this was worth it."""
    from models.circulant_attention import circ_adjacency
    from models.circulant_triton import TritonAdjacency
    print(f"\n--- benchmark: {B}x{Mh}x{H}x{W}, window {win} ---")
    q = torch.randn(B, Mh, H, W, dtype=torch.complex64, device=device)
    k = torch.randn(B, Mh, H, W, dtype=torch.complex64, device=device)
    v = torch.randn(B, Mh, H, W, device=device)

    def bench(fn, label, warmup=3, iters=10):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            print(f"  {label:<28} {ms:8.2f} ms   peak {peak:8.0f} MiB")
            return ms
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:<28} {'OOM':>8}")
            torch.cuda.empty_cache()
            return float("inf")

    def triton_fwd():
        TritonAdjacency(q, k, win, heads=1).apply(v)

    def gather_fwd():
        circ_adjacency("pidistance", q, k, win).apply(v)

    t_tri = bench(triton_fwd, "triton  forward")
    t_gat = bench(gather_fwd, "gather  forward")
    if t_gat < float("inf") and t_tri > 0:
        print(f"  speedup: {t_gat / t_tri:.1f}x")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--algebra-only", action="store_true",
                    help="Part A only -- runs anywhere, no CUDA or triton")
    args = ap.parse_args()

    print(f"torch {torch.__version__}")

    # ---- PART A: the algebra. Runs anywhere. --------------------------------
    if not args.bench_only:
        print("\n=== PART A: kernel algebra + bookkeeping (CPU) ===")
        for fn in (test_kernel_algebra, test_transpose_recipe,
                   test_layout_roundtrip):
            print(f"--- {fn.__name__} ---")
            try:
                fn()
            except Exception as exc:                              # noqa: BLE001
                check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()

    if args.algebra_only:
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
        if FAIL:
            print("failed: " + ", ".join(FAIL))
        sys.exit(1 if FAIL else 0)

    # ---- PART B: the compiled kernel. Needs CUDA + triton. ------------------
    from models.circulant_triton import HAVE_TRITON
    if not torch.cuda.is_available() or not HAVE_TRITON:
        why = ("CUDA is unavailable" if not torch.cuda.is_available()
               else "triton is not importable (it ships with PyTorch's Linux "
                    "CUDA wheels)")
        print(f"\n=== PART B skipped: {why} ===")
        print(f"\n{len(PASS)} passed, {len(FAIL)} failed (algebra only)")
        sys.exit(1 if FAIL else 0)

    device = "cuda"
    print(f"\n=== PART B: compiled kernel on {torch.cuda.get_device_name(0)} ===")

    # Smoke the kernel once before the suite. A compile error is the same error
    # in all eight tests, and eight identical stack traces help nobody.
    try:
        from models.circulant_triton import TritonAdjacency
        _q = torch.randn(1, 4, 8, 8, dtype=torch.complex64, device=device)
        TritonAdjacency(_q, _q, win=3, heads=1, block_m=16).apply(
            torch.randn(1, 4, 8, 8, device=device))
        print("smoke: kernel compiles and runs")
    except Exception as exc:                                      # noqa: BLE001
        print(f"\nsmoke FAILED -- the kernel does not compile/run, so the "
              f"suite below would report the same error eight times.\n")
        import traceback
        traceback.print_exc()
        print(f"\n{len(PASS)} passed (Part A), Part B blocked: "
              f"{type(exc).__name__}")
        sys.exit(1)

    if not args.bench_only:
        for fn in (test_forward, test_transpose, test_heads, test_wraparound,
                   test_shapes, test_gradients, test_transpose_gradients,
                   test_group_threshold_end_to_end, test_numerical_floor):
            print(f"\n--- {fn.__name__} ---")
            try:
                fn(device)
            except Exception as exc:                              # noqa: BLE001
                check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()

        print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
        if FAIL:
            print("failed: " + ", ".join(FAIL))

    if not args.no_bench:
        try:
            benchmark(device)
        except Exception as exc:                                  # noqa: BLE001
            print(f"benchmark failed: {type(exc).__name__}: {exc}")

    sys.exit(1 if FAIL else 0)
