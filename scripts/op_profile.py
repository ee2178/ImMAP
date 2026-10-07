# -*- coding: utf-8 -*-
"""
Which individual operation is the elementwise remainder made of?

`scripts/profile_mg.py --breakdown` times five hand-picked families (fft, conv,
convT, pad, fftshift) by wrapping the module-level functions, and calls whatever
is left "elementwise / overhead".  That remainder is ~60% of the runtime and it
is NOT attributed: it lumps together the arithmetic of the LPDS sweep, the
complex<->planar plumbing around every conv, the Sense Gram's coil multiplies,
the Python dispatch, and -- because the wrapped run is slower than the clean one
-- the instrumentation's own cost.  Reasoning about it from tensor shapes and an
assumed memory bandwidth got the answer wrong by 1.5x once already (an L40S
holds a 33 MB half-latent in its 48 MB L2, so the passes being counted at DRAM
speed were mostly cache hits).

So measure it instead.  Two passes, deliberately separate:

TIME   `op_times` -- torch.profiler (Kineto/CUPTI).  Per-kernel device time
       attributed to the aten op that launched it, keyed by op name AND input
       shapes, so `aten::mul` on the (1,169,320,160) latent is a different row
       from `aten::mul` on the (1,1,640,320) image.  Nothing is hand-picked and
       nothing escapes: every op the model runs appears.  Overhead is per-kernel
       bookkeeping in C++, not a Python wrapper, so the totals stay close to the
       clean wall time.

BYTES  `op_bytes` -- a `TorchDispatchMode` that sees the real tensors, so dtypes
       and OUTPUT shapes are exact rather than inferred.  No events, no
       synchronisation, no timing: it cannot distort what it measures.

Dividing one by the other gives the number that actually decides what is worth
optimising -- ACHIEVED BANDWIDTH per op:

    > 1000 GB/s   L2-resident.  Removing the pass saves little; the data was
                  never going to DRAM.
    400-900 GB/s  DRAM-bound on an L40S (864 GB/s peak).  Bytes removed here
                  convert to time saved, roughly one for one.
    < 100 GB/s    latency- or launch-bound.  Fusing KERNEL COUNT helps; moving
                  fewer bytes does not.

Those cut-offs are the L40S's; the report scales them to the card it ran on
(`dram_peak`). It also prints how long the GPU was BUSY against the wall time:
the difference is the host issuing kernels, which no per-op row can show.

A long tail of sub-100 GB/s rows means there is no single expensive elementwise
op to find, and the lever is kernel count (CUDA graphs, torch.compile) or the
architecture (M / s^2), not micro-optimisation.
"""

from __future__ import annotations

from collections import defaultdict

import torch

# ---------------------------------------------------------------------------
#  op -> family
# ---------------------------------------------------------------------------
# "plumbing" is the category that matters: cat / copy_ / complex / contiguous
# move the latent around WITHOUT doing arithmetic on it. If plumbing outweighs
# "math", the cost is the data layout, not the computation.
_FAMILY = [
    ("conv", ("convolution", "conv1d", "conv2d", "conv3d", "conv_transpose",
              "cudnn_convolution", "mkldnn_convolution", "slow_conv",
              "thnn_conv", "slow_conv_transpose", "slow_conv2d")),
    ("fft", ("_fft_", "fft_")),
    ("pad", ("constant_pad_nd", "pad", "reflection_pad", "replication_pad",
             "circular_pad", "pad_circular")),
    ("plumbing", ("cat", "copy_", "clone", "contiguous", "_to_copy", "to",
                  "complex", "stack", "zeros", "zeros_like", "roll", "set_",
                  "conj_physical", "_conj_copy", "resize_")),
    ("reduction", ("sum", "mean", "prod", "linalg_vector_norm", "norm", "max",
                   "min", "amax", "amin", "cumsum")),
    ("math", ("mul", "add", "sub", "div", "rsub", "neg", "abs", "sqrt", "rsqrt",
              "pow", "exp", "log", "clamp", "where", "sgn", "sign", "relu",
              "minimum", "maximum", "reciprocal", "addcmul", "lerp", "hypot",
              "masked_fill", "threshold", "silu", "sigmoid", "tanh", "softmax",
              "angle", "polar", "real", "imag")),
    ("matmul", ("mm", "bmm", "matmul", "addmm", "baddbmm", "einsum", "linear")),
    ("index", ("index", "select", "slice", "narrow", "gather", "scatter",
               "take", "repeat", "expand", "permute", "transpose", "reshape",
               "view", "unsqueeze", "squeeze", "flip", "unfold")),
]

# A view costs nothing: no kernel, no traffic. Kept out of the byte accounting
# so it cannot inflate a "plumbing" total with free operations.
_FREE = {"view", "view_as", "as_strided", "detach", "expand", "permute",
         "transpose", "reshape", "squeeze", "unsqueeze", "slice", "select",
         "narrow", "t", "unfold", "alias", "_conj", "conj", "view_as_real",
         "view_as_complex", "real", "imag", "resolve_conj", "flatten",
         "empty", "empty_like", "empty_strided", "lift_fresh"}


# DRAM peak bandwidth by card, GB/s, from the spec sheets. The bandwidth
# verdicts are fractions of it, so a row is judged against the card it ran on
# (they were hard-wired to the L40S, which mislabels an A100 or H200 run).
# First match wins, so the more specific name goes first.
_DRAM_PEAK = (("H200", 4800), ("H100", 3350), ("A100-SXM4-80GB", 2039),
              ("A100", 1555), ("L40S", 864), ("L40", 864))
_DEFAULT_PEAK = ("L40S (assumed)", 864)


def dram_peak(device):
    """`(card, GB/s)` for the device the profile ran on."""
    if device.type != "cuda":
        return _DEFAULT_PEAK
    name = torch.cuda.get_device_name(device)
    for key, gbs in _DRAM_PEAK:
        if key in name:
            return key, gbs
    return _DEFAULT_PEAK


def short(op):
    """One canonical name from either pass.

    torch.profiler keys look like `aten::mul`, `aten::_fft_c2c`; a
    TorchDispatchMode `OpOverload` stringifies as `aten.mul.Tensor`. Both have
    to reduce to `mul` or the time and byte tables cannot be joined -- which is
    exactly the bug that made the bandwidth column come back empty the first
    time this ran.
    """
    op = str(op)
    op = op.split("::")[-1]                     # aten::mul      -> mul
    if op.startswith(("aten.", "prims.", "aten::")):
        op = op.split(".", 1)[1]
    parts = op.split(".")                       # mul.Tensor     -> mul
    return parts[0] or op


def family(op):
    """`aten.mul.Tensor` -> "math". Longest key wins, so `conv_transpose`
    beats `conv`. Leading underscores are stripped first: the dispatcher's
    names are `_convolution`, `_slow_conv2d_forward`, `_pad_circular`, and
    matching them raw put every one of them in "other".
    """
    bare = short(op).lstrip("_")
    best, score = "other", 0
    for fam, keys in _FAMILY:
        for k in keys:
            kk = k.lstrip("_")
            if (bare == kk or bare.startswith(kk)) and len(kk) > score:
                best, score = fam, len(kk)
    return best


def _fmt_shapes(shapes, keep=3, width=34):
    """The input shapes, trimmed to the ones that carry size."""
    big = [s for s in shapes if s and len(s) >= 2]
    if not big:
        big = [s for s in shapes if s]
    txt = " ".join("x".join(str(d) for d in s) for s in big[:keep])
    return txt[:width]


# ---------------------------------------------------------------------------
#  TIME
# ---------------------------------------------------------------------------
def op_times(model, y, E, sigma, device, iters=3, warmup=3):
    """Per-(op, input shapes) device time, via torch.profiler.

    -> `(rows, total_ms, gpu)`, `rows` = list of dicts sorted by time.
    `total_ms` is the summed SELF device time -- the honest denominator, since
    self time does not double-count an op's children.

    `gpu` is `dict(busy_ms, launches)` on CUDA, else None: the summed execution
    time and the count of the GPU KERNELS themselves. The profiler lists each
    kernel as its own event AND credits its time to the aten op that launched
    it, so keeping both double-counts (it reported 119% of the wall time). The
    kernel events are therefore not rows; they give the time the GPU was
    actually busy, and wall minus that is the time it sat idle waiting for the
    host to issue the next kernel.
    """
    from torch.profiler import ProfilerActivity, profile

    cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):                      # cudnn autotune, allocator
            model(y, E=E, sigma=sigma)
    if cuda:
        torch.cuda.synchronize()

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts, record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(iters):
                model(y, E=E, sigma=sigma)
        if cuda:
            torch.cuda.synchronize()

    from torch.autograd import DeviceType

    rows, kernel_rows = [], []
    for ev in prof.key_averages(group_by_input_shape=True):
        t = ev.self_device_time_total if cuda else ev.self_cpu_time_total
        if t <= 0 or ev.count == 0:
            continue
        row = dict(op=short(ev.key), family=family(ev.key),
                   shapes=_fmt_shapes(ev.input_shapes),
                   ms=t / 1e3 / iters,
                   calls=ev.count / iters)
        is_kernel = cuda and getattr(ev, "device_type", None) == DeviceType.CUDA
        (kernel_rows if is_kernel else rows).append(row)

    gpu = None
    if kernel_rows:
        busy = sum(r["ms"] for r in kernel_rows)
        attributed = sum(r["ms"] for r in rows)
        if attributed < 0.5 * busy:
            # this torch does not credit kernels to their aten op: the kernel
            # rows are the only timing there is, so keep them as before
            rows += kernel_rows
        else:
            gpu = dict(busy_ms=busy, launches=sum(r["calls"] for r in kernel_rows))
            if busy - attributed > 0.05:
                # kernels no aten op launched, e.g. the fused Triton clip
                rows.append(dict(op="(kernels outside aten)", family="other", shapes="",
                                 ms=busy - attributed, calls=0))
    rows.sort(key=lambda r: -r["ms"])
    return rows, sum(r["ms"] for r in rows), gpu


# ---------------------------------------------------------------------------
#  BYTES
# ---------------------------------------------------------------------------
def _macs(func, args, out):
    """Multiply-accumulates for a convolution or matmul; 0 for everything else.

    For `aten.convolution(input, weight, bias, stride, padding, dilation,
    transposed, output_padding, groups)` each output element costs
    `C_in/groups * prod(kernel)` MACs, which is `weight[0].numel()`. A transposed
    conv scatters, so its count follows the INPUT.

    This is what explains a cost difference that `M / s^2` cannot: an LPDS
    analysis conv has C_in = 1, so it does ~49 MACs per output element and is
    pure bandwidth; a U-Net conv at C_in = 72 does ~648 and lives on the compute
    side of the roofline, where there is two orders of magnitude more headroom.
    """
    name = short(func)
    o = out[0] if isinstance(out, (list, tuple)) and out else out
    if not torch.is_tensor(o):
        return 0.0
    if "convolution" in name or "conv2d" in name or "conv_transpose" in name:
        if len(args) < 2 or not torch.is_tensor(args[1]):
            return 0.0
        w = args[1]
        per_out = w.numel() / max(w.shape[0], 1)
        transposed = bool(args[6]) if len(args) > 6 else "transpose" in name
        n = args[0].numel() if (transposed and torch.is_tensor(args[0])) else o.numel()
        return float(n) * per_out
    if name in ("mm", "bmm", "matmul", "addmm", "baddbmm"):
        a = args[-2] if name in ("addmm", "baddbmm") else args[0]
        if torch.is_tensor(a) and a.dim() >= 2:
            return float(o.numel()) * a.shape[-1]
    return 0.0


class _ByteCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Exact bytes read + written per aten op. No timing, so no distortion.

    Counts every tensor argument as read and every tensor result as written,
    which is the traffic an unfused kernel must move.  A broadcast operand is
    counted at its own size, not the output's -- that is what it costs to read.
    Views are skipped entirely (`_FREE`): they launch no kernel.
    """

    def __init__(self, by_shape=False):
        super().__init__()
        self.by_shape = by_shape
        self.bytes = defaultdict(float)
        self.macs = defaultdict(float)
        self.calls = defaultdict(int)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **(kwargs or {}))
        name = short(func)
        if name in _FREE:
            return out
        key = (name, family(func))
        n = 0
        big = None          # the largest tensor involved, for `by_shape`

        def acc(x):
            nonlocal n, big
            if torch.is_tensor(x):
                n += x.numel() * x.element_size()
                if big is None or x.numel() > big[0]:
                    big = (x.numel(), tuple(x.shape), str(x.dtype))

        for a in list(args) + list(kwargs.values()):
            if isinstance(a, (list, tuple)):
                for b in a:
                    acc(b)
            else:
                acc(a)
        if isinstance(out, (list, tuple)):
            for b in out:
                acc(b)
        else:
            acc(out)
        self.macs[key] += _macs(func, args, out)
        if self.by_shape and big is not None:
            # the dominant tensor is what identifies the call site: `mul` on a
            # 20-coil k-space tensor is the Gram, on the M-channel latent it is
            # the prox or the dual update
            key = key + ("x".join(str(d) for d in big[1])
                         + " " + big[2].replace("torch.", ""),)
        self.bytes[key] += n
        self.calls[key] += 1
        return out


def op_bytes(model, y, E, sigma, by_shape=False, macs=False):
    """-> `{(op, family[, shape]): (MB, calls)}` for one forward.

    `macs=True` returns `(MB, calls, MMACs)` instead. MAC/byte is the number
    that decides which side of the roofline an op sits on: an L40S retires
    ~105 FLOP per byte of DRAM traffic, so anything under ~50 MAC/byte is
    bandwidth-bound no matter how it is written.
    """
    ctr = _ByteCounter(by_shape=by_shape)
    with torch.no_grad(), ctr:
        model(y, E=E, sigma=sigma)
    if macs:
        return {k: (v / 2 ** 20, ctr.calls[k], ctr.macs[k] / 1e6)
                for k, v in ctr.bytes.items()}
    return {k: (v / 2 ** 20, ctr.calls[k]) for k, v in ctr.bytes.items()}


# ---------------------------------------------------------------------------
#  report
# ---------------------------------------------------------------------------
def report(model, y, E, sigma, device, clean_ms=None, top=22, iters=3):
    """Print the attributed breakdown. Returns the per-op rows."""
    rows, total, gpu = op_times(model, y, E, sigma, device, iters=iters)
    mb = op_bytes(model, y, E, sigma)

    # bytes are per-op (not per-shape), so fold the per-shape times to match
    # before dividing -- otherwise the bandwidth column is nonsense
    by_op = defaultdict(float)
    for r in rows:
        by_op[(r["op"], r["family"])] += r["ms"]

    tag = "device" if device.type == "cuda" else "cpu"
    print(f"--- attributed op breakdown ({tag} self time, mean of {iters} "
          f"forwards) ---")
    if clean_ms:
        print(f"    summed self time {total:.1f} ms vs clean wall {clean_ms:.1f} ms"
              f"   ({100 * total / clean_ms:.0f}% attributed)")
        if gpu:
            idle = clean_ms - gpu["busy_ms"]
            print(f"    GPU busy {gpu['busy_ms']:.1f} ms in {gpu['launches']:.0f} kernel launches"
                  f" -> idle {idle:.1f} ms ({100 * idle / clean_ms:.0f}% of the wall):"
                  f" host-side cost (Python, dispatch, launch),"
                  f" ~{1e3 * idle / max(gpu['launches'], 1):.1f} us per launch")
    print(f"    {'op':<22}{'family':<10}{'shapes':<34}{'ms':>8}{'%':>6}"
          f"{'calls':>7}{'us/call':>9}")
    for r in rows[:top]:
        print(f"    {r['op']:<22}{r['family']:<10}{r['shapes']:<34}"
              f"{r['ms']:>8.2f}{100 * r['ms'] / total:>6.1f}{r['calls']:>7.0f}"
              f"{1e3 * r['ms'] / max(r['calls'], 1):>9.1f}")
    rest = sum(r["ms"] for r in rows[top:])
    if rest > 0:
        print(f"    {'(' + str(len(rows) - top) + ' more rows)':<66}"
              f"{rest:>8.2f}{100 * rest / total:>6.1f}")

    print(f"\n    {'family':<12}{'ms':>9}{'%':>7}{'MB/fwd':>10}{'GB/s':>9}")
    fam_ms, fam_mb = defaultdict(float), defaultdict(float)
    for (op, fam), t in by_op.items():
        fam_ms[fam] += t
    for (op, fam), (m, _c) in mb.items():
        fam_mb[fam] += m
    for fam in sorted(fam_ms, key=lambda f: -fam_ms[f]):
        m, t = fam_mb.get(fam, 0.0), fam_ms[fam]
        bw = (m / 1024) / (t * 1e-3) if t > 0 and m > 0 else 0.0
        print(f"    {fam:<12}{t:>9.2f}{100 * t / total:>7.1f}{m:>10.0f}"
              f"{(f'{bw:.0f}' if bw else '-'):>9}")

    card, peak = dram_peak(device)
    print(f"\n    per-op bandwidth -- what is worth attacking"
          f"  ({card} DRAM peak {peak} GB/s)")
    print(f"    {'op':<22}{'family':<10}{'ms':>8}{'MB/fwd':>9}{'GB/s':>8}  verdict")
    joint = []
    for (op, fam), (m, c) in mb.items():
        t = by_op.get((op, fam))
        if not t or m <= 0:
            continue
        joint.append((t, op, fam, m, (m / 1024) / (t * 1e-3)))
    for t, op, fam, m, bw in sorted(joint, reverse=True)[:14]:
        # the same fractions of the peak the L40S cut-offs (1000 / 350) were
        v = ("L2-resident, few bytes to win" if bw > 1.16 * peak else
             "DRAM-bound: bytes -> time" if bw > 0.4 * peak else
             "launch/latency-bound: cut KERNELS")
        print(f"    {op:<22}{fam:<10}{t:>8.2f}{m:>9.0f}{bw:>8.0f}  {v}")
    return rows, by_op, mb


# ---------------------------------------------------------------------------
#  HOST
# ---------------------------------------------------------------------------
def host_report(model, y, E, sigma, device, clean_ms=None, top=26, iters=3, warmup=2):
    """Where the HOST spends the forward, call by call (cProfile).

    `report` above accounts for the GPU: when it says the GPU was busy for only
    part of the wall time, the rest is the host deciding what to launch next,
    and no device-time row can show where. A CUDA call returns as soon as its
    kernel is queued, so under cProfile a torch call's OWN time is its host
    cost -- dispatch, argument checks, cuDNN / cuFFT plan lookup, the launch --
    and a Python function's own time is interpreter work. A call that BLOCKS
    on the GPU (`.item()`, `bool(tensor)`, a `.cpu()`) shows up here as one
    very expensive row, which is worth knowing too.

    INFIX ARITHMETIC (`a - b`, `t * r`) is not a call cProfile can see: it is
    charged to the Python function it is written in. So a row like
    `lpds.py:91 forward` is that function's interpreter time PLUS the host
    cost of every `+ - *` in its body.

    cProfile adds about a microsecond per call, so the total is inflated;
    read the split and the ranking, not the absolute figures. Counting kernel
    launches does NOT predict this: removing 860 tiny `cat`s (the planar
    weight cache) saved 2 ms on a forward whose "per-launch" average said 8.
    """
    import cProfile
    import os
    import pstats

    cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(y, E=E, sigma=sigma)
        if cuda:
            torch.cuda.synchronize()
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(iters):
            model(y, E=E, sigma=sigma)
        pr.disable()
        if cuda:
            torch.cuda.synchronize()

    rows = []
    for (fn, line, name), (_cc, nc, tt, _ct, _callers) in pstats.Stats(pr).stats.items():
        if name in ("disable", "enable") or "cProfile" in name:
            continue
        builtin = fn == "~"
        if builtin:
            # "<built-in method torch.conv2d>", "<method 'abs' of 'torch._C.TensorBase' objects>"
            label = name.strip("<>").replace("built-in method ", "")
            if label.startswith("method "):
                label = "Tensor." + label[len("method "):].split(" of ")[0]
            label = label.replace("'", "")
        else:
            label = f"{os.path.basename(fn)}:{line} {name}"
        rows.append((1e3 * tt / iters, nc / iters, label, builtin))
    rows.sort(reverse=True)

    total = sum(r[0] for r in rows)
    t_builtin = sum(r[0] for r in rows if r[3])
    n_builtin = sum(r[1] for r in rows if r[3])
    print(f"--- host time per forward (cProfile own time, mean of {iters}; inflated "
          f"by ~1 us per call) ---")
    print(f"    profiled {total:.1f} ms" + (f" vs clean wall {clean_ms:.1f} ms" if clean_ms else "")
          + f":  torch / C calls {t_builtin:.1f} ms in {n_builtin:.0f} calls"
          f" ({1e3 * t_builtin / max(n_builtin, 1):.1f} us each),"
          f"  Python functions {total - t_builtin:.1f} ms")
    print("    (infix + - * are charged to the Python function they are written in)")
    print(f"    {'call':<58}{'ms':>8}{'%':>6}{'calls':>8}{'us/call':>9}")
    for ms, calls, label, _b in rows[:top]:
        print(f"    {label[:57]:<58}{ms:>8.2f}{100 * ms / total:>6.1f}{calls:>8.0f}"
              f"{1e3 * ms / max(calls, 1):>9.1f}")
    rest = sum(r[0] for r in rows[top:])
    if rest > 0:
        print(f"    {'(' + str(len(rows) - top) + ' more)':<58}{rest:>8.2f}{100 * rest / total:>6.1f}")
    return rows
