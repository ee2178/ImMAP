#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:45:00
#SBATCH --job-name=GLPDSProf
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# Where does an MGGroupLPDS forward+backward actually spend its time, on the
# hardware and with the compiled FlexAttention kernel it will actually run on?
#
# Everything measured about this model so far was CPU with UNCOMPILED flex,
# where fixed kernel overhead swamps every knob -- Mh 64->16 came out at 1.06x
# and dK 5->8 came out SLOWER. Those numbers are not usable. This job replaces
# them.
#
# THE ONE NUMBER THAT MATTERS FIRST: what fraction of the forward is the group
# prox? On the gather backend at small scale it was 91%, which would make every
# attention knob decisive. Under compiled flex it could plausibly be 40-60%, in
# which case the conv / physics path matters more and the attention sweep is
# beside the point. Everything below is conditional on that number.
#
# Uses CUDA events with explicit synchronize(). Wall-clock timing around GPU
# work measures queue submission, not execution.
#
# Reads config/<anatomy>/mg/mggrouplpds_<R>.json unchanged -- no training, no
# checkpoints written.
#
# Submit:   sbatch slurm/profile_grouplpds.sbatch
# Options:  ANATOMY=brain RTAG=R4 IMG=304x704 sbatch slurm/profile_grouplpds.sbatch

ANATOMY="${ANATOMY:-knee}"
RTAG="${RTAG:-R8}"
MODEL="${MODEL:-mggrouplpds}"
CONFIG_ROOT="${CONFIG_ROOT:-config}"
IMG="${IMG:-304x704}"                # padded H x W to profile at
COILS="${COILS:-15}"
BACKWARD="${BACKWARD:-1}"            # 1 = time fwd+bwd (what training pays)

source ~/.bashrc
conda activate gcdl

for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then IMMAP_ROOT="${_c}"; break; fi
done
[ -f "${IMMAP_ROOT:-}/train.py" ] || { echo "[prof] cannot locate ImMAP." >&2; exit 1; }
cd "${IMMAP_ROOT}"
set -eo pipefail
mkdir -p logs

CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"
[ -f "${CONFIG}" ] || { echo "[prof] ${CONFIG} missing." >&2; exit 1; }
echo "[prof] config=${CONFIG}  img=${IMG}  coils=${COILS}  backward=${BACKWARD}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - "${CONFIG}" "${IMG}" "${COILS}" "${BACKWARD}" <<'PY'
import json, math, sys, time, collections
import torch

from models import build_model
from models.prox import GroupThreshold
from operators import FFT2D, Mask, Sense
from physics.mask import make_acc_mask

cfg_path, img, coils, do_bwd = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
H, W = (int(v) for v in img.lower().split("x"))
dev = torch.device("cuda")
CFG = json.load(open(cfg_path))
BASE = dict(CFG["model"]["params"])
R, ACS = CFG["mri"]["R"], CFG["mri"]["acs_lines"]
pad = BASE["s"] * 2 ** (len(BASE["K"][1]) - 1)
assert H % pad == 0 and W % pad == 0, f"{H}x{W} must be a multiple of pad_stride={pad}"

# ---------------------------------------------------------------- fixtures
g = torch.Generator(device="cpu").manual_seed(0)
sm = torch.randn(1, coils, H, W, generator=g, dtype=torch.complex64)
sm = (sm / (sm.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)).to(dev)
mask = make_acc_mask(shape=(H, W), accel=R, acs_lines=ACS, mode="uniform")
mask = mask.reshape(1, 1, H, W).to(torch.complex64).to(dev)
E = Mask(mask) @ FFT2D() @ Sense(sm)
x0 = torch.randn(1, 1, H, W, generator=g, dtype=torch.complex64).to(dev)
y = E(x0)

def make(**over):
    P = dict(BASE); P.update(over)
    torch.manual_seed(0)
    m = build_model({"model": {"type": CFG["model"]["type"], "params": P}}).to(dev)
    # train.py does exactly this; without it the flex kernel runs uncompiled and
    # every measurement below is of the eager fallback instead.
    if getattr(m, "attn_backend", None) == "flex":
        m.compile_flex()
    return m

def timed(m, n=5, warmup=3, backward=do_bwd):
    """CUDA-event timing. Wall clock here would measure queue submission."""
    def once():
        if backward:
            m.zero_grad(set_to_none=True)
            o = m(y, E=E, sigma=0.005)
            xh = o[0] if isinstance(o, tuple) else o
            (xh.abs() - x0.abs()).pow(2).mean().backward()
        else:
            with torch.no_grad():
                m(y, E=E, sigma=0.005)
    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.reset_peak_memory_stats()
    s.record()
    for _ in range(n):
        once()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n, torch.cuda.max_memory_allocated() / 1e9

print(f"\n[prof] {CFG['model']['type']}  M={BASE['M']} Mh={BASE['Mh']} "
      f"W={BASE['window']} dK={BASE['dK']} K={BASE['K']} "
      f"backend={BASE['attn_backend']} block={BASE.get('flex_block_size')}")
print(f"[prof] {H}x{W}, {coils} coils, R={R}, acs={ACS}, "
      f"{'fwd+bwd' if do_bwd else 'fwd only'}")

net = make()
print(f"[prof] params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

# ============================================================ 1. prox share
base_ms, base_gb = timed(net)
print(f"\n=== 1. THE HEADLINE ===")
print(f"{'configuration':40s}{'ms':>10s}{'peak GB':>10s}{'share':>9s}")
print(f"{'full forward' + ('+backward' if do_bwd else ''):40s}{base_ms:>10.1f}{base_gb:>10.2f}{'100%':>9s}")

import types
gts = [m for m in net.modules() if isinstance(m, GroupThreshold)]
saved = [(m, m.forward) for m in gts]
for m in gts:
    def soft(self, z, sigma=None, cache=None, **k):
        return torch.sgn(z) * torch.clamp(z.abs() - 0.01, min=0), (cache or {})
    m.forward = types.MethodType(soft, m)
noattn_ms, noattn_gb = timed(net)
for m, f in saved:
    m.forward = f
print(f"{'group prox -> soft threshold':40s}{noattn_ms:>10.1f}{noattn_gb:>10.2f}"
      f"{100*noattn_ms/base_ms:>8.0f}%")
share = 100 * (1 - noattn_ms / base_ms)
print(f"\n  --> the group prox is {share:.0f}% of runtime.")
print(f"      {'Above ~70%: the attention sweep below is where the time is.'if share>70 else ''}"
      f"{'40-70%: attention matters, but so does everything else.' if 40<=share<=70 else ''}"
      f"{'Below 40%: do NOT tune attention first -- profile the conv/FFT path.' if share<40 else ''}")

# ============================================================ 2. knob sweep
print(f"\n=== 2. ATTENTION KNOBS (only meaningful if the share above is high) ===")
print(f"{'change':40s}{'ms':>10s}{'peak GB':>10s}{'speedup':>10s}")
sweeps = [("baseline", {}),
          ("Mh 64 -> 32", dict(Mh=32)),
          ("Mh 64 -> 16", dict(Mh=16)),
          ("window 15 -> 11", dict(window=11)),
          ("window 15 -> 9", dict(window=9)),
          ("window 15 -> 21", dict(window=21)),
          ("dK 5 -> 8", dict(dK=8)),
          ("dK 5 -> 3", dict(dK=3)),
          ("flex_block 128 -> 64", dict(flex_block_size=64)),
          ("attn_backend flex -> gather", dict(attn_backend="gather")),
          ("Mh=32 + window=11", dict(Mh=32, window=11))]
for tag, over in sweeps:
    try:
        m = make(**over)
        ms, gb = timed(m)
        print(f"{tag:40s}{ms:>10.1f}{gb:>10.2f}{base_ms/ms:>9.2f}x")
        del m; torch.cuda.empty_cache()
    except Exception as ex:
        print(f"{tag:40s}{'--':>10s}{'':>10s}   {type(ex).__name__}: {str(ex)[:44]}")

# ============================================================ 3. per level
print(f"\n=== 3. WHERE THE ATTENTION RUNS, AND WHAT IT CAN SEE ===")
calls, builds = collections.Counter(), collections.Counter()
owners = collections.Counter()
net2 = make()
for name, m in net2.named_modules():
    if isinstance(m, GroupThreshold):
        ob, of = m._build_gamma, m.forward
        def mk(ob, of, nm):
            def bg(z, s, _o=ob, _n=nm):
                builds[tuple(z.shape[-2:])] += 1
                owners["dF" if ".dF." in _n else "smoother"] += 1
                return _o(z, s)
            def fw(z, s=None, c=None, _o=of, **k):
                calls[tuple(z.shape[-2:])] += 1
                return _o(z, s, c, **k)
            return bg, fw
        m._build_gamma, m.forward = mk(ob, of, name)
with torch.no_grad():
    net2(y, E=E, sigma=0.005)
print(f"{'latent grid':>14s}{'prox calls':>12s}{'Gamma builds':>14s}{'reuse':>8s}")
for k in sorted(set(calls) | set(builds), key=lambda t: -t[0] * t[1]):
    c, b = calls[k], builds[k]
    print(f"{f'{k[0]}x{k[1]}':>14s}{c:>12d}{b:>14d}{c/max(b,1):>7.1f}x")
tc, tb = sum(calls.values()), sum(builds.values())
print(f"{'TOTAL':>14s}{tc:>12d}{tb:>14d}{tc/max(tb,1):>7.1f}x"
      f"   (dK={BASE['dK']} implies ~{BASE['dK']}.0x if the caches were shared)")
print(f"  rebuilds by owner: {dict(owners)}")
print(f"  -> dF holds deep-copied prox_fine/prox_coarse with SEPARATE caches, so it")
print(f"     rebuilds adjacencies the smoothers already built at the same grid.")

Wn = BASE["window"]
print(f"\n  alias reach (undersampling is along the {W}-axis, spacing {W}/{R} = {W//R} px):")
print(f"  {'lvl':>4s}{'latent':>12s}{'alias spacing':>15s}{'W half':>9s}{'reaches':>9s}{'W needed':>10s}")
for l in range(len(BASE["K"][1])):
    gh, gw = H // BASE["s"] // 2 ** l, W // BASE["s"] // 2 ** l
    sp = gw / R
    print(f"  {l:>4d}{f'{gh}x{gw}':>12s}{sp:>15.1f}{Wn//2:>9d}"
          f"{('yes' if Wn // 2 >= sp else 'no'):>9s}{2*int(math.ceil(sp))+1:>10d}")
print("\n[prof] done.")
PY
