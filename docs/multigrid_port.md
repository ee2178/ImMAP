# Sljiva (Julia/Lux) -> ImMAP (PyTorch): multigrid + LADMM port

Source files ported: `src/networks/mg_lista.jl`, `mg_group.jl`, `cdlnet.jl`,
`ladmm.jl` (plus the `lista.jl` / `layers.jl` / `group.jl` / `operators.jl`
pieces they depend on).

## File map

| Julia | PyTorch |
|---|---|
| `networks/lista.jl` (`LISTALayer`, `LISTA`) | `models/lista.py` |
| `networks/layers.jl` (`Polynomial`, `SoftThreshold`, `FenchelProx`) | `models/prox.py` |
| `networks/group.jl` (`GroupThreshold`, `NonLocalSimilarity`) | `models/prox.py` (`GroupThreshold`) |
| `networks/mg_group.jl` (`∂g`, `∂g_simple`) | `models/prox.py` (`GroupThreshold.subgradient`) |
| `networks/mg_lista.jl` (`mgVCycleLayer`, `mgObjectiveDownsampleLayer`, widening) | `models/multigrid.py` |
| `networks/cdlnet.jl` (`CDLNet`) | `models/multigrid.py` (`MGCDLNet`) |
| `networks/ladmm.jl` | `models/ladmm.py` |
| `operators.jl` `Resample` / `galerkin` / `meanpool` / `upsample` | `operators/resample.py` |
| `operators.jl` accessors / functional updaters | `operators/accessors.py` |
| `solver.jl` `cg` / `tcg` + its `rrule` | `solvers/cg.py` (`batched_cg`, `tcg`) |

## Layout conversion

Julia is batch-last `(H, W, C, B)`, PyTorch is batch-first `(B, C, H, W)`.
The conv weights happen to line up:

| | Julia (Lux) | PyTorch |
|---|---|---|
| analysis `A: C -> M` | `Conv((p,p), C=>M)` -> `(p, p, C, M)` | `Conv2d(C, M, p)` -> `(M, C, p, p)` |
| synthesis `B: M -> C` | `ConvTranspose((p,p), M=>C)` -> `(p, p, C, M)` | `ConvTranspose2d(M, C, p)` -> `(M, C, p, p)` |

Both store the subband axis in the *same slot relative to the other axes*, which
is why `B.weight = A.weight` works in Julia and `B.weight = A.weight.conj()`
works here. Widening M is `repeat(w, 1, 1, 1, widen)` (dim 4) in Julia and
`w.repeat(widen, 1, 1, 1)` (dim 0) in torch.

Lux's `st` NamedTuple is replaced by an explicit `cache` dict threaded through
the call chain. It holds the group-attention adjacency `Gamma`, and its nesting
(`cache["_coarse"]`, `cache["_dF_fine"]`, ...) mirrors the state tree Lux built
implicitly. Levels have different `Gamma` shapes, which is exactly why each
needs its own slot.

## Deliberate departures from the Julia source

1. **`GroupThreshold` rigorous subgradient (mg_group.jl §Rigorous).** The Julia
   expression forms `Gamma^T ( c u / xi_a )`, but `u` carries the inner index
   and `1/xi_a` the outer one, so `u` does not commute through `Gamma^T`. The
   two agree only when `Gamma = I`. `models/prox.py` implements the actual
   derivative, `W_alpha^T [ c . u . Gamma^T (1 / xi_a) ]`, at identical cost
   (one forward and one transposed adjacency apply);
   `tests/test_multigrid.py` verifies it against autograd on the energy.
   The same function also uses `c = W_beta^T tau` rather than `W_beta^T 1` with
   `tau` applied afterwards -- identical for a channel-uniform `tau`, correct
   when it varies per subband.
2. **Restriction padding.** Julia always pads `(1,0,1,0)` before mean-pooling,
   which shifts an even-sized grid by half a cell and attenuates its first
   row/column. Since inputs are padded up to a multiple of `s * 2^(L-1)`, every
   level is even and the padding is only needed for odd sizes. `restrict` pads
   only when a dimension is odd; `julia_compat=True` restores the old behaviour.
3. **`align_corners`.** NNlib defaults to vertex-centred (`True`); the 2x2 mean
   restriction is cell-centred, so `False` is the variationally consistent
   partner. Exposed on `Resample` / `prolong`.
4. **Padding stride.** Julia pads to `s * 2^L` for `L` levels; only `L - 1`
   coarsenings actually happen, so `MGCDLNet` pads to `s * 2^(L-1)`.
5. **Dead parameters.** The outermost V-cycle never receives a `pi` (the outer
   LISTA calls it with `pi=None`), so its smoothers no longer allocate the
   `eta` polynomial at all. In Julia those weights exist and never get a
   gradient.
6. **Batched CG.** Julia reduces CG inner products per batch element
   (`dims=1:N-1`); this repo's existing `cg` uses one global scalar. `tcg` uses
   the batched form, so a well-conditioned slice is not throttled by the worst
   one in its batch.
7. **`project!` recursion.** Replaced by a `project_()` hook discovered via
   `nn.Module.modules()`, so a new constrained submodule cannot be forgotten.

## The four multigrid model types

All one class, `MGCDLNet`; the names differ only in which flags they pin, and
exist so a config states its intent and fails loudly if it contradicts it.

| type | prox | read-out | flags |
|---|---|---|---|
| `MGCDLNet`    | soft threshold  | `D z`      | -- |
| `MGGroupCDL`  | group threshold | `D z`      | requires `W > 1`, `Mh` |
| `MGLPDS`      | clipping        | `y~ - D z` | `dual=True` |
| `MGGroupLPDS` | group clipping  | `y~ - D z` | `dual=True`, requires `W > 1`, `Mh` |

The LPDS pair is exactly the CDL pair with a residual connection around the
threshold -- Moreau's identity, `prox_{g*}(u) = u - prox_g(u)`, implemented once
in `FenchelProx` and verified in `tests/test_init.py`. `K` as a plain int drops
the V-cycle and leaves an ordinary (Group)CDLNet built from the same blocks, so
the non-multigrid baselines come from the same code path.

## Attention backends (MG-GroupCDL)

`attn_backend="gather"` (default) materialises the `(B, Q, K)` window values as
a `Circulant`. Exact, complex-capable, differentiable on CPU, and the only
backend that supports the convex adjacency blend of Alg. 4. Memory is
`O(B Mh Q W^2)` -- at `W=35` on a full-resolution latent that is gigabytes.

`attn_backend="flex"` uses `FlexAdjacency` (new, in `models/circulant_flex.py`):
the fused FlexAttention kernel, nothing of size `(B, Q, K)` allocated. Caveats:
real-valued queries/keys (complex ones are `[Re; Im]`-stacked, which leaves
distance and realdot unchanged), no adjacency blend (the query/key pair is
re-cached instead), and **torch has no CPU backward for FlexAttention, so this
backend trains on GPU only**. Call `net.compile_flex()` once after `.to(device)`
for the compiled kernel.

### Which similarities flex can carry

A `score_mod` sees exactly one number per `(q, k)` pair -- the raw dot product --
plus anything indexable by `b / h / q_idx / kv_idx`. So a similarity is fusible
**iff it is a pointwise function of `q.k` plus per-query and per-key terms**:

| sim | score_mod | fusible? |
|---|---|---|
| `dot` / `realdot` | identity | yes |
| `distance` | `score - ½‖k‖²` (per-key bias) | yes |
| `pidot` | `\|score\|` | real features only |
| `pidistance` | `\|score\| - ½‖k‖²` | real features only |

`distance = -½‖q-k‖² = -½‖q‖² + q·k - ½‖k‖²` and `pidistance` is the same with
`Re⟨q,k⟩` replaced by `|⟨q,k⟩|` -- the *phase-invariant* distance,
`-½ min_θ ‖q - e^{iθ}k‖²`. In both, the `-½‖q‖²` term is constant across `k` and
cancels inside the row softmax, so it never has to be formed.

For **real** features `⟨q,k⟩ = q·k`, so `|⟨q,k⟩|` is just `|score|` and the pi-
family fuses **exactly** (verified against the gather path in
`tests/test_init.py`, and the reduction itself checked densely in
`tests/test_synthetic_kspace.py::test_pidistance_score_mod_reduction`).

For **complex** features it is not expressible. The stacked `[Re; Im]` score is
`Re⟨q,k⟩ = [a;b]·[c;d]`, and the modulus needs `Im⟨q,k⟩ = [b;-a]·[c;d]` as well
-- a *second* bilinear form. One `flex_attention` call produces one score, and a
`score_mod` cannot see across heads or across calls, so there is no way to
combine them. (`|⟨q,k⟩|²` is a dot product in the lifted space of dimension
`(2Mh)²` -- 16384 at `Mh=64` -- which is not a usable head dim.)

`GroupThreshold._build_gamma` raises on the complex + pi- + flex combination
rather than silently attending on `Re⟨q,k⟩` (measured: the row softmaxes differ
by up to **0.41**, so the substitution is not a small perturbation).

**The transpose.** `FlexAdjacency.apply(transpose=True)` has to apply the same
`|·|`: its swapped call still scores `q_j · k_k`, so dropping the nonlinearity
there would transpose a *different* matrix. The similarity's own `-½‖k‖²` term
moves from a per-key bias to a per-query constant, which the existing `colsum`
factor already absorbs.

## The Triton backend: complex pidistance, fused

`attn_backend="triton"` (`models/circulant_triton.py`) is the case flex cannot
reach. A hand-written kernel can keep **two** accumulators, so the modulus is
available:

    re_jk = sum_c ( qr kr + qi ki )      im_jk = sum_c ( qi kr - qr ki )
    s_jk  = sqrt(re² + im² + eps) - ½ knorm_k + bias_k

That is what Julia's `CirculantAttention.jl` flash path does, and this is its
PyTorch counterpart.

**Not a matmul.** Tiling this as dense `tl.dot` blocks wastes ~86% of the work
-- a query attends `W² = 81` keys, but any tile large enough to feed a matmul
spans ~600. Instead each of the `W²` offsets is one reduction over channels,
`tl.sum(q * k_shifted, 1)`, which is entirely useful work; the shifted key tiles
overlap almost completely between adjacent offsets, so they come from L1/L2. The
`-½‖q‖²` term is constant across `k` and never formed.

**No atomics.** The circular window is symmetric, so key `k` is seen by queries
`j = k - o` over the same offset set negated. `_bwd_dkdv` walks that directly
rather than scattering out of `_bwd_dq`, which keeps gradients deterministic run
to run.

**The transpose reuses the forward kernel**, via the same lse identity
`FlexAdjacency` uses -- legal because `|⟨k,q⟩| = |conj⟨q,k⟩| = |⟨q,k⟩|` is
symmetric. The kernel's `lse` output is differentiable (`∂lse_j/∂s_jk = Γ_jk`,
folded into the softmax backward's `delta`), which is the gradient path the
transposed apply depends on.

Constraints: fp32, CUDA, `C` and `DV` each ≤ 128 after rounding up (`Mh` is
64-96 everywhere), and a **real** value tensor -- which every `apply_gamma` call
site already satisfies, since `PixelConv` has real weights and both sites pass
`_abs2(...)` or `1/xi_a`.

### Verifying it

`tests/test_triton_attention.py` is deliberately two layers, because they fail
for different reasons:

```bash
python -m tests.test_triton_attention --algebra-only   # Part A: runs anywhere
```

Part A re-expresses the kernel's forward (offset loop, streaming softmax) and
its hand-derived `dq`/`dk`/`dv`/`dbias` in plain PyTorch and checks them against
autograd -- no CUDA, no triton, no torch 2.x. It currently passes 34 checks at
~5e-7 across complex/real, pidistance/pidot, per-key bias, and the lse gradient
path. A failure here is an algebra error.

```bash
python -m tests.test_triton_attention                  # Part A + B + benchmark
```

Part B runs the compiled kernel against the gather path (forward, lse,
transpose, heads, wraparound, odd shapes, gradients, and end-to-end through
`GroupThreshold` in every subgradient mode), then benchmarks against gather at
`Mh=64, W=9, 160x160`. **A failure in Part B that Part A passed is a Triton
problem** -- indexing, masking, or the API -- not a derivation problem.

Until Part B has been run on a GPU, the generated configs stay on `--attn flex`.
Regenerate with `--attn triton` once it is green.

The non-obvious part is the TRANSPOSED apply, which the rigorous subgradient
needs and which a plain attention call cannot express. Writing the masked,
biased score as `S_jk` and `Z_j = sum_k exp(S_jk)`,

    (Gamma^T h)_k = sum_j exp(S_jk - log Z_j) h_j

is itself a windowed attention with q and k swapped and a PER-KEY bias of
`-log Z_j` -- exactly what a `score_mod` indexed by `kv_idx` expresses. Its own
softmax denominator is divided back out using the returned lse, and the factor
that comes back is `sum_j Gamma_jk`, a column sum of a row-stochastic matrix,
hence O(1): no large exponentials anywhere. The window relation (circular
Chebyshev distance) is symmetric, so one BlockMask serves both directions.
`tests/test_init.py` checks both directions against the dense gather path
(~3e-7). This replaces the custom `circulant_mh_flash_transposed_attention`
kernel the Julia source needs, with no new kernel.

BlockMasks are memoised process-wide by `get_block_mask`, keyed on
(grid, window, device) -- an unrolled multi-level network has one prox per layer
per level and they all want the same mask, and building one torch.compiles.

## Bug fixes to pre-existing code

* `models/components.py` -- `ConvTranspose2d` hard-coded `output_padding=1`,
  which torch rejects for `stride=1` (`output_padding` must be `< stride`).
  Now `stride - 1`: unchanged at `stride=2`, and `stride=1` works.
* `operators/mask.py` -- `Mask.adjoint` returned `mask * x` instead of
  `conj(mask) * x`. A no-op for real sampling masks, but wrong for the
  coil-map subproblem, which builds `E_x = M F diag(x)` with a complex image.

* `models/base.py::spectral_init` was a **silent no-op for every complex-valued
  model**. It did `self.A[k].weight.data /= scale`, but for `complex=True` the
  `weight` getter returns a freshly allocated `torch.complex(...)` tensor rather
  than a view of the storage -- so the division modified a temporary and was
  discarded, while still printing `Power method returns L = ...`. Measured
  before the fix: complex `CDLNet` initialised with `||B A|| = 149.6` instead of
  1, i.e. the ISTA step size the unrolling assumes was off by that factor.
  Real-valued models were unaffected (their getter returns the real storage).

  Fixed via `models/base.py::set_weight`, which picks the property setter or the
  Parameter depending on how the module exposes `.weight`; `init_filters`,
  `project_filters` and `spectral_init` all route through it, as do the ported
  modules. `tests/test_init.py` asserts `||B A|| = 1` for real AND complex
  CDLNet, GroupCDL and MGCDLNet, and pins the failing idiom as a regression
  test.

  **This changes fresh initialisation of existing complex configs.** Loading a
  checkpoint is unaffected (weights are overwritten by the `state_dict`), but a
  complex model trained from scratch before this fix started from a very
  different point -- reproducing old runs exactly requires reverting it.

* `models/components.py::ComplexConvTranspose2d` gained the same `weight`
  property as `_GaussConvNd`. Without it `self.B[k].weight = W` parked a plain
  attribute on the module and the filters were silently never initialised.
  Together with `set_weight`, this makes `IPALMNet` constructible again -- it
  raised `TypeError: cannot assign 'torch.Tensor' as parameter 'weight'` on
  every instantiation.

## Quick start

```python
from models.multigrid import MGCDLNet

# 3 outer V-cycles over 3 levels with 8 / 8 / 4 iterations, doubling subbands
net = MGCDLNet(K=[3, [8, 8, 4]], M=32, Mh=16, C=1, P=7, s=2, W=15, widen=2)
x_hat, z = net(y, E=E, sigma=sigma)
net.project()
```

Non-coarsest `iters` entries must be even (they split into pre/post smoothing).
`K` as a plain int gives an ordinary CDLNet from the same blocks.

```python
from models.ladmm import AltSplitCDLNet

net = AltSplitCDLNet(admm_iters=5, smap_update=True, implicit_cg=True,
                     denoiser_kws=dict(K=[2, [6, 6, 4]], M=32, Mh=16, s=2, W=15))
x_hat, extras = net(y, E=E, sigma=sigma)   # extras: z, x, u, smaps
```

```python
# MG-GroupCDL at a realistic window -- fused attention, GPU
net = MGCDLNet(K=[3, [8, 8, 4]], M=64, Mh=64, C=1, P=7, s=2, W=35, dK=5,
               attn_backend="flex").to("cuda").compile_flex()

# MG-GroupLPDS: the same network, residual connection around the threshold
net = MGCDLNet(..., dual=True)
```

Configs: `config/bsd432/mg_denoiser.json` (MGCDLNet),
`config/bsd432/mg_groupcdl.json` (MGGroupCDL),
`config/bsd432/mg_grouplpds.json` (MGGroupLPDS),
`config/knee/ladmm.json` (AltSplitCDLNet).

Tests: `python -m tests.test_multigrid`, `tests.test_ladmm`,
`tests.test_integration`, `tests.test_init`,
`tests.test_synthetic_kspace`.

## fastMRI reconstruction on synthetic k-space

The multigrid models above were built and validated on denoising. Running them
on fastMRI reconstruction needed the `kspace_type: "simulated"` path, which was
dead code: `prepare_measurement` called an `mri_awgn` it never imported, and
that function referenced an undefined name, ignored the sensitivity maps
entirely (returning single-coil k-space against a multicoil `E`), and drew one
sigma for the whole batch.

`operators/noise.py::mri_awgn` is now the counterpart of
`genobs(clo::SyntheticMRIReco, ...)` in `Sljiva/src/closures/mrireco.jl`:

    x_mc  <- s . x                         Sense forward
    y     <- mask . F (x_mc + sigma . n)   AWGN in the coil-image domain

Noise goes in before the transform, as in the Julia closure's
`acgn(slice_rng, xmc, L)`. `fftc` is `norm="ortho"`, so that is distributionally
the same as adding it in k-space. `sigma * randn` on a complex tensor has total
variance `sigma^2` in torch; `tests/test_synthetic_kspace.py` pins that
convention rather than trusting it.

Maps are assumed **unit-RSS**, which is what this repo's map generation already
produces, so unlike `mrireco.jl:220` nothing renormalizes them. That assumption
is what makes `sigma` meaningful: `E^H y = x sum_c |s_c|^2 + sum_c conj(s_c) n_c`
has residual variance `sigma^2 sum_c |s_c|^2`, which is `sigma^2` exactly when the
maps are unit-RSS. The test asserts both the adjointness of `E` and that the
adjoint residual tracks the requested sigma, rather than renormalizing to force it.

`prepare_measurement` still returns the maps `y` is consistent with, and
`train_recon` builds `E = Mask @ FFT2D @ Sense(extra["smaps"])` from them --
a pass-through in the simulated branch, but the whitening branch genuinely
replaces the maps, so the uniform contract is worth keeping.

**`kspace_type: "simulated"` is wired for `task: "recon"` only.** The
`train_joint_denoising_recon` and `train_ipalm` loops still build `E` before
calling `prepare_measurement`. They run on measured k-space today; fix the
ordering there before switching them.

### DC handling: why mean subtraction needed fixing

CDLNet-style preprocessing subtracts a mean from the input and adds it back at
the read-out, so the dictionary does not have to spend atoms on DC. For
denoising that is exactly right. For reconstruction it is not, and the algebra
says why. Split the image as `x = v + mu 1`:

    1/2 ||y - E(v + mu 1)||^2  =  1/2 || (y - mu E 1) - E v ||^2

    grad_v  =  E^H E v  -  ( E^H y  -  mu E^H E 1 )

A LISTA layer computes `z <- prox(z - A(E^H E B z - y~))`, so the surrogate the
network must be handed is

    y~  =  E^H y  -  mu . E^H E 1                                        (*)

with `mu` added back at the read-out. The old `preprocessing/image.py` path used
`E^H y - mean(E^H y) 1`, which is wrong twice over: the DC image has to go
through `E^H E`, and `mean(E^H y)` is not `mean(x)` once coil maps are involved
(`E^H E` is a Fourier multiplier only in the single-coil case). Measured on a
6-coil phantom at R=4: the plain-mean gradient is off by **16% relative**, and
`mean(E^H y)` misses `mean(x)` by ~30%. Both errors vanish at `E = I`
(`E^H E 1 = 1`), which is why denoising never showed it. Sljiva's
`KSpacePreprocess` has the same issue -- it fixes the padding half and keeps the
plain mean.

**Choosing mu.** Any scalar makes (*) an exact reformulation as long as the same
one is added back; what `mu` buys is centering (and it does shift the
regularizer, which acts on the code of `x - mu 1`). So take the `mu` that best
explains the measured DC content -- the exact minimizer of `1/2||y - E(mu 1)||^2`:

    mu = <E 1, y> / ||E 1||^2 = sum(E^H y) / sum(E^H E 1)                (**)

which is the old `mean(E^H y)` divided by `mean(E^H E 1)`, reuses the `E^H E 1`
that (*) needs anyway, and collapses to `mean(y)` at `E = I`. An ACS-based
estimate (inverse-FFT the fully sampled centre, average the low-resolution
image) is the other natural route and also works, but it is only approximate --
it needs the window's DC gain to be one and the maps to be smooth across it --
and it adds a window hyperparameter that (**) does not need. On the same
phantom (**) recovers `mean(x)` to 0.004 where `mean(E^H y)` is off by 0.21.

`preprocessing/kspace.py` implements this as `preproc="kspace"`, which also
**pads the operator** alongside `y~` (mask resampled, coil maps reflect-padded)
the way `KSpacePreprocess` does -- so reconstruction no longer needs data sized
to a multiple of `pad_stride`. `MGCDLNet` accepts `preproc` in
`{"image", "kspace", "identity"}` and `_check_padding` now points at `kspace`
when `image` would silently pad past a real operator.

Padding an MRI operator is inherently approximate (a Fourier transform on a
larger grid is a different transform), so the configs still size data to a
multiple of `pad_stride` where the pad is empty and this is the identity --
320x320 knee data with `s=2` and 3 levels hits that exactly.

Two smaller fixes the grid depends on: `MGCDLNet` now sets `self.attn_backend`
(`train.py` compiles the fused FlexAttention kernel off that attribute, and
without it `MGGroupCDL` silently ran uncompiled), and `physics/mask.py` gained a
`uniform_offset`-style `offset` knob -- `get_mask_cached` memoises one mask for
the whole run, which is right for a fixed uniform pattern and wrong for anything
meant to vary, so `mode="random"` and `offset="random"` now bypass the cache.

### The experiment grid

`scripts/make_mg_recon_configs.py` generates 20 cells --
5 models x {knee PD, brain T2} x {R=8, R=4}, both at 20 ACS lines, uniform mask,
`sigma ~ U[0, 0.05]`, full FOV at `batch_size=1`.

Network hyperparameters are matched to the Julia reference, so a deviation is
traceable: `Sljiva/config/groupcdl.yaml` supplies `M=169, p=7, s=2, d=1,
tau0=1e-3, Mh=64, gamma0=0.8, similarity=pidistance,
init_strategy=semi_orthogonal` and the `K=30` baseline;
`Sljiva/scripts/makeconfigs_multigrid.jl` supplies the multigrid sweep proper --
its `K888s2` cell (`K=[1,[8,8,8]]`, `s=2`, `widen=1`, `alpha0=0.1`,
`alpha_conv=false`, `dK=2`) crossed with `dual` in `{false,true}` and
`windowsize` in `{1,9}`. Those four corners **are** the four model types below.
`Sljiva/config/altsplit.yaml` supplies the LADMM cell.

| tag | type | K | group |
|---|---|---|---|
| `mgcdlnet` | `MGCDLNet` | `[1, [8, 8, 8]]` | `W=1` |
| `mggroupcdl` | `MGGroupCDL` | `[1, [8, 8, 8]]` | `W=9`, `Mh=64`, `dK=2` |
| `cdlnet` / `groupcdl` | same classes, plain int `K` | `30` | as above |
| `ladmm` | `AltSplitCDLNet` | `admm_iters=6`, `smap_update`, `[1,[4,4,8]]` prox | -- |

**`MGLPDS` / `MGGroupLPDS` are not in this grid.** ImMAP's `MGLPDS` is
`MGCDLNet(dual=True)`, the Fenchel/Moreau dual of the CDL prox. The reference's
multigrid MRI model called LPDS is `mg_lpdsnet` (`Sljiva/src/networks/mg_lpds.jl`),
a primal-dual splitting network with a sensenet, which is **not ported**. Same
name, different architecture -- so running the dual cells here would produce
numbers that read as an MGLPDS comparison without being one. The model types
still build and are still covered by `tests/test_integration.py`.

Two more things about this grid that are the reference's choices rather than
ours, and should not be read as controlled comparisons:

* `K=30` is **not** iteration-matched to the V-cycle (30 sweeps vs 8 on the fine
  grid) and none of the cells are parameter-matched -- a V-cycle also carries its
  coarse levels, the `dF` copies and `alpha`. `scripts/eval_mg_recon.py` reports
  `n_params` next to PSNR so the asymmetry stays visible.
* The group cells use the multigrid sweep's `W=9 / dK=2`, held identical across
  V-cycle and baseline cells, rather than `groupcdl.yaml`'s `W=35 / dK=5` (which
  belongs to its standalone denoising experiment). That keeps the
  multigrid-vs-baseline contrast from being confounded by the attention config.

**Attention backend.** This is the one place the grid cannot match the reference,
and it is worth understanding rather than working around. See the section below.

```bash
python scripts/make_mg_recon_configs.py --list-cells   # index -> cell, as the array job sees it
python scripts/make_mg_recon_configs.py                # write config/{knee,brain}/mg/*.json
sbatch slurm/mg_recon_grid.sbatch                      # all 28; --array=0 for one
python scripts/eval_mg_recon.py --runs trained_nets/mg_recon --out results/mg_recon.csv
```

Validation is pinned: `val_noise_std: 0.025` (the midpoint of the training range,
where `genobs(clo, Val{true}(), s)` evaluates), `val_seed: 1234`, and a single
fixed slice per validation volume. Val curves therefore move because the model
moved.
