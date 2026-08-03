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
`tests.test_integration`, `tests.test_init`.
