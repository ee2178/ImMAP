# Archived notebooks

Superseded by `notebooks/compare_i2sb.ipynb`, which takes a list of trained configs, samples one
shared validation batch through every net, and shows the results side by side split into slices
with and without tumor. Nothing here is deleted -- move a file back up a level if you want it.

| notebook | why it moved |
|---|---|
| `i2sb_sample.ipynb` | single-run sampling; `compare_i2sb` does the same thing for N runs |
| `compare_methods.ipynb` | hard-coded UNet / image-I2SB / latent-I2SB comparison |
| `compare_i2sb_sweep.ipynb` | swept-regressor comparison, **plus** diagnostics that did NOT carry over -- see below |
| `latent_i2sb_sample.ipynb` | latent bridge (`sb/latent_i2sb.py`), no longer in use |
| `inspect_latent_tau.ipynb` | latent-space `tau` selection, same |

## The one thing that did not carry over

`compare_i2sb_sweep.ipynb` ran three diagnostics per run, not one:

1. full reverse reconstruction (this is what `compare_i2sb` kept),
2. one-shot prediction with `x_t = x_1`, i.e. end-to-end synthesis with no bridge, and
3. teacher-forced per-step predictions across a grid of bridge positions.

(2) and (3) answer "does the sampling loop actually earn its cost, and where along the bridge is
the regressor weak" -- questions a val-batch picture cannot. Pull this notebook back out if you
need them again; its trajectory indexing is subtle (`reverse_sample` returns newest-first and logs
a prediction made at step `n` keyed on `n_prev`), so it is worth reusing rather than rewriting.

## Not archived, still current

* `inspect_beta_max.ipynb` -- sizing the bridge schedule from data
* `inspect_percentile_norm.ipynb` -- normalization schemes
* `inspect_i2sb_tau.ipynb` -- image-domain `tau` selection
