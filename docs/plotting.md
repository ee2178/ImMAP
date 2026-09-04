# Plotting cookbook

Everything lives in `visualization/`:

```python
from visualization import plot_image, subplot_images, plot_hist, subplot_hists, disp_window, to_numpy
```

Five functions cover what the notebooks were rewriting by hand.

| you want | call |
| --- | --- |
| one image | `plot_image(x, vmin=.., vmax=.., cmap=..)` |
| a row / grid of images | `subplot_images([...], titles=[...])` |
| a shared display window | `disp_window(x1, x0, mask=m, p=(1, 99))` |
| a tensor as 2-D float numpy | `to_numpy(x)` |
| one or several histograms | `plot_hist(data, mask=m, p=99.5)` |
| a row / grid of histograms | `subplot_hists([...])` |

`to_numpy` handles the `.detach().cpu()`, the complex magnitude, and the leading
batch/channel dims, so `(B, C, H, W)`, `(1, H, W)` and `(H, W)` all just work; pass
`index=` to pick a different slice.

---

## The window rule

`imshow` normalises to its own min/max. Two panels drawn that way are **not**
comparable: a reconstruction that overshoots by 20% gets rescaled into looking
correct, and a residual that grows over training looks constant.

So: panels that should be compared share one window, and a residual keeps its own
*fixed* one.

```python
vmin, vmax = disp_window(x1, x0, mask=mask, p=(1, 99))   # prior + GT define the scale
```

`p` is a percentile pair (`(1, 99)`), a scalar (`99` meaning `(1, 99)`), or `None`
for the full min/max. `mask=` restricts the statistics to the brain — without it the
background zeros own the low percentile. Inputs over 1M voxels are subsampled.

`subplot_images` does this for you: it pools every panel that was not given an
explicit `vmin`/`vmax` and applies one window to all of them.

---

## Images

```python
plot_image(x0, p=(1, 99), mask=mask, title="T1ce")

# a signed residual: symmetric window so bwr puts white at zero
plot_image(rec - x0, cmap="bwr", symmetric=True, p=99.5, colorbar=True)

# an ET outline over a masked panel, with the metric as a caption
plot_image(rec, mask=mask, apply_mask=True, overlay=et, xlabel=f"{psnr:.1f} dB")
```

`apply_mask=True` multiplies the displayed pixels by the mask; without it the mask
only affects the auto window (a mask should not silently change what you are looking
at). `plot_image` returns the axes and takes `ax=` to draw into an existing one.

### Grids

`images` can be a flat list (one row, or wrapped with `ncols=`), a list of rows, or a
batched tensor with `ndim >= 3` (split along the leading axis). `None` leaves a cell
blank.

```python
subplot_images(traj[:, 0], titles=[f"frame {j}" for j in range(len(traj))])
```

`cmap`, `vmin`, `vmax`, `p` and `overlay` each take one value for the whole grid **or**
a per-column list that repeats down the rows. That is the residual-column idiom:

```python
subplot_images(
    [[x1[i], cond[i, 0], rec[i], x0[i], (rec[i] - x0[i]).abs()] for i in range(3)],
    col_titles=["x1 (prior)", "FLAIR", "I2SB recon", "x0 = GT", "|GT - recon|"],
    window_from=[x1[:3], x0[:3]], mask=mask[:3], p=(1, 99),      # gray panels share this
    cmap=[  "gray", "gray", "gray", "gray", "magma"],
    vmin=[    None,   None,   None,   None,     0.0],            # residual keeps its own
    vmax=[    None,   None,   None,   None,    VMAX],            # FIXED scale
    xlabels=[[None, None, f"{psnr[i]:.1f} dB", None, None] for i in range(3)],
    overlay=[None, None, et[0], et[0], None],
    colorbar=True, cbar_label="residual",
    suptitle=f"I2SB  nfe={NFE}",
)
```

Other options worth knowing: `col_titles` labels the first row only, `row_labels`
labels the left edge, `window_from=` computes the shared window from images that are
not themselves panels, `colorbar="each"` gives every panel its own, and
`panel_size=(w, h)` sizes the figure per panel instead of `figsize`.

---

## Histograms

`plot_hist` takes one series, `{label: values}`, a list of `(label, values)` pairs, or
a plain list of arrays. Several series get **shared bin edges** from the pooled data
(otherwise the overlay means nothing), `density=True`, and `histtype="step"`.

```python
plot_hist(x0, mask=mask, p=99.5, title="T1ce, within brain")

plot_hist({"T1": t1, "T1ce": t1ce}, mask=brain, p=(0.5, 99.5), log=True,
          xlabel="normalized intensity")

# a signed difference: range symmetric about zero, marker at 0
plot_hist({"healthy": d[brain & ~et], "ET": d[et]},
          symmetric=True, p=99.5, vlines=0.0, log=True,
          xlabel=r"$\Delta$ = T1ce - T1")

# markers with labels
plot_hist(raw, mask=brain, p=99.9, vlines={"chosen SCALE": SCALE, "p99": p99})
```

Key arguments: `mask=` (brain voxels only), `p=` / `range=` (clip the *range*, not the
data — raw BraTS tails run to 10x the tissue range and a full-range histogram is one
bar), `symmetric=`, `log=` (log-y, worth it whenever a tail matters), `density=`,
`vlines=` (a scalar, a list, `(x, label)` pairs, or `{label: x}`), and
`return_values=True` to get back the pooled arrays that were plotted.

`subplot_hists` lays out several panels; `share_bins=True` makes them comparable, and
any keyword given as a list matching the panel count is spread across the panels.

```python
subplot_hists({nm: img[..., c] for c, nm in enumerate(CONTRASTS)},
              mask=brain, p=99.5, ncols=4, suptitle="within-brain distributions")
```


## Inspecting a set of runs: the dump + viewer

`visualization/` is for one array at a time, in a notebook. Comparing several
trained runs over many slices is a different job, and rebuilding every network
in process to do it (as `notebooks/compare_mg_recon.py` does) is far too slow to
scrub through. That path is two steps:

```bash
# once per run, on the cluster: forward pass -> per-volume HDF5
python scripts/dump_eval.py --runs trained_nets/mg_recon/brain \
    --only lpdsnet_R8 mglpds_R8 mggrouplpds_R8 --n-volumes 8 --slices 0:16
```

```bash
# locally, against an rsync'd copy: scrub, mark rows, place zoom windows
python figures/viewer.py figures/configs/mg_brain_R8.py -n 3
```

The dump reuses the adapter in `evaluation/tasks.py`, so its images and
`results/*.csv`'s numbers come from one forward pass and cannot disagree. The
viewer reads HDF5 + numpy + PIL only — no torch, no checkpoints — so it starts
instantly and needs no GPU. Picks land in `figures/rows/<variant>.json` and
`figures/zooms/<variant>.json`, kept apart so re-picking one never discards the
other. See the module docstrings for the key bindings and the file contract.
