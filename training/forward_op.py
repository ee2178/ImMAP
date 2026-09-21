"""
Forward-operator (E: CT1 -> T1, optionally conditioned on T2/FLAIR) training, as a train.py task.

    task: "forward_op",  model.type: "ForwardOp"

Same epoch / backtracking / checkpoint / resume machinery as train_synthesis, so an E is trained
exactly like the other nets. The ladder script (scripts/fit_forward_ladder.py) imports the
validation helpers below; both report the same metrics:

    rmse, rmse_enh, rmse_rest, rel_err   residual T1 - E(CT1, c) inside the brain mask
    gain                                 rmse(identity) / rmse
    enh_removed                          share of CT1 - T1 that E takes out on top-quantile pixels
    x_swap_rmse, x_swap_ratio            CT1 swapped for another slice's; ratio ~1 = E ignores CT1
plus psnr / ssim / nrmse of E(CT1, c) against T1 (masked, `data_range`), for parity with the
other tasks.

Batch: the nyumets_guided loader's 4-tuple (x0 = CT1, x1 = T1, cond = side information, mask).
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
# NO matplotlib.use() here: this module is imported (via training/__init__) by notebooks, and
# switching the backend at import silently turns their plt.show() into a no-op. Headless jobs
# fall back to Agg on their own.
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.common import save_ckpt, apply_loss_mask, POSTFIX_EVERY
from training.common import backtrack as do_backtrack, resync_schedule
from training.losses import LOSS_REGISTRY
from training.metrics import compute_metrics
from visualization.image import subplot_images


# ---------------------------------------------------------------------------------------------
def _inputs(E):
    if not E.cond_channels:
        return "x"
    return "x+c" if E.use_x else "c"


def masked_mse(a, b, m):
    return ((a - b) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def enh_region(x, y, m, q):
    """Per slice: brain pixels whose CT1 - T1 is in the top (1 - q) fraction."""
    d = x - y
    out = torch.zeros_like(m)
    for i in range(x.shape[0]):
        inb = m[i] > 0.5
        if inb.sum() < 10:
            continue
        thr = torch.quantile(d[i][inb].float(), q)
        out[i] = (inb & (d[i] >= thr) & (d[i] > 0)).float()
    return out


class ValStats:
    def __init__(self):
        self.s = dict(sse=0., n=0., sse_enh=0., n_enh=0., sse_rest=0., n_rest=0., ssy=0.,
                      proj=0., dd=0., sse_swap=0., n_swap=0.)

    def add_swap(self, y, e_swap, m):
        self.s["sse_swap"] += float(((y - e_swap) ** 2 * m).sum())
        self.s["n_swap"] += float(m.sum())

    def add(self, x, y, e, m, enh):
        r2 = (y - e) ** 2
        rest = m * (1 - enh)
        d = x - y
        s = self.s
        s["sse"] += float((r2 * m).sum());        s["n"] += float(m.sum())
        s["sse_enh"] += float((r2 * enh).sum());  s["n_enh"] += float(enh.sum())
        s["sse_rest"] += float((r2 * rest).sum()); s["n_rest"] += float(rest.sum())
        s["ssy"] += float((y ** 2 * m).sum())
        s["proj"] += float(((x - e) * d * enh).sum())
        s["dd"] += float((d ** 2 * enh).sum())

    def result(self):
        s = self.s
        nan = float("nan")
        return {
            "rmse": math.sqrt(s["sse"] / s["n"]) if s["n"] else nan,
            "rmse_enh": math.sqrt(s["sse_enh"] / s["n_enh"]) if s["n_enh"] else nan,
            "rmse_rest": math.sqrt(s["sse_rest"] / s["n_rest"]) if s["n_rest"] else nan,
            "rel_err": math.sqrt(s["sse"] / s["ssy"]) if s["ssy"] else nan,
            "enh_removed": s["proj"] / s["dd"] if s["dd"] else nan,
            "x_swap_rmse": math.sqrt(s["sse_swap"] / s["n_swap"]) if s["n_swap"] else nan,
        }


def batch_xyc(batch, device):
    x0, x1, cond, mask = batch[:4]
    # x = CT1 (the bridge unknown), y = T1 (the measurement), c = side information (cond_idx)
    to = lambda t: t.to(device, non_blocking=True)
    c = to(cond) if cond.shape[1] else None
    return to(x0), to(x1), c, to(mask)


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def validate(ops, loader, device, enh_q, panel_idx=0):
    for E in ops.values():
        E.eval()
    stats = {name: ValStats() for name in ["identity", *ops]}
    panel = None
    for bi, batch in enumerate(loader):
        x, y, c, m = batch_xyc(batch, device)
        enh = enh_region(x, y, m, enh_q)
        swap = x.shape[0] > 1                 # a final batch of one has no partner
        xs = x.roll(1, dims=0) if swap else None
        stats["identity"].add(x, y, x, m, enh)
        if swap:
            stats["identity"].add_swap(y, xs, m)
        outs = {}
        for name, E in ops.items():
            outs[name] = E(x, c)
            stats[name].add(x, y, outs[name], m, enh)
            if swap:
                stats[name].add_swap(y, E(xs, c), m)
        if bi == panel_idx:
            panel = (x[:1].cpu(), y[:1].cpu(), m[:1].cpu(), {k: v[:1].cpu() for k, v in outs.items()})
    for E in ops.values():
        E.train()
    res = {k: v.result() for k, v in stats.items()}
    base = res["identity"]["rmse"]
    for r in res.values():
        r["gain"] = base / r["rmse"] if r["rmse"] else float("nan")
        r["x_swap_ratio"] = r["x_swap_rmse"] / r["rmse"] if r["rmse"] else float("nan")
    return res, panel


def panel_figure(panel, step, res):
    """One val slice, one row per operator. Columns:

      1  image      gray, ONE window taken from T1 (the target) for every row -- E(CT1) should look
                    like the T1 row; the CT1 (identity) row saturates where it is enhanced
      2  left over  E(CT1) - T1: the error. White = perfect; red = E too bright (enhancement
                    not removed); blue = E too dark
      3  removed    CT1 - E(CT1): what E subtracted from CT1

    Columns 2 and 3 share ONE diverging colormap, sign and fixed window (99th pct of |CT1 - T1| in
    the brain), and columns 2 + 3 = CT1 - T1 on every row. So the identity row's column 2 is the
    total to remove, and a perfect operator has a white column 2 and a column 3 that matches it.
    """
    x, y, m, outs = panel
    rows = [[y, None, None], [x, x - y, x - x]]
    labels = ["T1 (target)", "identity: E(CT1)=CT1"]
    xlab = [[None, None, None],
            [f"rmse {res['identity']['rmse']:.3f}", None, None]]
    for name, e in outs.items():
        rows.append([e, e - y, x - e])
        labels.append(name)
        r = res[name]
        xlab.append([f"rmse {r['rmse']:.3f}  gain {r['gain']:.2f}", None,
                     f"removed {r['enh_removed']:.2f}"])
    inb = (x - y)[m > 0.5]
    v = float(torch.quantile(inb.abs().float(), 0.99)) if inb.numel() else 1.0
    fig, _ = subplot_images(
        rows, row_labels=labels, xlabels=xlab,
        col_titles=["image (T1 window)", "left over: E(CT1) - T1", "removed: CT1 - E(CT1)"],
        cmap=["gray", "RdBu_r", "RdBu_r"],
        vmin=[None, -v, -v], vmax=[None, v, v],
        window_from=[y], p=(1, 99), mask=m, apply_mask=True, magnitude=False,
        colorbar="each", panel_size=(3.0, 2.9),
        suptitle=f"CT1 -> T1 forward operators, val step {step}  "
                 f"(cols 2+3 = CT1 - T1; red = brighter)", show=False)
    return fig



# ---------------------------------------------------------------------------------------------
def _brain_frac(item):
    """Fraction of the frame the brain mask covers, from one dataset item (tuple or dict)."""
    mask = item["mask"] if isinstance(item, dict) else item[3]
    return float((torch.as_tensor(mask) > 0.5).float().mean())


def fixed_val_subset(val_loader, n, seed, min_brain_frac=0.0):
    """A fixed random subset of the val set, in RANDOM order (x_swap pairs batch neighbours, and
    sorted order would make them adjacent slices of one volume). n <= 0 or None: the whole set.

    `min_brain_frac` > 0 keeps only slices whose brain mask covers at least that fraction of the
    frame -- the end-of-volume slices (vertex, skull base) are mostly background, reconstruct
    identically under any method, and dilute every metric and figure. Candidates are drawn in a
    fixed random order and read one at a time until `n` pass, so the cost is ~n / acceptance item
    reads. With min_brain_frac = 0 the selection is EXACTLY the old one (same slices, same order),
    so switching this on changes a run's val set but leaving it off never does.
    """
    ds = val_loader.dataset
    rng = np.random.default_rng(seed)
    if not min_brain_frac or min_brain_frac <= 0:
        if not n or n <= 0 or n >= len(ds):
            idx = rng.permutation(len(ds))
        else:
            idx = rng.choice(len(ds), size=int(n), replace=False)
        idx = idx.tolist()
    else:
        want = len(ds) if (not n or n <= 0) else int(n)
        idx, fracs, scanned = [], [], 0
        for i in rng.permutation(len(ds)).tolist():
            f = _brain_frac(ds[i])
            scanned += 1
            fracs.append(f)
            if f >= min_brain_frac:
                idx.append(i)
                if len(idx) >= want:
                    break
        q = np.percentile(fracs, [10, 50, 90]) if fracs else [float("nan")] * 3
        print(f"[val subset] kept {len(idx)} of {scanned} scanned slices with brain frac >= "
              f"{min_brain_frac:g} ({100 * len(idx) / max(scanned, 1):.0f}% pass; scanned brain frac "
              f"p10/p50/p90 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f})")
        if len(idx) < want:
            print(f"[val subset] only {len(idx)} slices pass (wanted {want}); lower min_brain_frac?")
    return DataLoader(Subset(ds, idx), batch_size=val_loader.batch_size, shuffle=False,
                      num_workers=val_loader.num_workers, pin_memory=True)


def train_forward_op(
    net, opt, sched, device,
    train_loader,
    val_loader,
    wandb=None,
    start_epoch=0,
    num_epochs=1500,
    steps_per_epoch=200,
    val_every_epochs=25,
    clip_grad=1.0,
    backtrack_thresh=1000,
    backtrack_factor=0.9,
    backtrack_count=0,
    best_loss=float("inf"),
    loss_type="complex-mse",
    use_mask=True,
    psnr_only=False,
    data_range=2.0,
    val_slices=4000,                 # fixed random val subset; 0 / null = the whole val set
    val_seed=0,
    val_min_brain_frac=0.0,          # drop mostly-background slices from the val subset
    enh_quantile=0.98,
    save_dir=None,
    ckpt=None,                       # signature parity; resume handled in train.py
    save_ckpt_fn=save_ckpt,
):
    net.to(device)
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"unknown loss_type {loss_type!r}; expected one of {sorted(LOSS_REGISTRY)}")
    loss_fn = LOSS_REGISTRY[loss_type]

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "net.ckpt")
    val_sub = (fixed_val_subset(val_loader, val_slices, val_seed, val_min_brain_frac)
               if val_loader is not None else None)

    checked = False
    if backtrack_count:
        print(f"resuming at backtrack_count={backtrack_count}: LR -> "
              f"{resync_schedule(opt, sched, backtrack_count, backtrack_factor)}")

    train_iter = iter(train_loader)
    pbar = tqdm(total=num_epochs * steps_per_epoch, initial=start_epoch * steps_per_epoch,
                desc="FORWARD_OP", dynamic_ncols=True)

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss, n_batches = 0.0, 0
        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x, y, c, m = batch_xyc(batch, device)
            if not checked:
                got = 0 if c is None else c.shape[1]
                if net.cond_channels not in (0, got):
                    raise ValueError(f"model cond_channels={net.cond_channels} but the loader "
                                     f"supplies {got} cond channel(s) (data.*.cond_idx)")
                print(f"[forward_op] inputs={_inputs(net)}  x {tuple(x.shape)}  "
                      f"cond {None if c is None else tuple(c.shape)}")
                checked = True

            opt.zero_grad()
            pred = net(x, c)
            tgt_m, pred_m = apply_loss_mask(y, pred, m, use_mask)
            loss = loss_fn(pred_m, tgt_m, None)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            opt.step()
            if sched is not None and not isinstance(sched, ReduceLROnPlateau):
                sched.step()

            running_loss += loss.detach()
            n_batches += 1
            pbar.update(1)
            if n_batches % POSTFIX_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.3e}", epoch=epoch)

        global_step = (epoch + 1) * steps_per_epoch
        avg_loss = float(running_loss) / max(n_batches, 1)
        nonfinite = not math.isfinite(avg_loss)

        # ---- averaged-loss backtracking (as train_synthesis) ----
        if nonfinite or (avg_loss > best_loss + backtrack_thresh):
            reason = "non-finite loss" if nonfinite else (
                f"avg loss {avg_loss:.3e} > best {best_loss:.3e} + {backtrack_thresh}")
            print(f"[epoch {epoch}] {reason} -- backtracking")
            if os.path.exists(ckpt_path) and math.isfinite(best_loss):
                backtrack_count, new_lr = do_backtrack(
                    ckpt_path, model=net, optimizer=opt, scheduler=sched, device=device,
                    backtrack_count=backtrack_count, backtrack_factor=backtrack_factor)
                print(f"backtrack #{backtrack_count} -> LR {new_lr}")
            else:
                raise RuntimeError(f"Backtrack at epoch {epoch} but no valid checkpoint "
                                   f"(best_loss={best_loss}).")
        elif save_ckpt_fn and avg_loss < best_loss:
            save_ckpt_fn(ckpt_path, model=net, optimizer=opt, scheduler=sched, step=global_step,
                         backtrack_count=backtrack_count, best_loss=avg_loss)
            best_loss = avg_loss

        if not nonfinite:
            with torch.no_grad():
                tm = compute_metrics(tgt_m, pred_m.detach(), psnr_only=psnr_only,
                                     data_range=data_range, mask=m if use_mask else None)
            log = {"train/loss": avg_loss, "train/lr": opt.param_groups[0]["lr"],
                   "train/epoch": epoch, **{f"train/{k}": float(v) for k, v in tm.items()}}
            if wandb:
                wandb.log(log, step=global_step)
            else:
                print(log)

        # ---- validation ----
        if val_sub is not None and val_every_epochs and (epoch + 1) % val_every_epochs == 0:
            res, panel = validate({"E": net}, val_sub, device, enh_quantile)
            net.eval()
            agg, n_s = {}, 0
            with torch.no_grad():
                for batch in val_sub:
                    xv, yv, cv, mv = batch_xyc(batch, device)
                    tv, pv = apply_loss_mask(yv, net(xv, cv), mv, use_mask)
                    mets = compute_metrics(tv, pv, psnr_only=psnr_only, data_range=data_range,
                                           mask=mv if use_mask else None)
                    mets["loss"] = loss_fn(pv, tv, None)
                    bs = xv.shape[0]
                    for k, v in mets.items():
                        agg[k] = agg.get(k, 0.0) + float(v) * bs
                    n_s += bs
            net.train()
            val = {k: v / max(n_s, 1) for k, v in agg.items()}
            val.update(res["E"])
            val["identity_rmse"] = res["identity"]["rmse"]
            val["identity_x_swap_rmse"] = res["identity"]["x_swap_rmse"]

            if wandb:
                log = {f"val/{k}": v for k, v in val.items()}
                if panel is not None:
                    name = f"E ({_inputs(net)})"
                    p = (panel[0], panel[1], panel[2], {name: panel[3]["E"]})
                    fig = panel_figure(p, global_step, {"identity": res["identity"], name: res["E"]})
                    log["val/panel"] = wandb.Image(fig)
                    plt.close(fig)
                wandb.log(log, step=global_step)
            else:
                print(f"[VAL] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in val.items()))

            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val["loss"])

    pbar.close()
    return net
