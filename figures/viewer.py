#!/usr/bin/env python3
"""
Scrub the dumped reconstructions, mark the rows a figure should show, place their zooms.

    python figures/viewer.py configs/mg_brain_R8.py -n 3

Reads only what `scripts/dump_eval.py` wrote -- no torch, no checkpoints -- so
it starts instantly and runs on a laptop against an rsync'd copy of the
cluster's `eval_dump/` directories.

BROWSE -- scrub the set and mark the rows you want:
    left / right     slice -1 / +1              (shift: -5 / +5)
    up / down        previous / next volume
    e                cycle which column is displayed
    x                toggle the residual of that column (|ref - col|)
    d                toggle the inter-method disagreement overlay
    c                toggle the FOV crop
    - / =            dimmer / brighter          (session only; the value is
                                                 printed so you can pin it)
    space            mark or unmark this volume+slice
    m                jump to the next marked row
    enter            zoom THIS slice (marking it first if it is not marked)
    q                save and quit
    esc              abort: discard everything this session changed

ZOOM -- place each marked row's window:
    click / drag     centre the box at the cursor
    arrow keys       nudge one pixel
    n / p or m       next / previous marked row
    d                overlay          r  revert to the automatic placement
    b                back to BROWSE
    q                save and quit    esc  abort

The status line reports every column's stored per-slice metric, so you can see
which slices actually separate the methods before committing to one. Those are
the numbers `scripts/evaluate.py` reports, read back rather than recomputed --
a slice that looks decisive here is decisive in the table too.

Picks are written as you go, into two files so that redoing one never discards
the other:

    rows/<VARIANT>.json    which volume and slice each row shows
    zooms/<VARIANT>.json   where each row's zoom box sits

Only the columns named in the config's COLUMNS are read; the volume list is
their intersection, and the metric readout and disagreement overlay cover
exactly the ones that are a method's output.
"""

import argparse
import os
import sys
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figures import common as fc
from figures.common import (anatomy_mask, apply_cmap, auto_zoom, disagreement,
                            fov_box, load, nrmse, pretty, row_tag, to8, window)

FIT = 680       # target on-screen size of the image, in pixels


class Picker:
    def __init__(self, target):
        self.vols = fc.volumes()
        if not self.vols:
            raise SystemExit(
                "no volume is present in every column of COLUMNS.\n"
                "  Run scripts/dump_eval.py for each run with the SAME "
                "--volumes / --n-volumes.")
        self.target = target
        self.vi, self.pos, self.si = 0, 0, 0
        self.mode, self.dis, self.view, self.resid = "browse", False, 0, False
        self.crop = fc.CROP_FOV
        self.flash = ""

        # Picks are written as you go, so abort restores what was on disk at
        # startup rather than trying to buffer every edit.
        self._snap = {f: (open(f).read() if os.path.exists(f) else None)
                      for f in (fc.rows_file(), fc.zooms_file())}

        # Open on the first saved row, so browsing starts somewhere real. Only
        # a previous SELECTION is pre-marked -- seeding from the config's ROWS
        # would leave "pick N" already full before you touch anything.
        rows = fc.active_rows()
        self.sel = ([dict(r) for r in rows][:target]
                    if os.path.exists(fc.rows_file()) else [])
        if rows and rows[0]["volume"] in self.vols:
            self.vi = self.vols.index(rows[0]["volume"])
            self.pos = self._pos_of(rows[0]["volume"], rows[0]["slice"])

        for w in fc.check_comparable(self.vols[0]):
            print(f"[warning] {w}")

        self.root = tk.Tk()
        self.root.title(f"figure picker -- {fc.variant()}")
        self.canvas = tk.Canvas(self.root, highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.status = tk.Label(self.root, anchor="w", justify="left",
                               font=("Consolas", 10), padx=6, pady=4)
        self.status.pack(fill="x")

        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<B1-Motion>", self.click)
        b = self.root.bind
        b("<Left>",  lambda e: self.horiz(-1))
        b("<Right>", lambda e: self.horiz(1))
        b("<Shift-Left>",  lambda e: self.horiz(-5))
        b("<Shift-Right>", lambda e: self.horiz(5))
        b("<Up>",   lambda e: self.vert(-1))
        b("<Down>", lambda e: self.vert(1))
        b("<space>", lambda e: self.mark())
        b("<Return>", lambda e: self.to_zoom())
        b("<Key-b>", lambda e: self.to_browse())
        b("<Key-n>", lambda e: self.cycle(1))
        b("<Key-p>", lambda e: self.cycle(-1))
        b("<Key-d>", lambda e: self.set(dis=not self.dis))
        b("<Key-x>", lambda e: self.set(resid=not self.resid))
        b("<Key-c>", lambda e: self.set(crop=not self.crop))
        b("<Key-e>", lambda e: self.set(view=(self.view + 1) % len(fc.COLUMNS)))
        b("<Key-r>", lambda e: self.auto())
        b("<Key-m>", lambda e: self.cycle_marked())
        b("<Key-minus>", lambda e: self.gain(1 / 1.15))
        b("<Key-equal>", lambda e: self.gain(1.15))
        b("<Key-q>", lambda e: self.root.destroy())
        b("<Escape>", lambda e: self.abort())
        self.render()

    # --- data --------------------------------------------------------------
    def _pos_of(self, volume, sl):
        """Stack position of source slice `sl`, or 0 if this dump lacks it.

        Rows are saved with the SOURCE slice number, not a stack position, so a
        re-dump over a different `--slices` range still resolves them.
        """
        spec = next(s for _, s in fc.COLUMNS if fc.col_spec(s)[0] is not None)
        pos = load(spec, volume).at(sl)
        return 0 if pos is None else pos

    def current(self):
        """Everything needed to draw the current view.

        `cols` is every column, in config order, for display. `methods` is the
        subset that is actually a network's output -- the reference and the
        zero-filled adjoint are not methods, and letting them into the
        disagreement map would swamp it with the undersampling artifact.
        """
        vol = (self.vols[self.vi] if self.mode == "browse"
               else self.sel[self.si]["volume"])
        cols, methods, ref, stack_n = {}, {}, None, None
        for lab, spec in fc.COLUMNS:
            v = load(spec, vol)
            ref = v.ref if ref is None else ref
            stack_n = len(v) if stack_n is None else min(stack_n, len(v))
            cols[lab] = v
            if fc.col_spec(spec)[0] is not None:
                methods[lab] = v

        pos = (self.pos if self.mode == "browse"
               else self._pos_of(vol, self.sel[self.si]["slice"]))
        pos = int(np.clip(pos, 0, stack_n - 1))

        first = next(iter(cols.values()))
        sl = int(first.slice_index[pos])
        ref_s = first.ref[pos]
        organ = first.organ[pos] if first.organ is not None else anatomy_mask(ref_s)

        vmax = window(ref_s, organ)
        if self.crop:
            c0, r0, W = fov_box(ref_s)
            crop = lambda a: a[r0:r0 + W, c0:c0 + W]
        else:
            W, crop = min(ref_s.shape), (lambda a: a)

        return dict(
            vol=vol, pos=pos, sl=sl, W=W, vmax=vmax,
            ref=crop(ref_s), organ=crop(organ),
            cols={k: crop(v.img[pos]) for k, v in cols.items()},
            methods={k: crop(v.img[pos]) for k, v in methods.items()},
            stored={k: {m: vals[pos] for m, vals in v.metrics.items()}
                    for k, v in methods.items()},
            n=stack_n,
        )

    # --- navigation --------------------------------------------------------
    def horiz(self, d):
        if self.mode == "zoom":
            self.nudge(int(np.sign(d)), 0)
            return
        self.pos = int(np.clip(self.pos + d, 0, self.current()["n"] - 1))
        self.render()

    def vert(self, d):
        if self.mode == "zoom":
            self.nudge(0, int(np.sign(d)))
            return
        self.vi = (self.vi + d) % len(self.vols)
        self.pos = 0
        self.render()

    def cycle(self, d):
        if self.mode == "zoom" and self.sel:
            self.si = (self.si + d) % len(self.sel)
            self.render()

    def set(self, **kw):
        self.__dict__.update(kw)
        self.render()

    def gain(self, f):
        fc.BRIGHTNESS = round(fc.BRIGHTNESS * f, 3)
        self.flash = f"BRIGHTNESS = {fc.BRIGHTNESS}"
        print(f"BRIGHTNESS = {fc.BRIGHTNESS}")
        self.render()

    # --- selection ---------------------------------------------------------
    def mark(self):
        if self.mode != "browse":
            return
        c = self.current()
        row = dict(volume=c["vol"], slice=c["sl"])
        hit = next((r for r in self.sel if row_tag(r) == row_tag(row)), None)
        if hit:
            self.sel.remove(hit)
        elif len(self.sel) >= self.target:
            self.flash = f"already have {self.target} rows -- unmark one first"
            self.render()
            return
        else:
            self.sel.append(row)
        fc.save_rows(self.sel)
        self.render()

    def cycle_marked(self):
        if not self.sel:
            self.flash = "nothing marked yet"
            self.render()
            return
        if self.mode == "zoom":
            self.cycle(1)
            return
        tags = [row_tag(r) for r in self.sel]
        c = self.current()
        here = row_tag(dict(volume=c["vol"], slice=c["sl"]))
        nxt = self.sel[(tags.index(here) + 1) % len(tags)] if here in tags else self.sel[0]
        self.vi = self.vols.index(nxt["volume"])
        self.pos = self._pos_of(nxt["volume"], nxt["slice"])
        self.render()

    def abort(self):
        """Quit, putting rows/ and zooms/ back exactly as they were at startup."""
        for f, content in self._snap.items():
            if content is None:
                if os.path.exists(f):
                    os.remove(f)
            else:
                with open(f, "w") as fh:
                    fh.write(content)
        print("aborted -- rows/ and zooms/ restored to their state at startup")
        self.root.destroy()

    def to_zoom(self):
        """Zoom the row you are looking at -- not whichever row happens to be first."""
        c = self.current()
        here = row_tag(dict(volume=c["vol"], slice=c["sl"]))
        tags = [row_tag(r) for r in self.sel]
        if here not in tags:
            if len(self.sel) >= self.target:
                self.flash = (f"{self.target} rows already marked -- unmark one "
                              f"(space) to zoom this slice")
                self.render()
                return
            self.mark()
            tags = [row_tag(r) for r in self.sel]
        self.mode, self.si = "zoom", tags.index(here)
        self.render()

    def to_browse(self):
        if self.sel:
            row = self.sel[self.si % len(self.sel)]
            self.vi = self.vols.index(row["volume"])
            self.pos = self._pos_of(row["volume"], row["slice"])
        self.mode = "browse"
        self.render()

    # --- zoom --------------------------------------------------------------
    def zoom_of(self, c):
        z = fc.load_zooms().get(row_tag(self.sel[self.si]))
        return tuple(z) if z else auto_zoom(
            c["ref"], disagreement(c["ref"], c["methods"]), c["W"])

    def set_zoom(self, c0, r0, W):
        z = (int(np.clip(c0, 0, max(0, W - fc.ZOOM_W))),
             int(np.clip(r0, 0, max(0, W - fc.ZOOM_H))))
        zooms = fc.load_zooms()
        zooms[row_tag(self.sel[self.si])] = z
        fc.save_zooms(zooms)
        self.render()

    def click(self, ev):
        if self.mode != "zoom":
            return
        c = self.current()
        s = max(2, round(FIT / c["W"]))
        self.set_zoom(ev.x / s - fc.ZOOM_W / 2, ev.y / s - fc.ZOOM_H / 2, c["W"])

    def nudge(self, dc, dr):
        c = self.current()
        c0, r0 = self.zoom_of(c)
        self.set_zoom(c0 + dc, r0 + dr, c["W"])

    def auto(self):
        if self.mode != "zoom":
            return
        zooms = fc.load_zooms()
        zooms.pop(row_tag(self.sel[self.si]), None)   # falls back to automatic
        fc.save_zooms(zooms)
        self.render()

    # --- drawing -----------------------------------------------------------
    def render(self):
        if self.sel:                      # unmarking can leave si past the end
            self.si %= len(self.sel)
        c = self.current()
        s = max(2, round(FIT / c["W"]))

        labels = [lab for lab, _ in fc.COLUMNS]
        pick = labels[self.view % len(labels)]
        shown = c["cols"][pick]

        if self.resid:
            # The residual gets its own window: at the image's ceiling every
            # error map is black, which is exactly when you stop being able to
            # tell two methods apart.
            r = np.abs(c["ref"] - shown)
            a = np.clip(r / max(c["vmax"] / fc.RESID_GAIN, 1e-12), 0, 1)
            img = Image.fromarray(
                apply_cmap(a, fc.RESID_CMAP) if fc.RESID_CMAP
                else (a * 255 + 0.5).astype(np.uint8),
                "RGB" if fc.RESID_CMAP else "L").convert("RGB")
        else:
            img = Image.fromarray(to8(shown, c["vmax"]), "L").convert("RGB")

        if self.dis:
            dis = disagreement(c["ref"], c["methods"])
            heat = Image.fromarray(
                apply_cmap(dis / (dis.max() + 1e-12), fc.RESID_CMAP or "inferno"),
                "RGB")
            img = Image.blend(img, heat, 0.55)

        self.photo = ImageTk.PhotoImage(
            img.resize((c["W"] * s, c["W"] * s), Image.NEAREST))
        self.canvas.config(width=c["W"] * s, height=c["W"] * s)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        marked = any(row_tag(r) == row_tag(dict(volume=c["vol"], slice=c["sl"]))
                     for r in self.sel)
        if self.mode == "zoom":
            c0, r0 = self.zoom_of(c)
            self.canvas.create_rectangle(
                c0 * s, r0 * s, (c0 + fc.ZOOM_W) * s, (r0 + fc.ZOOM_H) * s,
                outline="#00dc00", width=2)
        elif marked:
            self.canvas.create_rectangle(
                1, 1, c["W"] * s - 1, c["W"] * s - 1, outline="#00dc00", width=3)

        tag = pick + (" residual" if self.resid else "")
        self.canvas.create_text(5, 4, anchor="nw", fill="yellow",
                                font=("Consolas", 11),
                                text=tag + (" + disagreement" if self.dis else ""))
        self.status.config(text=self._status(c, marked))

    def _status(self, c, marked):
        m = fc.STATUS_METRIC
        parts = []
        for lab, spec in fc.COLUMNS:
            if lab in c["stored"] and m in c["stored"][lab]:
                parts.append(f"{lab}={c['stored'][lab][m]:.3f}")
            elif fc.col_spec(spec)[1] != "reference":
                # No stored metric: a shared dataset such as the zero-filled
                # adjoint, which no run scored. Live NRMSE is labelled as such
                # so it is never read as one of the table's numbers.
                parts.append(f"{lab}~nrmse {nrmse(c['ref'], c['cols'][lab]):.3f}")

        head = (f"[{self.vi + 1}/{len(self.vols)}] {pretty(c['vol'])} "
                f"slice {c['sl']} ({c['pos'] + 1}/{c['n']})"
                f"{'  [MARKED]' if marked else ''}"
                if self.mode == "browse" else
                f"ZOOM [{self.si + 1}/{len(self.sel)}] {pretty(c['vol'])} "
                f"slice {c['sl']}")
        keys = ("<-/-> slice (shift 5) | up/dn volume | e column | x residual | "
                "d overlay | c crop | -/= gain | space mark | m next | "
                "enter zoom | q save+quit | esc abort"
                if self.mode == "browse" else
                "click/drag place | arrows nudge | n/p next row | r auto | "
                "b browse | q save+quit | esc abort")
        note = self.flash
        self.flash = ""
        return (f"{head}   selected {len(self.sel)}/{self.target}   {note}\n"
                f"{m}: {'  '.join(parts)}\n{keys}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="config file (see figures/configs/)")
    ap.add_argument("-n", "--rows", type=int, default=3, help="how many rows to pick")
    ap.add_argument("--name", help="override the variant name")
    ap.add_argument("--dump-root", help="override DUMP_ROOT")
    a = ap.parse_args()

    if a.config:
        fc.apply_config(a.config)
    if a.name:
        fc.NAME = fc.VARIANT = a.name
    if a.dump_root:
        fc.DUMP_ROOT = a.dump_root

    Picker(a.rows).root.mainloop()


if __name__ == "__main__":
    main()
