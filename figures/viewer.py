#!/usr/bin/env python3
"""
Scrub the dumped reconstructions, mark the rows a figure should show, place their zooms.

    python figures/viewer.py configs/mg_brain_R8.py -n 3
    python figures/viewer.py configs/main_brain_R12.py --volume 2000045 --slice 6

Reads only what `scripts/dump_eval.py` wrote -- no torch, no checkpoints -- so
it starts instantly and runs on a laptop against an rsync'd copy of the
cluster's `eval_dump/` directories.

To run it ON THE CLUSTER and drive it with the mouse from a laptop browser
(over an SSH tunnel, no copying dumps down), use figures/webviewer.py: same
core (`PickerCore` below), same picks files.

BROWSE -- scrub the set and mark the rows you want:
    left / right     slice -1 / +1              (shift: -5 / +5)
    up / down        previous / next volume
    o                open a volume by name: type any part of it (e.g. 2000045),
                     optionally followed by a slice number (2000045 6), then
                     enter -- or pick from the filtered list
    g                every column side by side (the default) / one at a time
    1 .. 9           switch column 1..9 on/off in the side-by-side view
    s                snapshot: the current view as a PNG, figures/out/snapshots/
    e                one at a time: cycle which column is displayed
    x                toggle the residual of each column (|ref - col|)
    d                toggle the inter-method disagreement overlay
    c                toggle the FOV crop
    k                toggle masking every image to the organ (default: on when
                     the dumps' metrics are masked -- MASK_DISPLAY in common.py)
    - / =            dimmer / brighter          (session only; the value is
                                                 printed so you can pin it)
    space            mark or unmark this volume+slice
    m                jump to the next marked row
    enter            zoom THIS slice (marking it first if it is not marked)
    q                save and quit
    esc              abort: discard everything this session changed

ZOOM -- place each marked row's window. Under every panel sits the strip the
figure prints there: the magnified box (left) and its |residual| (right), from
the same code as figures/make_figure.py, so what you place is what you get.
    click / drag     centre the box at the cursor -- in ANY panel's image (not
                     its strip); every panel shows the same box, since a
                     figure row crops one region
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
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figures import common as fc
from figures.common import (anatomy_mask, apply_cmap, auto_zoom, disagreement,
                            fov_box, load, nrmse, pretty, row_tag, to8, window)

FIT = 680       # on-screen size of one panel (the most a grid panel gets), pixels
GAP = 8         # pixels between panels in the grid
GRID_COLS = 3   # panels per row in the grid: six columns -> 3 x 2
STRIP_GAP = 3   # pixels between a panel and its zoom strip (ZOOM mode)


class PickerCore:
    """Everything the viewer DOES -- data, navigation, marking, zoom placement,
    panel images, the status text -- with no window.

    Each front end subclasses it and overrides `render()`, which every state
    change calls: `Picker` below draws a Tk canvas, figures/webviewer.py
    serves the same panels to a browser. One core means the two cannot
    drift: the same keys of state, the same picks files, the same images.
    """
    def __init__(self, target, volume=None, sl=None):
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
        # None = follow fc.MASK_DISPLAY ("auto": masked when the dumps' metrics
        # are); `k` pins it on or off for the session.
        self.mask_view = None
        self.flash = ""
        # Every column side by side (the default), or one at a time (`g`).
        self.grid = True
        # Columns switched OFF in the side-by-side view (by label). The figure
        # is unaffected -- that is the config's COLUMNS -- this only declutters.
        self.hidden = set()
        self._spec = dict(fc.COLUMNS)
        self._geom = None             # the last render's layout, for clicks

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

        # --volume / --slice: start there instead.
        if volume is not None:
            hit = self.find_volume(volume)
            if hit is None:
                raise SystemExit(f"--volume {volume!r} matches "
                                 f"{self._matches(volume)[:5] or 'no volume'} -- "
                                 f"give enough of the name to pick ONE")
            self.vi = hit
            self.pos = self._pos_of(self.vols[hit], sl) if sl is not None else 0

        for w in fc.check_comparable(self.vols[0]):
            print(f"[warning] {w}")

    def render(self):
        """Redraw after a state change. A front end overrides this."""

    # --- display-independent helpers -------------------------------------
    def shown_labels(self):
        """The columns on screen: every one not switched off (grid), or the one
        in view (a column picked on its own is shown even if switched off)."""
        labels = [lab for lab, _ in fc.COLUMNS]
        if not self.grid:
            return [labels[self.view % len(labels)]]
        return [lab for lab in labels if lab not in self.hidden] or labels

    def visible_methods(self, c):
        """The methods not switched off -- what the disagreement overlay and the
        automatic zoom placement compare, so a hidden net does not steer them."""
        vis = {k: v for k, v in c["methods"].items() if k not in self.hidden}
        return vis or c["methods"]

    def toggle_column(self, lab):
        """Switch a column on/off; the last visible one cannot be switched off."""
        labels = [l for l, _ in fc.COLUMNS]
        if lab not in labels:
            return
        if lab in self.hidden:
            self.hidden.discard(lab)
        elif len([l for l in labels if l not in self.hidden]) > 1:
            self.hidden.add(lab)
        else:
            self.flash = "at least one column has to stay on"
        self.render()

    def heat(self, c):
        """The disagreement overlay as an RGB image, or None when it is off."""
        if not self.dis:
            return None
        dis = disagreement(c["ref"], self.visible_methods(c))
        return Image.fromarray(
            apply_cmap(dis / (dis.max() + 1e-12), fc.RESID_CMAP or "inferno"),
            "RGB")

    def panel_text(self, c, lab, i):
        """The label drawn on panel `i` (column `lab`)."""
        m = fc.STATUS_METRIC
        text = lab
        if self.resid and fc.col_spec(self._spec[lab])[1] != "reference":
            text += " |ref - x|"
        if lab in c["stored"] and m in c["stored"][lab]:
            text += f"   {m} {c['stored'][lab][m]:.2f}"
        if self.dis and i == 0:
            text += "   + disagreement"
        return text

    def is_marked(self, c):
        return any(row_tag(r) == row_tag(dict(volume=c["vol"], slice=c["sl"]))
                   for r in self.sel)

    def unmark(self, i):
        """Drop marked row `i` (and keep its zoom, as unmarking by space does)."""
        if 0 <= i < len(self.sel):
            self.sel.pop(i)
            fc.save_rows(self.sel)
            if self.mode == "zoom" and not self.sel:
                self.mode = "browse"
        self.render()

    # --- data --------------------------------------------------------------
    def _pos_of(self, volume, sl):
        """Stack position of source slice `sl`, or 0 if this dump lacks it."""
        pos = fc.pos_of(volume, sl)
        return 0 if pos is None else pos

    def current(self):
        """Everything needed to draw the current view -- `figures.common.slice_view`,
        the same function the figure builder uses."""
        vol = (self.vols[self.vi] if self.mode == "browse"
               else self.sel[self.si]["volume"])
        pos = (self.pos if self.mode == "browse"
               else self._pos_of(vol, self.sel[self.si]["slice"]))
        return fc.slice_view(vol, pos, crop=self.crop, mask=self.mask_view)

    def toggle_mask(self):
        """`k`: flip whether the images are drawn masked to the organ."""
        self.mask_view = not self.current()["display_masked"]
        self.flash = ("images masked to the organ" if self.mask_view
                      else "images unmasked (full field of view)")
        self.render()

    def snapshot(self, c=None):
        """The current view as one image: every panel on screen in a row, its
        label, and -- while placing boxes -- the box and the figure's strip under
        each panel. For notes and slides; the manuscript figure is
        figures/make_figure.py."""
        from PIL import ImageDraw, ImageFont
        c = c or self.current()
        shown = self.shown_labels()
        heat = self.heat(c)
        zoom = self.zoom_of(c) if self.mode == "zoom" else None
        k = max(1, int(np.ceil(400 / c["W"])))            # panels at least ~400 px
        P, gap = c["W"] * k, 6
        size = max(14, P // 22)                            # readable at any panel size
        try:                                               # matplotlib ships DejaVu
            from matplotlib import font_manager
            font = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), size)
        except Exception:                                  # noqa: BLE001
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:                              # Pillow < 10.1
                font = ImageFont.load_default()
        head = size + 14
        Hs = int(round(P * fc.ZOOM_H / (2 * fc.ZOOM_W + 1))) if zoom else 0
        H = P + (gap + Hs if zoom else 0)
        img = Image.new("RGB", (len(shown) * P + (len(shown) - 1) * gap, head + H),
                        (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((4, 6), f"{fc.variant()}   {fc.pretty(c['vol'])}   slice {c['sl']}",
               fill=(0, 0, 0), font=font)
        for i, lab in enumerate(shown):
            x = i * (P + gap)
            img.paste(self._panel(c, lab, heat).resize((P, P), Image.NEAREST), (x, head))
            t = self.panel_text(c, lab, i)
            bb = d.textbbox((x + 5, head + 4), t, font=font)
            d.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=(0, 0, 0))
            d.text((x + 5, head + 4), t, fill=(255, 216, 74), font=font)
            if zoom is not None:
                c0, r0 = zoom
                d.rectangle((x + c0 * k, head + r0 * k, x + (c0 + fc.ZOOM_W) * k - 1,
                             head + (r0 + fc.ZOOM_H) * k - 1), outline=(0, 220, 0), width=2)
                img.paste(self.strip(c, lab, zoom).resize((P, Hs), Image.NEAREST),
                          (x, head + P + gap))
        return img

    def snapshot_name(self, c=None):
        c = c or self.current()
        return (f"{fc.variant()}_{fc.pretty(c['vol'])}_s{c['sl']}"
                + ("_zoom" if self.mode == "zoom" else "") + ".png")

    def strip(self, c, lab, zoom):
        """What the figure puts UNDER column `lab`'s panel: the magnified box
        (left) and its |residual| (right), side by side with a 1-pixel gap, at
        source resolution. The front end scales it to the panel's width."""
        left, right = fc.zoom_pair(c, lab, zoom)
        gap = np.full((left.shape[0], 1, 3), 255, np.uint8)
        return Image.fromarray(np.concatenate([left, gap, right], axis=1), "RGB")

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

    def _matches(self, query):
        q = query.strip().lower()
        return [v for v in self.vols if q in v.lower()]

    def find_volume(self, query):
        """Index of the ONE volume `query` names: an exact name, or a substring
        only one volume contains. None if it names none or several."""
        q = query.strip().lower()
        exact = [i for i, v in enumerate(self.vols) if v.lower() in (q, "file_" + q)]
        if exact:
            return exact[0]
        hits = [i for i, v in enumerate(self.vols) if q in v.lower()]
        return hits[0] if len(hits) == 1 else None

    def goto(self, vi, sl=None):
        """Show volume `vi` at SOURCE slice `sl` (the number the status line and
        the scan report), or its first slice."""
        self.vi = vi
        self.pos = self._pos_of(self.vols[vi], sl) if sl is not None else 0
        if sl is not None and self.current()["sl"] != sl:
            self.flash = f"(slice {sl} is not in this dump -- showing its first)"
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

    def restore(self):
        """Put rows/ and zooms/ back exactly as they were at startup."""
        for f, content in self._snap.items():
            if content is None:
                if os.path.exists(f):
                    os.remove(f)
            else:
                with open(f, "w") as fh:
                    fh.write(content)
        print("aborted -- rows/ and zooms/ restored to their state at startup")

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
            c["ref"], disagreement(c["ref"], self.visible_methods(c)), c["W"])

    def set_zoom(self, c0, r0, W):
        z = (int(np.clip(c0, 0, max(0, W - fc.ZOOM_W))),
             int(np.clip(r0, 0, max(0, W - fc.ZOOM_H))))
        zooms = fc.load_zooms()
        zooms[row_tag(self.sel[self.si])] = z
        fc.save_zooms(zooms)
        self.render()

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
    def _panel(self, c, lab, heat):
        """One column's image: the reconstruction, or its residual (`x`), with
        the disagreement overlay (`d`) on top. The reference has no residual of
        its own, so it stays the image in residual mode."""
        shown = c["cols"][lab]
        is_ref = fc.col_spec(self._spec[lab])[1] == "reference"
        if self.resid and not is_ref:
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
        if heat is not None:
            img = Image.blend(img, heat, 0.55)
        return img

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
        keys = ("<-/-> slice (shift 5) | up/dn volume | o open | g all/one | e column | "
                "1-9 column on/off | x residual | d overlay | c crop | k mask | -/= gain | "
                "s snapshot | space mark | m next | enter zoom | q save+quit | esc abort"
                if self.mode == "browse" else
                "click/drag place (any panel) | arrows nudge | n/p next row | r auto | "
                "b browse | q save+quit | esc abort")
        note = self.flash
        self.flash = ""
        head += "   [masked to organ]" if c.get("display_masked") else "   [unmasked]"
        return (f"{head}   selected {len(self.sel)}/{self.target}   {note}\n"
                f"{m}: {'  '.join(parts)}\n{keys}")


class Picker(PickerCore):
    """The Tk window: keys, a canvas, the volume chooser."""

    def __init__(self, target, volume=None, sl=None):
        global tk, ImageTk
        # Imported here, not at module level, so the core (and the web
        # viewer) runs where tkinter is not installed -- a cluster node.
        import tkinter as tk
        from PIL import ImageTk
        super().__init__(target, volume=volume, sl=sl)
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
        b("<Key-o>", lambda e: self.chooser())
        b("<Up>",   lambda e: self.vert(-1))
        b("<Down>", lambda e: self.vert(1))
        b("<space>", lambda e: self.mark())
        b("<Return>", lambda e: self.to_zoom())
        b("<Key-b>", lambda e: self.to_browse())
        b("<Key-n>", lambda e: self.cycle(1))
        b("<Key-p>", lambda e: self.cycle(-1))
        b("<Key-d>", lambda e: self.set(dis=not self.dis))
        b("<Key-x>", lambda e: self.set(resid=not self.resid))
        b("<Key-s>", lambda e: self.save_snapshot())
        for i in range(1, 10):          # 1..9: switch column i on/off (grid view)
            b(f"<Key-{i}>", lambda e, i=i: self.toggle_column(
                ([l for l, _ in fc.COLUMNS] + [None] * 9)[i - 1]))
        b("<Key-c>", lambda e: self.set(crop=not self.crop))
        b("<Key-k>", lambda e: self.toggle_mask())
        b("<Key-g>", lambda e: self.set(grid=not self.grid))
        # In the grid `e` drops to one column at a time; there it cycles.
        b("<Key-e>", lambda e: (self.set(grid=False) if self.grid else
                                self.set(view=(self.view + 1) % len(fc.COLUMNS))))
        b("<Key-r>", lambda e: self.auto())
        b("<Key-m>", lambda e: self.cycle_marked())
        b("<Key-minus>", lambda e: self.gain(1 / 1.15))
        b("<Key-equal>", lambda e: self.gain(1.15))
        b("<Key-q>", lambda e: self.root.destroy())
        b("<Escape>", lambda e: self.abort())
        self.render()

    def abort(self):
        """Quit, putting rows/ and zooms/ back exactly as they were at startup."""
        self.restore()
        self.root.destroy()

    def chooser(self):
        """`o`: pick a volume from a list that filters as you type."""
        if self.mode != "browse":
            return
        top = tk.Toplevel(self.root)
        top.title("open volume")
        top.transient(self.root)
        tk.Label(top, anchor="w", justify="left", font=("Consolas", 9),
                 text="type any part of the name, optionally then a slice "
                      "(2000045 6)\nenter opens the highlighted one, esc cancels"
                 ).pack(fill="x", padx=6, pady=(6, 2))
        entry = tk.Entry(top, font=("Consolas", 11), width=44)
        entry.pack(fill="x", padx=6)
        lb = tk.Listbox(top, font=("Consolas", 10), height=min(20, len(self.vols)),
                        width=44, activestyle="dotbox", exportselection=False)
        lb.pack(fill="both", expand=True, padx=6, pady=6)
        shown = []

        def parse():
            parts = entry.get().split()
            sl = None
            if len(parts) > 1 and parts[-1].lstrip("-").isdigit():
                sl = int(parts.pop())
            return " ".join(parts), sl

        def refresh(*_):
            q, _sl = parse()
            shown[:] = [i for i, v in enumerate(self.vols) if q.lower() in v.lower()]
            lb.delete(0, "end")
            for i in shown:
                lb.insert("end", f"{i + 1:>4}  {pretty(self.vols[i])}")
            if shown:
                cur = shown.index(self.vi) if self.vi in shown else 0
                lb.selection_set(cur)
                lb.see(cur)

        def move(d):
            if not shown:
                return "break"
            cur = (lb.curselection() or (0,))[0]
            cur = max(0, min(len(shown) - 1, cur + d))
            lb.selection_clear(0, "end")
            lb.selection_set(cur)
            lb.see(cur)
            return "break"

        def accept(*_):
            if not shown:
                return
            _q, sl = parse()
            i = shown[(lb.curselection() or (0,))[0]]
            top.destroy()
            self.goto(i, sl)

        entry.bind("<KeyRelease>", lambda e: None if e.keysym in
                   ("Up", "Down", "Return", "Escape") else refresh())
        entry.bind("<Up>", lambda e: move(-1))
        entry.bind("<Down>", lambda e: move(1))
        top.bind("<Return>", accept)
        lb.bind("<Double-Button-1>", accept)
        top.bind("<Escape>", lambda e: top.destroy())
        refresh()
        entry.focus_set()
        top.grab_set()

    def save_snapshot(self):
        """`s`: the current view as a PNG under figures/out/snapshots/."""
        c = self.current()
        path = os.path.join(fc.HERE, "out", "snapshots", self.snapshot_name(c))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.snapshot(c).save(path)
        self.flash = f"snapshot -> {os.path.relpath(path, fc.ROOT)}"
        print(self.flash)
        self.render()

    def click(self, ev):
        """Centre the zoom box on the cursor, in image pixels of the panel the
        cursor is over. A click in the gap between panels does nothing."""
        if self.mode != "zoom" or self._geom is None:
            return
        g = self._geom
        col, row = int(ev.x // (g["P"] + GAP)), int(ev.y // (g["H"] + GAP))
        if col >= g["per_row"] or row * g["per_row"] + col >= g["n"]:
            return
        lx, ly = ev.x - col * (g["P"] + GAP), ev.y - row * (g["H"] + GAP)
        if not (0 <= lx < g["P"] and 0 <= ly < g["P"]):   # the strip is not a target
            return
        c = self.current()
        self.set_zoom(lx / g["s"] - fc.ZOOM_W / 2, ly / g["s"] - fc.ZOOM_H / 2, c["W"])

    def _layout(self, n, extra=0.0):
        """`(panel size in screen pixels, panels per row)` for n panels.

        The size does NOT depend on the image: the FOV crop is fitted to each
        slice's anatomy, so its width changes as you scroll, and a panel sized
        from it grew and shrank with every slice. Instead every crop is
        resampled into the same square -- FIT for one column, and for the grid
        (GRID_COLS to a row, 3 x 2 for six columns) as large as fits on the
        screen, capped at FIT.
        """
        if n == 1:
            return FIT, 1
        per_row = min(GRID_COLS, n)
        rows = math.ceil(n / per_row)
        sw = self.root.winfo_screenwidth() - 40
        sh = self.root.winfo_screenheight() - 190     # title bar + status lines
        # `extra`: the strip under each panel in ZOOM, as a fraction of P
        P = min((sw - (per_row - 1) * GAP) // per_row,
                int((sh - (rows - 1) * GAP) / rows / (1.0 + extra)), FIT)
        return max(1, int(P)), per_row

    def _label(self, x, y, text):
        """Yellow text on a black backing, so it reads over any image."""
        t = self.canvas.create_text(x + 5, y + 4, anchor="nw", fill="yellow",
                                    font=("Consolas", 10), text=text)
        bb = self.canvas.bbox(t)
        back = self.canvas.create_rectangle(bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1,
                                            fill="black", outline="")
        self.canvas.tag_lower(back, t)

    def render(self):
        if self.sel:                      # unmarking can leave si past the end
            self.si %= len(self.sel)
        c = self.current()

        shown = self.shown_labels()
        heat = self.heat(c)

        zoom = self.zoom_of(c) if self.mode == "zoom" else None
        # In ZOOM each panel carries the figure's strip beneath it -- the
        # magnified box and its residual -- so the box is placed by what it
        # will look like in the figure, not just where it sits.
        ratio = fc.ZOOM_H / (2 * fc.ZOOM_W + 1)
        P, per_row = self._layout(len(shown),
                                  extra=(ratio + STRIP_GAP / FIT) if zoom else 0.0)
        Hs = int(round(P * ratio)) if zoom is not None else 0
        H = P + (STRIP_GAP + Hs if zoom is not None else 0)
        rows = math.ceil(len(shown) / per_row)
        self.canvas.config(width=per_row * P + (per_row - 1) * GAP,
                           height=rows * H + (rows - 1) * GAP)
        self.canvas.delete("all")
        self._geom = dict(s=P / c["W"], P=P, H=H, per_row=per_row, n=len(shown))

        marked = any(row_tag(r) == row_tag(dict(volume=c["vol"], slice=c["sl"]))
                     for r in self.sel)
        m = fc.STATUS_METRIC
        # A whole-number magnification keeps hard pixels; any other ratio is
        # resampled smoothly, since an uneven nearest-neighbour stretch reads
        # as structure on a reconstruction.
        resample = Image.NEAREST if P % c["W"] == 0 else Image.LANCZOS
        self.photos = []                  # Tk drops an image nobody references
        for i, lab in enumerate(shown):
            x0, y0 = (i % per_row) * (P + GAP), (i // per_row) * (H + GAP)
            ph = ImageTk.PhotoImage(self._panel(c, lab, heat).resize((P, P), resample))
            self.photos.append(ph)
            self.canvas.create_image(x0, y0, anchor="nw", image=ph)
            if zoom is not None:
                st = ImageTk.PhotoImage(
                    self.strip(c, lab, zoom).resize((P, Hs), Image.NEAREST))
                self.photos.append(st)
                self.canvas.create_image(x0, y0 + P + STRIP_GAP, anchor="nw", image=st)
                self.canvas.create_rectangle(x0, y0 + P + STRIP_GAP, x0 + P - 1,
                                             y0 + P + STRIP_GAP + Hs - 1,
                                             outline="#00dc00", width=1)
            k = P / c["W"]
            if zoom is not None:
                c0, r0 = zoom
                self.canvas.create_rectangle(
                    x0 + c0 * k, y0 + r0 * k,
                    x0 + (c0 + fc.ZOOM_W) * k, y0 + (r0 + fc.ZOOM_H) * k,
                    outline="#00dc00", width=2)
            elif marked:
                self.canvas.create_rectangle(x0 + 1, y0 + 1, x0 + P - 1, y0 + P - 1,
                                             outline="#00dc00", width=3)
            text = lab
            if self.resid and fc.col_spec(self._spec[lab])[1] != "reference":
                text += " |ref - x|"
            if lab in c["stored"] and m in c["stored"][lab]:
                text += f"   {m} {c['stored'][lab][m]:.2f}"
            if self.dis and i == 0:
                text += "   + disagreement"
            self._label(x0, y0, text)
        self.status.config(text=self._status(c, marked))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="config file (see figures/configs/)")
    ap.add_argument("-n", "--rows", type=int, default=3, help="how many rows to pick")
    ap.add_argument("--name", help="override the variant name")
    ap.add_argument("--dump-root", help="override DUMP_ROOT")
    ap.add_argument("--volume", help="start at this volume (any unique part of its "
                                     "name, e.g. 2000045)")
    ap.add_argument("--slice", type=int, help="...and this source slice number")
    a = ap.parse_args()

    if a.config:
        fc.apply_config(a.config)
    if a.name:
        fc.NAME = fc.VARIANT = a.name
    if a.dump_root:
        fc.DUMP_ROOT = a.dump_root

    Picker(a.rows, volume=a.volume, sl=a.slice).root.mainloop()


if __name__ == "__main__":
    main()
