#!/usr/bin/env python3
"""
The figure picker in a web browser: run it on the cluster, use it on your laptop.

ON THE CLUSTER (a login node is fine -- it only reads HDF5 and draws PNGs):

    python -m figures.webviewer                       # every dataset, from a menu
    python -m figures.webviewer figures/configs/main_knee_R8.py --port 8770 -n 3

Every config under figures/configs/ is on the page's DATASET menu (greyed out
until its runs are dumped), so one viewer reaches all of them; switching
reloads that config's options, columns and its own picks. A config given on the
command line is only where it opens.

It prints the tunnel command for the node it is on. ON YOUR LAPTOP:

    ssh -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 <you>@<login host>
    (a LOGIN node: start the viewer inside that same session; its internal
    name does not resolve from outside, and the public address picks any
    login node)
    open http://127.0.0.1:8765

Same core as the Tk viewer (figures/viewer.py::PickerCore), so the same
columns, the same panel images, and the same picks files -- rows/<VARIANT>.json
and zooms/<VARIANT>.json, written as you go. Picks made here open in the Tk
viewer and the other way round.

Everything is mouse-driven: volume list with a filter box, slice slider,
display toggles, brightness, a list of marked rows, and zoom placement by
clicking or dragging on ANY panel. While placing boxes, every panel shows the
figure's strip under it -- the magnified box and its |residual|, as
figures/make_figure.py prints them. The Tk viewer's keys work too (arrows,
space, enter, b, n/p, x, d, c, g, r, -/=).

The server listens on 127.0.0.1 only: nothing on the cluster network can reach
it, only your own SSH tunnel. Standard library + numpy + Pillow + h5py -- no
web framework to install, and no internet needed on the node.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figures import common as fc                                    # noqa: E402
from figures.common import nrmse, pretty, row_tag                    # noqa: E402
from figures.viewer import PickerCore                                # noqa: E402


class WebPicker(PickerCore):
    """PickerCore driven by HTTP requests instead of key presses."""

    def __init__(self, target, volume=None, sl=None):
        self.lock = threading.RLock()
        self._panels = {}                 # image key -> [PNG bytes per shown column]
        self.done = False
        super().__init__(target, volume=volume, sl=sl)

    # Nothing to redraw: the browser asks for state and images after each action.
    def render(self):
        if self.sel:                      # unmarking can leave si past the end
            self.si %= len(self.sel)

    # --- what the page shows ------------------------------------------------
    def _image_key(self, c):
        """Everything the PANEL IMAGES depend on -- not the zoom box, which the
        page draws itself, so dragging it never re-encodes an image."""
        raw = repr((fc.variant(), fc.DUMP_ROOT, c["vol"], c["pos"],
                    self.resid, self.dis, self.crop,
                    fc.BRIGHTNESS, tuple(self.shown_labels())))
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def panels(self, c, key):
        if key not in self._panels:
            heat = self.heat(c)
            out = []
            for lab in self.shown_labels():
                buf = io.BytesIO()
                self._panel(c, lab, heat).save(buf, format="PNG", compress_level=1)
                out.append(buf.getvalue())
            if len(self._panels) >= 8:
                self._panels.pop(next(iter(self._panels)))
            self._panels[key] = out
        return self._panels[key]

    def state(self):
        self.render()
        c = self.current()
        key = self._image_key(c)
        marked = self.is_marked(c)
        m = fc.STATUS_METRIC
        metrics = []
        for lab, spec in fc.COLUMNS:
            if lab in c["stored"] and m in c["stored"][lab]:
                metrics.append(dict(label=lab, value=float(c["stored"][lab][m]),
                                    kind="stored"))
            elif fc.col_spec(spec)[1] != "reference":
                metrics.append(dict(label=lab, value=float(nrmse(c["ref"], c["cols"][lab])),
                                    kind="live nrmse"))
        zooms = fc.load_zooms()
        flash = self.flash
        status = self._status(c, marked)            # consumes the flash
        return dict(
            variant=fc.variant(), target=self.target, mode=self.mode,
            vols=[pretty(v) for v in self.vols], vi=self.vi, vol=pretty(c["vol"]),
            pos=c["pos"], n=c["n"], sl=c["sl"],
            all_columns=[lab for lab, _ in fc.COLUMNS], shown=self.shown_labels(),
            grid=self.grid, view=self.view % len(fc.COLUMNS),
            resid=self.resid, dis=self.dis, crop=self.crop,
            brightness=fc.BRIGHTNESS, key=key,
            panel_text=[self.panel_text(c, lab, i)
                        for i, lab in enumerate(self.shown_labels())],
            marked=marked, metric=m, metrics=metrics,
            sel=[dict(volume=pretty(r["volume"]), slice=r["slice"],
                      zoom=zooms.get(row_tag(r))) for r in self.sel],
            si=self.si if self.sel else 0,
            zoom=list(self.zoom_of(c)) if self.mode == "zoom" else None,
            zoom_w=fc.ZOOM_W, zoom_h=fc.ZOOM_H, W=c["W"],
            flash=flash, status=status, done=self.done,
        )

    # --- what the page can do -------------------------------------------------
    def do(self, op, a):
        """One action from the page. Mirrors the Tk viewer's keys, mode rules
        included: slice / volume moves happen in BROWSE, box moves in ZOOM."""
        browse, zoom = self.mode == "browse", self.mode == "zoom"
        if op == "slice" and browse:
            self.horiz(int(a["d"]))
        elif op == "slice_to" and browse:
            self.pos = int(np.clip(int(a["pos"]), 0, self.current()["n"] - 1))
        elif op == "volume" and browse:
            self.vert(int(a["d"]))
        elif op == "goto" and browse:
            sl = a.get("sl")
            self.goto(int(a["vi"]), None if sl in (None, "") else int(sl))
        elif op == "toggle" and a.get("name") in ("resid", "dis", "crop", "grid"):
            self.set(**{a["name"]: not getattr(self, a["name"])})
        elif op == "view":
            i = int(a["i"])
            self.set(grid=True) if i < 0 else self.set(grid=False, view=i)
        elif op == "gain":
            self.gain(float(a["f"]))
        elif op == "brightness":
            fc.BRIGHTNESS = round(float(a["value"]), 3)
            self.flash = f"BRIGHTNESS = {fc.BRIGHTNESS}"
        elif op == "mark" and browse:
            self.mark()
        elif op == "unmark":
            self.unmark(int(a["i"]))
        elif op == "next_marked":
            self.cycle_marked()
        elif op == "goto_row" and self.sel:
            i = int(a["i"]) % len(self.sel)
            if zoom:
                self.si = i
            else:
                r = self.sel[i]
                self.vi = self.vols.index(r["volume"])
                self.pos = self._pos_of(r["volume"], r["slice"])
        elif op == "zoom" and browse:
            self.to_zoom()
        elif op == "zoom_row" and self.sel:
            self.mode, self.si = "zoom", int(a["i"]) % len(self.sel)
        elif op == "browse" and zoom:
            self.to_browse()
        elif op == "row" and zoom:
            self.cycle(int(a["d"]))
        elif op == "place" and zoom:
            c = self.current()
            self.set_zoom(float(a["x"]) - fc.ZOOM_W / 2,
                          float(a["y"]) - fc.ZOOM_H / 2, c["W"])
        elif op == "nudge" and zoom:
            self.nudge(int(a["dc"]), int(a["dr"]))
        elif op == "auto" and zoom:
            self.auto()
        elif op == "abort":
            self.restore()
            self.done = True
            self.flash = "aborted -- rows/ and zooms/ restored to their state at startup"
        elif op == "quit":
            self.done = True
            self.flash = "saved -- picks are written as you go"
        self.render()


class Session:
    """The server's ONE piece of state: which config is open, and its picker.

    The dataset menu switches config at run time. `figures.common` holds the
    config as module globals, so a switch resets them to their defaults before
    applying the next one, and a config that cannot be opened (no dumps yet)
    leaves the current one in place with the reason in the flash line.
    """

    def __init__(self, target, configs, start, volume=None, sl=None, overrides=None):
        self.lock = threading.RLock()
        self.target = target
        # --name / --dump-root: applied whenever the START config is (re)opened
        self.overrides = (start, dict(overrides or {}))
        self.configs = configs                   # fc.list_configs() entries
        self.path = None
        self.picker = None
        self.open(start, volume=volume, sl=sl)

    def open(self, path, volume=None, sl=None):
        prev = self.path
        try:
            fc.reset()
            if path:
                fc.apply_config(path)
            if path == self.overrides[0]:
                for k, v in self.overrides[1].items():
                    setattr(fc, k, v)
            self.picker = WebPicker(self.target, volume=volume, sl=sl)
            self.path = path
        except (SystemExit, Exception) as e:     # noqa: BLE001 -- keep serving
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            if prev is None or prev == path:
                raise
            fc.reset()
            fc.apply_config(prev)                # back to what was working
            if prev == self.overrides[0]:
                for k, v in self.overrides[1].items():
                    setattr(fc, k, v)
            self.picker.flash = f"cannot open {os.path.basename(path)}: {msg}"

    def do(self, op, a):
        if op == "config":
            hit = next((c for c in self.configs if c["name"] == a.get("name")), None)
            if hit is not None and hit["path"] != self.path:
                self.open(hit["path"])
            return
        self.picker.do(op, a)

    def state(self):
        st = self.picker.state()
        st["configs"] = [dict(name=c["name"], n=c["n_volumes"], cols=len(c["columns"]),
                              error=c["error"]) for c in self.configs]
        st["config"] = fc.variant()
        return st


# =============================================================================
#  HTTP
# =============================================================================
def make_handler(sess, server_box):
    page = PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # quiet: one line per image is noise
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype != "image/png"
                             else "private, max-age=600")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    self._send(200, page, "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    with sess.lock:
                        self._json(sess.state())
                elif u.path == "/api/strip":
                    # The figure's under-panel strip for panel i: magnified box
                    # (left) and its |residual| (right), at the box the page is
                    # showing (`z`), so a drag never waits on the server's copy.
                    q = parse_qs(u.query)
                    i = int(q.get("i", ["0"])[0])
                    with sess.lock:
                        picker = sess.picker
                        c = picker.current()
                        shown = picker.shown_labels()
                        if picker.mode != "zoom" or not 0 <= i < len(shown):
                            return self._send(404, b"no strip", "text/plain")
                        z = q.get("z", [""])[0].split("_")
                        zoom = ((int(z[0]), int(z[1])) if len(z) == 2
                                else picker.zoom_of(c))
                        zoom = (int(np.clip(zoom[0], 0, max(0, c["W"] - fc.ZOOM_W))),
                                int(np.clip(zoom[1], 0, max(0, c["W"] - fc.ZOOM_H))))
                        buf = io.BytesIO()
                        picker.strip(c, shown[i], zoom).save(buf, format="PNG",
                                                             compress_level=1)
                    self._send(200, buf.getvalue(), "image/png")
                elif u.path == "/api/panel":
                    q = parse_qs(u.query)
                    i = int(q.get("i", ["0"])[0])
                    with sess.lock:
                        picker = sess.picker
                        c = picker.current()
                        pngs = picker.panels(c, picker._image_key(c))
                    if not 0 <= i < len(pngs):
                        self._send(404, b"no such panel", "text/plain")
                    else:
                        self._send(200, pngs[i], "image/png")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:                  # noqa: BLE001 -- show, don't die
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_HEAD(self):
            # `curl -I` -- the quick "is the tunnel up?" check. Headers only.
            self.send_response(200 if urlparse(self.path).path == "/" else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/do":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                a = json.loads(self.rfile.read(n) or b"{}")
                with sess.lock:
                    sess.do(a.get("op", ""), a)
                    st = sess.state()
                    done = sess.picker.done
                self._json(st)
                if done:                     # stop after the reply is out
                    threading.Thread(target=server_box[0].shutdown, daemon=True).start()
            except Exception as e:                  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(sess, host, port, tries=20):
    """Bind the first free port from `port` up; return (server, port)."""
    box = [None]
    for p in range(port, port + tries):
        try:
            box[0] = ThreadingHTTPServer((host, p), make_handler(sess, box))
            return box[0], p
        except OSError:
            continue
    raise SystemExit(f"[webviewer] no free port in {port}..{port + tries - 1}; pass --port")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?",
                    help="config to open first (default: the first one with dumps); "
                         "every config under --configs is on the page's Dataset menu")
    ap.add_argument("--configs", default=os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "configs"),
                    help="directory of configs offered on the Dataset menu")
    ap.add_argument("-n", "--rows", type=int, default=3, help="how many rows to pick")
    ap.add_argument("--name", help="override the variant name")
    ap.add_argument("--dump-root", help="override DUMP_ROOT")
    ap.add_argument("--volume", help="start at this volume (any unique part of its name)")
    ap.add_argument("--slice", type=int, help="...and this source slice number")
    ap.add_argument("--port", type=int, default=8765,
                    help="first port to try (the next free one is used if taken)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to listen on. Keep 127.0.0.1: then only your "
                         "own SSH tunnel can reach it")
    a = ap.parse_args(argv)

    configs = fc.list_configs(a.configs)
    if a.config:
        start = os.path.abspath(a.config)
        if not any(os.path.abspath(c["path"]) == start for c in configs):
            configs.append(dict(path=start, name=os.path.splitext(os.path.basename(start))[0],
                                n_volumes=0, columns=[], error=None))
    else:
        ok = [c for c in configs if not c["error"]]
        if not ok:
            raise SystemExit("[webviewer] no config under " + a.configs + " has dumps:\n"
                             + "\n".join(f"  {c['name']}: {c['error']}" for c in configs))
        start = ok[0]["path"]
    for c in configs:
        print(f"[webviewer] {c['name']:<18} "
              + (f"{c['n_volumes']:>4} volumes, {len(c['columns'])} columns"
                 if not c["error"] else f"-- {c['error']}"))

    overrides = {}
    if a.name:
        overrides.update(NAME=a.name, VARIANT=a.name)
    if a.dump_root:
        overrides["DUMP_ROOT"] = a.dump_root
    sess = Session(a.rows, configs, start, volume=a.volume, sl=a.slice,
                   overrides=overrides)
    picker = sess.picker
    server, port = serve(sess, a.host, a.port)
    node, user = socket.gethostname(), getpass.getuser()
    print(f"[webviewer] {fc.variant()}: {len(picker.vols)} volumes, "
          f"columns {', '.join(lab for lab, _ in fc.COLUMNS)}")
    print(f"[webviewer] serving on http://{a.host}:{port}  (node {node})")
    # 127.0.0.1, not "localhost": the server listens on IPv4 only, and localhost
    # can resolve to ::1 on either end.
    if os.environ.get("SLURM_JOB_ID"):
        print("[webviewer] on your laptop (this is a compute node; hop through a login node):")
        print(f"    ssh -N -L {port}:127.0.0.1:{port} -J {user}@<login host> {user}@{node}")
    else:
        # A login node. Its cluster-internal name does not resolve from outside,
        # and the public login address lands on ANY login node -- so the tunnel
        # has to ride the same SSH session this was started from.
        print("[webviewer] this is a login node: forward the port on the SAME ssh session")
        print("    you start the viewer from (the public login address picks a login node")
        print(f"    at random, and {node.split('.')[0]} is not reachable by name from outside):")
        print(f"    ssh -o ExitOnForwardFailure=yes -L {port}:127.0.0.1:{port} {user}@<login host>")
        print("    then start this viewer in that session (append & to keep the shell)")
    print(f"    and open http://127.0.0.1:{port}")
    print("[webviewer] picks are saved as you go; Ctrl+C or 'Done' in the page to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("[webviewer] stopped")


# =============================================================================
#  The page
# =============================================================================
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure picker</title>
<style>
:root{--bg:#111214;--panel:#1b1c1f;--line:#2c2d31;--ink:#e8e7e3;--muted:#9a9890;
      --accent:#3b8eea;--good:#2fbf71;--warn:#e0a526;--danger:#e5534b;--box:#00dc00}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
#app{display:flex;height:100vh}
aside{width:320px;min-width:320px;overflow-y:auto;background:var(--panel);
  border-right:1px solid var(--line);padding:14px 14px 24px}
main{flex:1;position:relative;overflow:auto;padding:12px}
h1{font-size:15px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12px;margin-bottom:12px;word-break:break-all}
section{border-top:1px solid var(--line);padding:12px 0 4px}
section h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);margin:0 0 8px}
.row{display:flex;gap:6px;align-items:center;margin:6px 0;flex-wrap:wrap}
button{background:#26282c;color:var(--ink);border:1px solid #3a3c41;border-radius:6px;
  padding:6px 10px;font:inherit;cursor:pointer}
button:hover{border-color:#5a5d63}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.on{background:#20402e;border-color:var(--good)}
button.danger{border-color:#6b2e2b;color:#f2b5b1}
button:disabled{opacity:.4;cursor:default}
input[type=text]{width:100%;background:#0f1012;color:var(--ink);border:1px solid #3a3c41;
  border-radius:6px;padding:6px 8px;font:inherit}
select{width:100%;background:#0f1012;color:var(--ink);border:1px solid #3a3c41;
  border-radius:6px;font:12px ui-monospace,Consolas,monospace}
input[type=range]{flex:1;accent-color:var(--accent)}
label.chk{display:flex;gap:6px;align-items:center;cursor:pointer;margin:4px 0}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
.big{font-size:15px;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:3px 4px;border-bottom:1px solid var(--line)}
td.v{text-align:right;font-family:ui-monospace,Consolas,monospace}
td.k{color:var(--muted);font-size:11px}
.picks .pick{display:flex;align-items:center;gap:6px;padding:5px 6px;border-radius:6px;
  border:1px solid transparent;cursor:pointer;font-size:12px}
.picks .pick:hover{border-color:#3a3c41}
.picks .pick.cur{border-color:var(--box);background:#13261a}
.picks .pick span.t{flex:1;font-family:ui-monospace,Consolas,monospace}
.picks .x{padding:1px 7px}
#flash{min-height:20px;color:var(--warn);font-size:12px;margin-top:6px}
#modebar{display:none;position:sticky;top:0;z-index:5;margin:-12px -12px 12px;padding:8px 12px;
  background:#13261a;border-bottom:1px solid var(--good);font-size:13px}
#grid{display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start}
.cell{display:flex;flex-direction:column;gap:3px;flex:none}
.pnl{position:relative;background:#000;flex:none;user-select:none}
.strip{display:none;image-rendering:pixelated;border:1px solid var(--box);
  box-sizing:border-box;background:#000}
.pnl img{display:block;width:100%;height:100%;pointer-events:none}
.pnl.pix img{image-rendering:pixelated}
.pnl .lab{position:absolute;left:5px;top:4px;background:rgba(0,0,0,.78);color:#ffd84a;
  font:12px ui-monospace,Consolas,monospace;padding:1px 5px;border-radius:3px;pointer-events:none}
.pnl .zbox{position:absolute;border:2px solid var(--box);pointer-events:none;
  box-shadow:0 0 0 1px rgba(0,0,0,.6)}
.pnl.marked{outline:3px solid var(--box);outline-offset:-3px}
.pnl.zoomable{cursor:crosshair}
.hint{color:var(--muted);font-size:12px}
#overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:20;
  align-items:center;justify-content:center;font-size:18px;text-align:center;padding:30px}
</style></head>
<body><div id="app">
<aside>
  <h1>Figure picker</h1>
  <div class="sub" id="variant"></div>

  <section>
    <h2>Dataset</h2>
    <select id="cfg" title="every config under figures/configs; greyed = no dumps yet"></select>
    <div class="hint" id="cfgnote"></div>
  </section>

  <section>
    <h2>Volume</h2>
    <input type="text" id="vfilter" placeholder="filter, e.g. 2000045  (add a slice: 2000045 6)">
    <select id="vlist" size="9" style="margin-top:6px"></select>
    <div class="row"><button id="vprev">&#9650; prev</button><button id="vnext">&#9660; next</button>
      <span class="hint" id="vcount"></span></div>
  </section>

  <section>
    <h2>Slice</h2>
    <div class="row"><span class="big" id="sl"></span><span class="hint" id="slpos"></span></div>
    <div class="row"><button id="s5m">&laquo;</button><button id="s1m">&lsaquo;</button>
      <input type="range" id="slider" min="0" max="0" value="0">
      <button id="s1p">&rsaquo;</button><button id="s5p">&raquo;</button></div>
  </section>

  <section>
    <h2>Display</h2>
    <label class="chk"><input type="checkbox" id="t_resid"> Residual |ref &minus; x|</label>
    <label class="chk"><input type="checkbox" id="t_dis"> Disagreement overlay</label>
    <label class="chk"><input type="checkbox" id="t_crop"> Crop to anatomy</label>
    <label class="chk"><input type="checkbox" id="t_pix"> Hard pixels (no smoothing)</label>
    <div class="row"><span class="hint" style="width:70px">Columns</span>
      <select id="viewsel" style="flex:1"></select></div>
    <div class="row"><span class="hint" style="width:70px">Brightness</span>
      <input type="range" id="bright" min="-1.5" max="1.5" step="0.01" value="0">
      <span class="mono" id="brightv"></span></div>
  </section>

  <section>
    <h2>Metric <span id="metricname"></span></h2>
    <table id="metrics"></table>
  </section>

  <section>
    <h2>Picks <span id="pickcount"></span></h2>
    <div class="row">
      <button class="primary" id="mark">Mark this slice</button>
      <button id="nextmarked">Next marked</button>
    </div>
    <div class="picks" id="picks"></div>
    <div class="row" id="zoomctl">
      <button class="primary" id="tozoom">Place zoom boxes &rarr;</button>
    </div>
    <div id="zoomtools" style="display:none">
      <div class="hint">Click or drag on any panel to centre the box there.</div>
      <div class="row"><button id="rprev">&lsaquo; row</button><button id="rnext">row &rsaquo;</button>
        <button id="auto">Auto placement</button></div>
      <div class="row"><span class="hint">Nudge</span>
        <button id="nl">&larr;</button><button id="nu">&uarr;</button>
        <button id="nd">&darr;</button><button id="nr">&rarr;</button></div>
      <div class="row"><button id="back">&larr; Back to browsing</button></div>
    </div>
    <div id="flash"></div>
  </section>

  <section>
    <h2>Session</h2>
    <div class="hint" style="margin-bottom:6px">Picks are saved to disk as you go.</div>
    <div class="row"><button id="quit">Done</button>
      <button class="danger" id="abort">Discard this session's changes</button></div>
  </section>
</aside>
<main>
  <div id="modebar"></div>
  <div id="grid"></div>
</main>
</div>
<div id="overlay"></div>
<script>
"use strict";
let S = null, busy = false, queue = [], panels = [], strips = [], dragging = false;
const $ = id => document.getElementById(id);
const GAP = 8;

async function api(op, args) {
  const r = await fetch("/api/do", {method: "POST", headers: {"Content-Type": "application/json"},
                                    body: JSON.stringify(Object.assign({op}, args || {}))});
  const s = await r.json();
  if (s.error) { $("flash").textContent = s.error; return; }
  apply(s);
}
// One request in flight, the rest queued IN ORDER, so three quick clicks on
// "next slice" move three slices. Only ABSOLUTE actions (a slider position, a
// box position, a brightness) collapse to their latest value while waiting --
// a drag would otherwise replay every intermediate point.
const COALESCE = new Set(["slice_to", "place", "brightness"]);
function send(op, args) {
  const last = queue[queue.length - 1];
  if (COALESCE.has(op) && last && last[0] === op) last[1] = args;
  else queue.push([op, args]);
  pump();
}
function pump() {
  if (busy || !queue.length) return;
  busy = true;
  const [op, args] = queue.shift();
  api(op, args).finally(() => { busy = false; pump(); });
}

function apply(s) {
  const imagesChanged = !S || S.key !== s.key || S.shown.length !== s.shown.length;
  S = s;
  $("variant").textContent = s.variant;
  fillConfigs();
  if (s.done) {
    $("overlay").style.display = "flex";
    $("overlay").textContent = s.flash + " -- the server has stopped; you can close this tab.";
    return;
  }
  // volume list
  fillVolumes();
  $("vcount").textContent = (s.vi + 1) + " / " + s.vols.length;
  // slice
  $("sl").textContent = "slice " + s.sl;
  $("slpos").textContent = "(" + (s.pos + 1) + " / " + s.n + ")  " + s.vol;
  const sld = $("slider"); sld.max = s.n - 1; if (!sld.matches(":active")) sld.value = s.pos;
  // display
  $("t_resid").checked = s.resid; $("t_dis").checked = s.dis; $("t_crop").checked = s.crop;
  const vs = $("viewsel");
  if (vs.options.length !== s.all_columns.length + 1) {
    vs.innerHTML = "<option value='-1'>All, side by side</option>" +
      s.all_columns.map((c, i) => `<option value='${i}'>${esc(c)} only</option>`).join("");
  }
  vs.value = s.grid ? "-1" : String(s.view);
  $("brightv").textContent = s.brightness.toFixed(2);
  if (!$("bright").matches(":active")) $("bright").value = Math.log2(s.brightness);
  // metrics
  $("metricname").textContent = "(" + s.metric + ")";
  $("metrics").innerHTML = s.metrics.map(m =>
    `<tr><td>${esc(m.label)}</td><td class="k">${m.kind === "stored" ? "" : m.kind}</td>` +
    `<td class="v">${m.value.toFixed(m.kind === "stored" ? 3 : 4)}</td></tr>`).join("");
  // picks
  const zoom = s.mode === "zoom";
  $("pickcount").textContent = "(" + s.sel.length + " / " + s.target + ")";
  $("mark").textContent = s.marked ? "Unmark this slice" : "Mark this slice";
  $("mark").disabled = zoom || (!s.marked && s.sel.length >= s.target);
  $("nextmarked").disabled = !s.sel.length;
  $("picks").innerHTML = s.sel.map((r, i) =>
    `<div class="pick${zoom && i === s.si ? " cur" : ""}" data-i="${i}">` +
    `<span class="t">${i + 1}. ${esc(r.volume)}  sl ${r.slice}</span>` +
    `<span class="hint">${r.zoom ? "box set" : "auto box"}</span>` +
    `<button class="x" data-x="${i}" title="unmark">&times;</button></div>`).join("") ||
    "<div class='hint'>Nothing marked yet.</div>";
  $("zoomctl").style.display = zoom ? "none" : "flex";
  $("tozoom").disabled = !s.marked && s.sel.length >= s.target;
  $("zoomtools").style.display = zoom ? "block" : "none";
  $("modebar").style.display = zoom ? "block" : "none";
  if (zoom) $("modebar").textContent = `Placing the zoom box for row ${s.si + 1} of ` +
    `${s.sel.length}: ${s.vol}, slice ${s.sl}. Click or drag on any panel.`;
  for (const id of ["vprev", "vnext", "s5m", "s1m", "s1p", "s5p"]) $(id).disabled = zoom;
  $("slider").disabled = zoom; $("vlist").disabled = zoom;
  $("flash").textContent = s.flash || "";
  // panels
  if (imagesChanged) buildPanels(); else updateOverlays();
  layout();
}

let lastVolKey = "";
function fillVolumes() {
  const raw = $("vfilter").value.trim().split(/\s+/).filter(Boolean);
  if (raw.length > 1 && /^-?\d+$/.test(raw[raw.length - 1])) raw.pop();
  const q = raw.join(" ").toLowerCase();
  const key = S.config + "|" + q + "|" + S.vols.length + "|" + S.vi;
  if (key === lastVolKey) return;
  lastVolKey = key;
  const list = $("vlist");
  list.innerHTML = S.vols.map((v, i) => [v, i]).filter(([v]) => v.toLowerCase().includes(q))
    .map(([v, i]) => `<option value="${i}"${i === S.vi ? " selected" : ""}>` +
                     `${String(i + 1).padStart(4, " ")}  ${esc(v)}</option>`).join("");
}

function buildPanels() {
  const g = $("grid"); g.innerHTML = ""; panels = []; strips = [];
  S.shown.forEach((lab, i) => {
    // A cell = the panel, and under it (ZOOM only) the figure's strip: the
    // magnified box and its residual, as make_figure.py lays them out.
    const cell = document.createElement("div"); cell.className = "cell";
    const st = document.createElement("img"); st.className = "strip";
    st.alt = "zoom | residual";
    const d = document.createElement("div");
    d.className = "pnl" + ($("t_pix").checked ? " pix" : "");
    d.innerHTML = `<img alt="${esc(lab)}" src="/api/panel?i=${i}&k=${S.key}">` +
                  `<div class="lab"></div><div class="zbox" style="display:none"></div>`;
    d.addEventListener("mousedown", e => { if (S.mode !== "zoom") return;
      dragging = true; place(e, d); e.preventDefault(); });
    d.addEventListener("mousemove", e => { if (dragging) place(e, d); });
    cell.appendChild(d); cell.appendChild(st); g.appendChild(cell);
    panels.push(d); strips.push(st);
  });
  updateOverlays();
}
window.addEventListener("mouseup", () => { dragging = false; });

function updateOverlays() {
  panels.forEach((d, i) => {
    d.querySelector(".lab").textContent = S.panel_text[i] || "";
    d.classList.toggle("marked", S.mode === "browse" && S.marked);
    d.classList.toggle("zoomable", S.mode === "zoom");
    drawBox(d, S.zoom);
  });
  drawStrips();
}
// The strips follow the box the server last confirmed (a drag updates them as
// each coalesced `place` returns), so the image under a panel is always the
// window the figure will print.
function drawStrips() {
  strips.forEach((st, i) => {
    if (S.mode !== "zoom" || !S.zoom) { st.style.display = "none"; return; }
    const src = `/api/strip?i=${i}&k=${S.key}&z=${S.zoom[0]}_${S.zoom[1]}`;
    if (st.dataset.src !== src) { st.src = src; st.dataset.src = src; }
    st.style.display = "block";
  });
}
function drawBox(d, z) {
  const b = d.querySelector(".zbox");
  if (!z) { b.style.display = "none"; return; }
  const P = d.clientWidth, k = P / S.W;
  Object.assign(b.style, {display: "block", left: z[0] * k + "px", top: z[1] * k + "px",
                          width: S.zoom_w * k + "px", height: S.zoom_h * k + "px"});
}
// Centre the box at the cursor, in image pixels; drawn at once, sent coalesced.
function place(e, d) {
  const r = d.getBoundingClientRect(), k = S.W / r.width;
  const x = (e.clientX - r.left) * k, y = (e.clientY - r.top) * k;
  const c0 = Math.max(0, Math.min(S.W - S.zoom_w, Math.round(x - S.zoom_w / 2)));
  const r0 = Math.max(0, Math.min(S.W - S.zoom_h, Math.round(y - S.zoom_h / 2)));
  S.zoom = [c0, r0]; panels.forEach(p => drawBox(p, S.zoom));
  send("place", {x, y});
}

function layout() {
  if (!S || !panels.length) return;
  const m = document.querySelector("main"), modebar = $("modebar");
  const W = m.clientWidth - 24, H = m.clientHeight - 24 - (modebar.offsetHeight || 0);
  const n = panels.length, per = n === 1 ? 1 : Math.min(3, n), rows = Math.ceil(n / per);
  // In ZOOM each panel has its strip underneath: height P * zh / (2 zw + 1), +3px gap.
  const zoom = S.mode === "zoom", ratio = S.zoom_h / (2 * S.zoom_w + 1);
  const P = Math.max(80, Math.floor(Math.min((W - GAP * (per - 1)) / per,
                                             ((H - GAP * (rows - 1)) / rows - (zoom ? 3 : 0))
                                             / (1 + (zoom ? ratio : 0)))));
  $("grid").style.width = (per * P + (per - 1) * GAP) + "px";
  panels.forEach(d => { d.style.width = P + "px"; d.style.height = P + "px"; drawBox(d, S.zoom); });
  strips.forEach(st => { st.style.width = P + "px"; st.style.height = Math.round(P * ratio) + "px"; });
}
window.addEventListener("resize", layout);

function esc(t) { return String(t).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

// --- controls ---
// Dataset menu: one entry per config; switching reloads everything.
let lastCfgKey = "";
function fillConfigs() {
  const key = S.config + "|" + S.configs.map(c => c.name + c.n + (c.error || "")).join(",");
  if (key === lastCfgKey) return;
  lastCfgKey = key;
  $("cfg").innerHTML = S.configs.map(c =>
    `<option value="${esc(c.name)}"${c.name === S.config ? " selected" : ""}` +
    `${c.error ? " disabled" : ""}>${esc(c.name)}` +
    (c.error ? "  (no dumps)" : `  \u2014 ${c.n} vol, ${c.cols} col`) + `</option>`).join("");
  const cur = S.configs.find(c => c.name === S.config);
  $("cfgnote").textContent = cur && cur.error ? cur.error : "";
}
$("cfg").onchange = () => { lastVolKey = ""; send("config", {name: $("cfg").value}); };
$("vfilter").addEventListener("input", () => fillVolumes());
$("vfilter").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const opt = $("vlist").options[$("vlist").selectedIndex >= 0 ? $("vlist").selectedIndex : 0];
  if (!opt) return;
  const parts = $("vfilter").value.trim().split(/\s+/);
  const sl = parts.length > 1 && /^-?\d+$/.test(parts[parts.length - 1]) ? parts[parts.length - 1] : "";
  send("goto", {vi: +opt.value, sl}); e.preventDefault();
});
$("vlist").addEventListener("change", e => send("goto", {vi: +e.target.value}));
$("vprev").onclick = () => send("volume", {d: -1});
$("vnext").onclick = () => send("volume", {d: 1});
$("s5m").onclick = () => send("slice", {d: -5});
$("s1m").onclick = () => send("slice", {d: -1});
$("s1p").onclick = () => send("slice", {d: 1});
$("s5p").onclick = () => send("slice", {d: 5});
$("slider").addEventListener("input", e => send("slice_to", {pos: +e.target.value}));
for (const n of ["resid", "dis", "crop"]) $("t_" + n).onchange = () => send("toggle", {name: n});
$("t_pix").onchange = () => panels.forEach(d => d.classList.toggle("pix", $("t_pix").checked));
$("viewsel").onchange = e => send("view", {i: +e.target.value});
$("bright").addEventListener("input", e => send("brightness", {value: Math.pow(2, +e.target.value)}));
$("mark").onclick = () => send("mark");
$("nextmarked").onclick = () => send("next_marked");
$("tozoom").onclick = () => send("zoom");
$("back").onclick = () => send("browse");
$("rprev").onclick = () => send("row", {d: -1});
$("rnext").onclick = () => send("row", {d: 1});
$("auto").onclick = () => send("auto");
$("nl").onclick = () => send("nudge", {dc: -1, dr: 0});
$("nr").onclick = () => send("nudge", {dc: 1, dr: 0});
$("nu").onclick = () => send("nudge", {dc: 0, dr: -1});
$("nd").onclick = () => send("nudge", {dc: 0, dr: 1});
$("picks").addEventListener("click", e => {
  const x = e.target.closest("[data-x]");
  if (x) { send("unmark", {i: +x.dataset.x}); return; }
  const p = e.target.closest(".pick");
  // In ZOOM: switch to that row's box. In BROWSE: jump to that slice.
  if (p) send("goto_row", {i: +p.dataset.i});
});
$("picks").addEventListener("dblclick", e => {
  const p = e.target.closest(".pick"); if (p) send("zoom_row", {i: +p.dataset.i});
});
$("quit").onclick = () => send("quit");
$("abort").onclick = () => { if (confirm("Put rows/ and zooms/ back to how they were " +
                                         "when this session started, and stop?")) send("abort"); };

// The Tk viewer's keys, unless you are typing in a box.
document.addEventListener("keydown", e => {
  if (!S || e.target.matches("input[type=text]")) return;
  const z = S.mode === "zoom", k = e.key;
  const map = {
    ArrowLeft: () => z ? send("nudge", {dc: -1, dr: 0}) : send("slice", {d: e.shiftKey ? -5 : -1}),
    ArrowRight: () => z ? send("nudge", {dc: 1, dr: 0}) : send("slice", {d: e.shiftKey ? 5 : 1}),
    ArrowUp: () => z ? send("nudge", {dc: 0, dr: -1}) : send("volume", {d: -1}),
    ArrowDown: () => z ? send("nudge", {dc: 0, dr: 1}) : send("volume", {d: 1}),
    " ": () => send("mark"), Enter: () => send("zoom"), b: () => send("browse"),
    n: () => send("row", {d: 1}), p: () => send("row", {d: -1}), m: () => send("next_marked"),
    x: () => send("toggle", {name: "resid"}), d: () => send("toggle", {name: "dis"}),
    c: () => send("toggle", {name: "crop"}), g: () => send("toggle", {name: "grid"}),
    r: () => send("auto"), "-": () => send("gain", {f: 1 / 1.15}), "=": () => send("gain", {f: 1.15}),
    o: () => { $("vfilter").focus(); $("vfilter").select(); },
  };
  if (map[k] && !e.ctrlKey && !e.metaKey && !e.altKey) { map[k](); e.preventDefault(); }
});

fetch("/api/state").then(r => r.json()).then(s => s.error ? ($("flash").textContent = s.error) : apply(s));
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
