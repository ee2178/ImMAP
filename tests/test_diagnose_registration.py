# -*- coding: utf-8 -*-
"""scripts/diagnose_nyumets_registration.py must RECOVER a known misregistration.

A diagnostic that reported the wrong rotation, or failed to see one, would send the rebuild in
the wrong direction, so the search is pinned on a phantom moved by a known rotation, scale and
shift, and the whole script is run once on a tiny h5 set whose second study is that phantom.
"""
import os
import sys

import h5py
import numpy as np
import pytest
import torch

from scripts import diagnose_nyumets_registration as dg

H = 96


def phantom(seed=0):
    """A lopsided 'head': an off-centre ellipse with blobs, so rotation and scale are visible."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(H).float(), indexing="ij")
    head = (((yy - 50) / 30) ** 2 + ((xx - 46) / 22) ** 2) < 1
    img = torch.zeros(H, H)
    for _ in range(60):
        cy, cx = 30 + 40 * torch.rand(1, generator=g), 30 + 34 * torch.rand(1, generator=g)
        s = 1.2 + 1.8 * torch.rand(1, generator=g)
        img += torch.rand(1, generator=g) * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
    img = (img + 0.3) * head
    return img, head


def move(img, mask, angle, scale, shift):
    a = dg.warp(img[None, None], angle, scale, shift)[0, 0]
    m = dg.warp(mask[None, None].float(), angle, scale, shift)[0, 0] > 0.5
    return a * m, m


@pytest.mark.parametrize("angle,scale,shift", [(7.0, 1.0, (0, 0)), (0.0, 1.08, (0, 0)),
                                               (0.0, 1.0, (6, -8)), (-5.0, 0.94, (4, 6))])
def test_search_recovers_a_known_similarity(angle, scale, shift):
    a, ma = phantom()
    b, mb = move(a, ma, angle, scale, shift)
    assert dg.masked_ncc(a, b, ma & mb) < 0.97, "the phantom was not really moved"
    r = dg.similarity_search(a, ma, b, mb, np.arange(-12, 13, 1.0).tolist(),
                             np.arange(0.88, 1.121, 0.02).round(4).tolist())
    assert r["after"] > 0.97 and r["after"] > r["before"] + 0.02, r
    # the transform found UNDOES the one applied
    assert abs(r["angle"] + angle) <= 1.0, r
    assert abs(r["scale"] * scale - 1.0) <= 0.025, r


def test_identity_is_left_alone():
    a, ma = phantom()
    r = dg.similarity_search(a, ma, a.clone(), ma.clone(), np.arange(-6, 7, 1.0).tolist(),
                             [0.96, 1.0, 1.04])
    assert (r["angle"], r["scale"], r["dh"], r["dw"]) == (0.0, 1.0, 0, 0) and r["after"] > 0.999


def test_best_dz_finds_the_matching_slice():
    # neighbouring slices differ in COARSE structure, which is what survives the 8x pooling
    base, head = phantom()
    yy = torch.arange(H).float()[:, None]
    vol = torch.stack([base * (1 + 0.6 * torch.cos(yy / 8.0 + 0.9 * s)) for s in range(7)])
    m = head[None].expand(7, -1, -1)
    assert dg.best_dz(vol[5], m[5], vol, m, 3, 3) == 2


def test_script_end_to_end(tmp_path, capsys, monkeypatch):
    base, mask = phantom()
    for s, (ang, sc, sh) in enumerate([(0.0, 1.0, (0, 0)), (6.0, 1.06, (4, -4))]):
        d = tmp_path / ("PT_s%02d" % s)
        os.makedirs(d)
        vol = np.zeros((6, H, H, 4), np.float32)
        msk = np.zeros((6, H, H, 1), np.uint8)
        for z in range(6):
            img, m = phantom(z)
            img, m = move(img, m, ang, sc, sh)
            vol[z, :, :, 1] = img.numpy()
            vol[z, :, :, 2] = img.numpy() * 1.1               # CT1: aligned with its own T1
            msk[z, :, :, 0] = m.numpy()
        with h5py.File(d / ("PT_s%02d_img.h5" % s), "w") as f:
            f.create_dataset("img_median_mad", data=vol)
            f.create_dataset("mask", data=msk)
            f.create_dataset("slice_index", data=np.arange(60, 66, dtype=np.int32))
            f.attrs["patient"], f.attrs["session"] = "PT", "s%02d" % s
            f.attrs["reg_method"], f.attrs["reg_ncc"] = "centroid", (1.0 if s == 0 else 0.4)
    csv = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(tmp_path), "--device", "cpu",
                                      "--slices", "2", "--max-angle", "8", "--csv", str(csv)])
    dg.main()
    out = capsys.readouterr().out
    assert "centroid x2" in out and "CROSS-STUDY, IN-PLANE" in out and "WITHIN ONE STUDY" in out
    lines = csv.read_text().strip().splitlines()
    assert len(lines) == 3
    cols = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert abs(float(cols["angle"]) + 6.0) <= 1.0 and float(cols["after"]) > 0.95
    assert int(cols["dz"]) == 0
