"""
Figure tooling: inspect dumped reconstructions and pick what a figure shows.

    scripts/dump_eval.py   run the nets once, write per-volume HDF5s
    figures/viewer.py      scrub those, mark rows, place zoom windows
    figures/common.py      the config, loading and display both sides share

Deliberately torch-free below `scripts/`: `common` and `viewer` import numpy,
h5py, PIL and matplotlib only, so they run against an rsync'd copy of the
cluster's output without a GPU or a model definition in sight.
"""
