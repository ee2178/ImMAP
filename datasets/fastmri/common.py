import os
import random
import h5py
import torch


# ============================================================
# Dataset registry
# ============================================================

FASTMRI_PATHS = {
    "knee": {
        "kspace_root": "/home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_val",
        "smap_root": "/home/ee2178/scratch/ee2178/datasets/fastmri_preprocessed/knee_coil_combined/pd/val",
        "scale_fac": 5e3,
    },
    "brain": {
        "kspace_root": "/home/ee2178/scratch/ee2178/datasets/fastmri/brain/multicoil_val",
        "smap_root": "/home/ee2178/scratch/ee2178/datasets/fastmri_preprocessed/brain_T2W_coil_combined/val",
        "scale_fac": 2e3,
    },
}


# ============================================================
# Generic FastMRI loader
# ============================================================

def load_fastmri_data(
    anatomy,
    kspace_fname=None,
    slice_idx=None,
    start_slice=None,
    end_slice=None,
    kspace_root=None,
    smap_root=None,
    scale_fac=None,
    device="cpu",
):
    """
    Load FastMRI multicoil data.

    Parameters
    ----------
    anatomy : str
        "brain" or "knee"

    kspace_fname : str or None
        File to load. If None, randomly samples a file.

    slice_idx : int or None
        Explicit slice index to load.

    start_slice : int or None
        Lower bound for random slice sampling.

    end_slice : int or None
        Upper bound for random slice sampling.

    kspace_root : str or None
        Override default kspace path.

    smap_root : str or None
        Override default smaps path.

    scale_fac : float or None
        Override default scaling factor.

    device : str
        Torch device.

    Returns
    -------
    kspace
    smaps
    mask
    gnd_truth
    """

    # --------------------------------------------------------
    # Validate anatomy
    # --------------------------------------------------------

    if anatomy not in FASTMRI_PATHS:

        supported = list(FASTMRI_PATHS.keys())

        raise ValueError(
            f"Unsupported anatomy '{anatomy}'. "
            f"Supported anatomies: {supported}"
        )

    cfg = FASTMRI_PATHS[anatomy]

    # --------------------------------------------------------
    # Use overrides if provided
    # --------------------------------------------------------

    if kspace_root is None:
        kspace_root = cfg["kspace_root"]

    if smap_root is None:
        smap_root = cfg["smap_root"]

    if scale_fac is None:
        scale_fac = cfg["scale_fac"]

    # --------------------------------------------------------
    # Random file selection
    # --------------------------------------------------------

    if kspace_fname is None:

        files = [
            f for f in os.listdir(smap_root)
            if f.endswith(".h5")
        ]

        if len(files) == 0:

            raise ValueError(
                f"No .h5 files found in {smap_root}"
            )

        kspace_fname = random.choice(files)

    # --------------------------------------------------------
    # Filepaths
    # --------------------------------------------------------

    kspace_path = os.path.join(
        kspace_root,
        kspace_fname,
    )

    smaps_path = os.path.join(
        smap_root,
        kspace_fname,
    )

    # --------------------------------------------------------
    # Load sensitivity maps + ground truth
    # --------------------------------------------------------

    with h5py.File(smaps_path, "r") as f:

        smaps = f["smaps"][()]
        gnd_truth = f["image"][()]

    # --------------------------------------------------------
    # Load k-space
    # --------------------------------------------------------

    with h5py.File(kspace_path, "r") as f:

        kspace = f["kspace"][()]

    # --------------------------------------------------------
    # Random slice sampling
    # --------------------------------------------------------

    num_slices = kspace.shape[0]

    if slice_idx is None:

        s0 = 0 if start_slice is None else start_slice
        s1 = num_slices if end_slice is None else end_slice

        s0 = max(0, s0)
        s1 = min(num_slices, s1)

        if s0 >= s1:
            raise ValueError(
                f"Invalid slice range [{s0}, {s1})"
            )

        slice_idx = random.randint(s0, s1 - 1)

    # --------------------------------------------------------
    # Slice extraction
    # --------------------------------------------------------

    sl = slice(slice_idx, slice_idx + 1)

    kspace = kspace[sl]
    smaps = smaps[sl]
    gnd_truth = gnd_truth[sl]

    # --------------------------------------------------------
    # Convert to torch
    # --------------------------------------------------------

    kspace = (
        torch.from_numpy(kspace)
        .to(device)
        * scale_fac
    )

    smaps = (
        torch.from_numpy(smaps)
        .to(device)
    )

    gnd_truth = (
        torch.from_numpy(gnd_truth)
        .to(device)
        * scale_fac
    )

    # --------------------------------------------------------
    # Organ mask from sensitivity maps
    # --------------------------------------------------------

    mask = (
        torch.sum(
            smaps.abs(),
            dim=1,
            keepdim=True,
        ) > 0
    )
    print(f"Loading slice {slice_idx} from "+kspace_path)
    return (
        kspace,
        smaps,
        mask,
        gnd_truth,
    )


# ============================================================
# Thin convenience wrappers
# ============================================================

def load_knee_data(**kwargs):

    return load_fastmri_data(
        anatomy="knee",
        **kwargs,
    )


def load_brain_data(**kwargs):

    return load_fastmri_data(
        anatomy="brain",
        **kwargs,
    )
