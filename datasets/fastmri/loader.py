import os
import random
import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets.registry import register_loader
from operators.truncate import embedded_size

# ============================================================
# Config
# ============================================================

FASTMRI_PATHS = {
    "knee": {
        "kspace_root": "../../datasets/fastmri/knee/multicoil_val",
        "smap_root": "../../datasets/fastmri_preprocessed/knee_coil_combined/pd/val",
        "scale_fac": 5e3,
        "filter": "PD",
    },
    "brain": {
        "kspace_root": "../../datasets/fastmri/brain/multicoil_val",
        "smap_root": "../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val",
        "scale_fac": 2e3,
        "filter": "T2",
    },
}


# ============================================================
# Filtering helpers
# ============================================================

def is_pd_scan(fname):
    with h5py.File(fname, "r") as f:
        return f.attrs.get("acquisition", "") == "CORPD_FBK"


def is_t2_scan(fname):
    with h5py.File(fname, "r") as f:
        acq = f.attrs.get("acquisition", "")
        return "T2" in acq or "T2W" in acq


# ============================================================
# Dataset
# ============================================================

class FastMRIDataset(Dataset):
    def __init__(
        self,
        task="recon", # Default to a Recon task
        anatomy="brain",
        ### For denoising, introduce transformation parameters
        crop_size=None,
        center_crop=None,
        random_flips=True,
        ### Sampling Parameters
        start_slice=0,
        end_slice=None,
        kspace_root=None,
        smap_root=None,
        scale_fac=None,
        pad_multiple=1,
        enumerate_slices=False,
        volumes=None,
    ):

        if anatomy not in FASTMRI_PATHS:
            raise ValueError(f"Unknown anatomy {anatomy}")

        cfg = FASTMRI_PATHS[anatomy]

        self.kspace_root = kspace_root or cfg["kspace_root"]
        self.smap_root = smap_root or cfg["smap_root"]
        self.scale_fac = scale_fac or cfg["scale_fac"]
        self.start_slice = start_slice
        self.end_slice = end_slice
        self.task = task
        # Image-domain embedding for unrolled multigrid nets: the recon branch
        # reports the smallest grid >= the measured one that is divisible by
        # this, and training builds `E @ Truncate` onto it. 1 disables it.
        # See operators/truncate.py for why this beats padding the operator.
        self.pad_multiple = int(pad_multiple)
        self.enumerate_slices = bool(enumerate_slices)

        # ----------------------------------------------------
        # Build filtered file list (IMPORTANT PART)
        # ----------------------------------------------------
        self.files = self._build_file_list(anatomy)

        if volumes is not None:
            # An explicit subset, named the way `item_id` reports it. Order
            # follows `volumes`, so a figure's column order is the caller's.
            want = [v if v.endswith(".h5") else f"{v}.h5" for v in volumes]
            have = set(self.files)
            missing = [v for v in want if v not in have]
            if missing:
                raise ValueError(
                    f"volume(s) {missing} are not in {self.kspace_root} (or were "
                    f"filtered out by anatomy={anatomy!r}); "
                    f"{len(self.files)} available")
            self.files = want

        if len(self.files) == 0:
            raise ValueError(f"No valid scans found for anatomy={anatomy}")

        # ----------------------------------------------------
        # Index: one entry per ITEM the dataset serves
        # ----------------------------------------------------
        # Default (`enumerate_slices=False`) is one item per volume with the
        # slice drawn at __getitem__ time -- the training behaviour, unchanged.
        # `enumerate_slices=True` makes each (volume, slice) its own item, so a
        # sequential pass covers whole volumes in a reproducible order. That is
        # what `scripts/dump_eval.py` needs: a viewer scrubs slices, and a
        # dataset that hands back one random slice per volume cannot supply
        # them.
        self.index = self._build_index()

        # If the task is denoising, build a transform. 
        if task == "denoising":

            tfms = []

            if center_crop is not None:
                tfms.append(transforms.CenterCrop(center_crop))

            if crop_size is not None:
                tfms.append(transforms.RandomCrop(crop_size))

            if random_flips:
                tfms += [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]

            self.transform = transforms.Compose(tfms)

        else:
            self.transform = None


    def _build_file_list(self, anatomy):

        files = [f for f in os.listdir(self.kspace_root) if f.endswith(".h5")]

        valid = []

        for f in files:
            path = os.path.join(self.kspace_root, f)

            try:
                with h5py.File(path, "r") as h:
                    acq = h.attrs.get("acquisition", "")

                if anatomy == "knee" and acq == "CORPD_FBK":
                    valid.append(f)

                if anatomy == "brain" and ("T2" in acq or "T2W" in acq):
                    valid.append(f)

            except Exception:
                continue

        return valid

    def _volume_slices(self, fname):
        """How many slices `fname` actually has, from the preprocessed image."""
        with h5py.File(os.path.join(self.smap_root, fname), "r") as f:
            return int(f["image"].shape[0])

    def _build_index(self):
        """`[(fname, slice_or_None)]`, one entry per item served."""
        if not self.enumerate_slices:
            return [(f, None) for f in self.files]

        out = []
        for f in self.files:
            n = self._volume_slices(f)
            # `end_slice=None` means "to the end of the volume" here. In the
            # sampling path it means "always start_slice", which is the right
            # default there and the wrong one for a dump.
            hi = n if self.end_slice is None else min(int(self.end_slice), n)
            lo = min(int(self.start_slice), hi)
            out.extend((f, s) for s in range(lo, hi))
        if not out:
            raise ValueError(
                f"enumerate_slices produced no items: start_slice="
                f"{self.start_slice}, end_slice={self.end_slice} selects nothing "
                f"in volumes of {[self._volume_slices(f) for f in self.files[:3]]}"
                f"... slices")
        return out

    def __len__(self):
        return len(self.index)

    def _sample_slice(self):
        lo = self.start_slice
        hi = self.end_slice
        if hi is None:
            return lo
        return random.randint(lo, hi - 1)

    def item_id(self, idx):
        """`(volume, slice)` for item `idx` -- the key a picked row is saved under.

        Only meaningful under `enumerate_slices=True`; in the sampling path the
        slice is not decided until `__getitem__` runs, so there is no stable
        answer to give and asking is a bug worth naming.
        """
        fname, sl = self.index[idx]
        if sl is None:
            raise RuntimeError(
                "item_id() needs enumerate_slices=True: without it the slice is "
                "drawn inside __getitem__ and no stable identity exists.")
        return os.path.splitext(fname)[0], int(sl)

    def __getitem__(self, idx):

        fname, sl = self.index[idx]

        kspace_path = os.path.join(self.kspace_root, fname)
        smap_path = os.path.join(self.smap_root, fname)

        if sl is None:
            sl = self._sample_slice()
        sl = slice(sl, sl + 1)

        # Split into recon and denoising branches
        if self.task == "denoising":
            # Grab image
            with h5py.File(smap_path, "r") as f:
                image = f["image"][sl]

            image = torch.from_numpy(image)
            # Treat complex valued image as two channel
            image_2ch = torch.cat(
                [image.real, image.imag],
                dim=0
            )

            if self.transform is not None:
                image_2ch = self.transform(image_2ch)
            
            image = torch.complex(
                image_2ch[0],
                image_2ch[1],
            )
            # Return image as a one-channel image
            image = image.unsqueeze(0)

            return image * self.scale_fac

        elif self.task == "recon":
            # ---------------------------
            # Load kspace + image + smaps
            # ---------------------------
        
            with h5py.File(kspace_path, "r") as f:
                kspace = f["kspace"][sl]

            with h5py.File(smap_path, "r") as f:
                smaps = f["smaps"][sl]
                image = f["image"][sl]

            # ---------------------------
            # Convert
            # ---------------------------
                
            kspace = torch.from_numpy(kspace).squeeze() * self.scale_fac
            smaps = torch.from_numpy(smaps).squeeze()
            image = torch.from_numpy(image) * self.scale_fac
            
            #For some reason these come out with a batch dimension, we should squeeze everything

            # ---------------------------
            # Mask from coil support
            # ---------------------------

            # dim=0, NOT dim=1: the `.squeeze()` above already dropped the
            # leading singleton slice axis, so smaps is (NC, H, W) and the coil
            # axis is first. dim=1 summed over H and returned (NC, 1, W), which
            # -- being the same trailing size as the image -- BROADCASTS
            # against a (1, H, W) image rather than raising, so `image * mask`
            # would have quietly grown a coil axis instead of masking anything.
            # It went unnoticed while use_organ_mask was False everywhere and
            # nothing consumed this.
            if smaps.dim() != 3:
                raise ValueError(
                    f"expected smaps (NC, H, W) after squeeze, got "
                    f"{tuple(smaps.shape)}. A single-coil volume squeezes the "
                    f"coil axis away entirely, and the sum below would then "
                    f"run over a spatial axis.")

            mask = (smaps.abs().sum(dim=0, keepdim=True) > 0)

            # Cheap, and it is the invariant every consumer relies on: the mask
            # multiplies the image in the loss, the metrics and the val panel,
            # and a mismatch there broadcasts into a wrong shape instead of
            # failing. Checking it at the source names the file, not a tensor.
            if mask.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"organ mask {tuple(mask.shape)} does not cover the image "
                    f"{tuple(image.shape)} in {fname}.")

            # The grid the NETWORK should solve on. Derived here rather than in
            # the training loop because this is where the final image size is
            # settled (crops, per-volume matrix sizes), and because an eval
            # script then reproduces the embedding from the batch alone.
            H, W = image.shape[-2:]
            pad_hw = torch.tensor(
                embedded_size((H, W), self.pad_multiple), dtype=torch.long)

            return kspace, smaps, image, mask, pad_hw


# ============================================================
# Loader
# ============================================================

@register_loader("fastmri")
def get_fastmri_loader(
    anatomy="brain",
    task="recon", # Default to a Recon task
    crop_size=None,
    center_crop=None,
    random_flips=True,
    batch_size=1,
    shuffle=True,
    start_slice=0,
    end_slice=None,
    kspace_root=None,
    smap_root=None,
    scale_fac=None,
    pad_multiple=1,
    drop_last=True,
    enumerate_slices=False,
    volumes=None,
    num_workers=8,
):
    dataset = FastMRIDataset(
        task =task,
        anatomy=anatomy,
        crop_size=crop_size,
        center_crop=center_crop,
        random_flips=random_flips,
        start_slice=start_slice,
        end_slice=end_slice,
        kspace_root=kspace_root,
        smap_root=smap_root,
        scale_fac=scale_fac,
        pad_multiple=pad_multiple,
        enumerate_slices=enumerate_slices,
        volumes=volumes,
    )

    # Every keyword this function does not name is silently dropped by the
    # registry, so a new dataset option must be added HERE as well as on the
    # class or it becomes a no-op that looks like it worked.
    num_workers = int(num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        # Some extra defaults to improve GPU utilization
        num_workers=num_workers,          # tune
        pin_memory=True,
        # Both of these are illegal at num_workers=0, which a dump run wants so
        # that item order is trivially the index order.
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 4} if num_workers > 0 else {}),
    )
