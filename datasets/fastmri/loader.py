import os
import random
import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets.registry import register_loader

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

        # ----------------------------------------------------
        # Build filtered file list (IMPORTANT PART)
        # ----------------------------------------------------
        self.files = self._build_file_list(anatomy)

        if len(self.files) == 0:
            raise ValueError(f"No valid scans found for anatomy={anatomy}")
        
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

    def __len__(self):
        return len(self.files)

    def _sample_slice(self):
        lo = self.start_slice
        hi = self.end_slice
        if hi is None:
            return lo
        return random.randint(lo, hi - 1)

    def __getitem__(self, idx):

        fname = self.files[idx]

        kspace_path = os.path.join(self.kspace_root, fname)
        smap_path = os.path.join(self.smap_root, fname)

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

            mask = (smaps.abs().sum(dim=1, keepdim=True) > 0)

            return kspace, smaps, image, mask


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
    drop_last=True,
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
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        # Some extra defaults to improve GPU utilization
        num_workers=8,          # tune
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
