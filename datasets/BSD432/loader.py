import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets.registry import register_loader

# ============================================================
# Config
# ============================================================

BSD432_PATHS = {
    "root": "../../datasets/BSD432",
    "scale_fac": 1.0,   # ToTensor already maps to [0, 1]; keep at 1.0 by default
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ============================================================
# Dataset
# ============================================================

class BSD432Dataset(Dataset):
    """
    Natural-image dataset (BSD432) for denoising.

    Always returns a single image tensor in [0, 1] (scaled by `scale_fac`),
    shaped [C, H, W] with C = 1 (grayscale, default) or C = 3 (color).
    """

    def __init__(
        self,
        color=False,            # False -> grayscale (default), True -> RGB
        ### Transformation parameters (mirrors the FastMRI denoising branch)
        crop_size=None,
        center_crop=None,
        random_flips=True,
        ### Paths / scaling
        root=None,
        scale_fac=None,
    ):

        self.root = root or BSD432_PATHS["root"]
        self.scale_fac = (
            scale_fac if scale_fac is not None else BSD432_PATHS["scale_fac"]
        )
        self.color = color

        # ----------------------------------------------------
        # Build file list
        # ----------------------------------------------------
        self.files = self._build_file_list()

        if len(self.files) == 0:
            raise ValueError(f"No images found in {self.root}")

        # ----------------------------------------------------
        # Transform pipeline (applied on PIL images).
        # ToTensor() is appended last so the output is [0, 1], [C, H, W].
        # ----------------------------------------------------
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

        tfms.append(transforms.ToTensor())   # PIL -> float tensor in [0, 1]

        self.transform = transforms.Compose(tfms)

    def _build_file_list(self):
        files = [
            f for f in os.listdir(self.root)
            if f.lower().endswith(IMG_EXTS)
        ]
        return sorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = os.path.join(self.root, self.files[idx])

        # Load and force the channel layout we want.
        img = Image.open(path)
        img = img.convert("RGB" if self.color else "L")

        # transform -> [0, 1] tensor of shape [C, H, W]
        image = self.transform(img)

        return image * self.scale_fac


# ============================================================
# Loader
# ============================================================

@register_loader("BSD432")
def get_bsd432_loader(
    color=False,
    crop_size=None,
    center_crop=None,
    random_flips=True,
    batch_size=1,
    shuffle=True,
    root=None,
    scale_fac=None,
    drop_last=True,
):
    dataset = BSD432Dataset(
        color=color,
        crop_size=crop_size,
        center_crop=center_crop,
        random_flips=random_flips,
        root=root,
        scale_fac=scale_fac,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        # Same GPU-utilization defaults as the FastMRI loader
        num_workers=8,          # tune
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
