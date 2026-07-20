# -*- coding: utf-8 -*-
"""
Channel-stack dataset for GroupCDL dictionary pretraining (latent-I2SB steps 1 and 2).

Thin wrapper over I2SBDataset that collapses its (x0, x1, cond, mask) tuple into the SINGLE
tensor train_denoiser expects, so the dictionaries are learned with the existing denoiser loop
and no changes to the model or the training code.

The stack is cat([ch0, cond]) -- the same channel order diffusion/i2sb.py:cdlnet_pred builds
with torch.cat([xt, cond], dim=1) -- so a dictionary learned here slots into the bridge without
re-indexing channels.

    ch0 = "bridge"  a random point on the x1 -> x0 bridge MEAN (step ~ U{0..n_points-1}), i.e.
                    the path x_t traverses. Use with cond_idx: [0, 1, 3] for the 4-channel joint
                    dictionary (step 1). Noiseless by construction: train_denoiser's awgn()
                    supplies the noise, so the sigma GroupCDL conditions on stays honest and
                    there is exactly one noise source.
    ch0 = "x0"      the target contrast (T1ce). Use with cond_idx: [] for the 1-channel T1ce
                    dictionary (step 2).
    ch0 = "x1"      the prior contrast (T1).

Config keys: every I2SBDataset key (root/manifest, x0_idx, x1_idx, cond_idx, scales, image_key,
center_crop, crop_size, random_flips) plus

    ch0                                     "bridge" | "x0" | "x1"
    bridge_type, n_points, bridge_shape, beta_max
                                            read only when ch0 == "bridge". Mirror your i2sb
                                            config so the pretraining interpolants match the
                                            bridge the dictionary later sees.

There is deliberately NO tau knob here. tau enters bridge_schedule only as sum(betas) =
(2*tau)**2 -- a uniform scale on every beta -- while mu_x0/mu_x1 are ratios of cumsums, so it
cancels exactly and the interpolant mean is independent of it (verified on both shapes). tau sets
only the noise magnitude (std_sb / std_fwd), which this dataset does not use: awgn() in
train_denoiser is the single noise source. Set the noise level via the training block's
noise_std, which should span the bridge's std_fwd range [0, 2*tau].

beta_max is NOT dead and is kept: for bridge_type="i2sb" the betas are build_symmetric_betas
(NOT rescaled) with linear_start pinned at 1e-4 and linear_end = beta_max/interval, so beta_max
bends the ramp SHAPE and does move mu. For bridge_type="brownian" the profile is rescaled to
(2*tau)**2 and beta_max is inert, same as tau.

Both dictionaries must be built with the same GroupCDL M and sc, or the two latents will not
share a shape and step 3 cannot bridge between them.
"""

import torch

from torch.utils.data import Dataset

from datasets.BraTS.i2sb_dataset import I2SBDataset
from physics.bbridge import build_bridge, n_steps


class ContrastStackDataset(Dataset):
    def __init__(self, cfg):
        self.ds = I2SBDataset(cfg)          # indexing, h5 handles, scales, joint transforms
        self.ch0 = str(getattr(cfg, "ch0", "bridge"))
        if self.ch0 not in ("bridge", "x0", "x1"):
            raise ValueError(f"ch0 must be 'bridge', 'x0' or 'x1', got {self.ch0!r}")

        # bridge_sample: ch0 becomes a bridge SAMPLE (mean + std_sb(k)*noise) with CLEAN cond, and
        # the net conditions on std_fwd(k) -- aligning the dict with the I2SB bridge state it sees at
        # inference. False (default, legacy) = noiseless mean, and train_denoiser's awgn adds the noise.
        self.bridge_sample = bool(getattr(cfg, "bridge_sample", False))
        if self.ch0 == "bridge":
            # tau NOW matters when bridge_sample=True (it sets the absolute std_sb / std_fwd); it still
            # cancels out of mu_x0 / mu_x1, so the legacy noiseless-mean path is unaffected by it.
            sched = build_bridge(
                bridge_type=str(getattr(cfg, "bridge_type", "brownian")),
                n_points=int(getattr(cfg, "n_points", 1000)),
                device="cpu",
                tau=float(getattr(cfg, "tau", 0.19)),
                shape=str(getattr(cfg, "bridge_shape", "constant")),
                beta_max=float(getattr(cfg, "beta_max", 0.3)),
            )
            # mu_x0 + mu_x1 == 1 (Gaussian-product coefficients), so this is a convex mix.
            self.mu_x0, self.mu_x1 = sched.mu_x0, sched.mu_x1
            self.std_sb, self.std_fwd = sched.std_sb, sched.std_fwd   # used only when bridge_sample
            self.n_points = n_steps(sched)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x0, x1, cond, _mask = self.ds[idx]

        if self.ch0 == "bridge":
            # torch RNG, not numpy: the DataLoader reseeds torch per worker but numpy's global
            # seed is shared, which would repeat the same steps across workers.
            k = int(torch.randint(self.n_points, (1,)))
            mean = self.mu_x0[k] * x0 + self.mu_x1[k] * x1           # noiseless bridge mean at step k
            if self.bridge_sample:
                # input: bridge SAMPLE in ch0 (noise on ch0 only) + CLEAN cond; target: the mean +
                # clean cond; the net conditions on std_fwd(k). Same step k for mean and noise.
                noisy_ch0 = mean + self.std_sb[k] * torch.randn_like(mean)
                noisy = torch.cat([noisy_ch0, cond], dim=0)          # network input
                clean = torch.cat([mean, cond], dim=0)               # denoising target
                sigma = self.std_fwd[k].reshape(1, 1, 1)             # conditioning sigma (I2SB fwd std)
                return noisy, clean, sigma
            ch0 = mean
        else:
            ch0 = x0 if self.ch0 == "x0" else x1

        return torch.cat([ch0, cond], dim=0)
