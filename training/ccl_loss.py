# -*- coding: utf-8 -*-
"""
Constrained Contrastive Loss (CCL) for MR-contrast guided contrastive learning — PyTorch.

Faithful port of the TensorFlow `lossObj` (Umapathy et al., Med. Phys. 2024).

For each randomly sampled anchor patch in an image, positives/negatives are the
INTERSECTION of two criteria:
  1) the top-(topk+1) most similar patches in representation space (cosine similarity),
  2) split by whether they share the anchor's constraint-map cluster label:
       same label  -> positive,   different label -> negative.
The memory bank is built PER IMAGE, so all comparisons are within a single image.

Inputs to forward():
  features : (B, C, H, W)  network representation  Psi(x)
  y_true   : (B, Hc, Wc, K) constraint tensor from the data generator, where
                 y_true[..., 0]  = constraint-map cluster labels (already at patch res)
                 y_true[..., -1] = sampling mask (1 = use this patch as an anchor),
                                   present only when use_mask_sampling=True.

Notes on resolution:
  * partial_decoder=True forces patch_size=1: the feature map is already at the
    constraint-map resolution (a 1x1 token there == 4x4 in the full decoder).
  * patch_size>1 unfolds the full-res feature map into non-overlapping patch tokens
    (concatenated features), matching tf.image.extract_patches in the original.
  In both cases the number of tokens must equal Hc*Wc (the generator downsampled the
  constraint map by patch_size via majority vote), so the labels line up token-for-token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedContrastiveLoss(nn.Module):
    def __init__(self,
                 patch_size=1,
                 topk=100,
                 num_samples_loss_eval=100,
                 temperature=0.1,
                 contrastive_loss_type=2,   # 2 = pairwise (recommended), 1 = setwise
                 use_mask_sampling=True,
                 partial_decoder=False,
                 exclude_border=3,          # only used when use_mask_sampling=False
                 exclude_self=False,        # TF reference keeps self (False)
                 eps=1e-7):
        super().__init__()
        self.patch_size = patch_size
        self.topk = topk
        self.num_samples_loss_eval = num_samples_loss_eval
        self.temperature = temperature
        self.contrastive_loss_type = contrastive_loss_type
        self.use_mask_sampling = use_mask_sampling
        self.partial_decoder = partial_decoder
        self.exclude_border = exclude_border
        self.exclude_self = exclude_self
        self.eps = eps

    # ------------------------------------------------------------------ tokens
    def _extract_tokens(self, ft_img):
        """
        ft_img : (C, H, W) feature map for one image.
        Returns (N, D) memory bank of patch tokens, row-major over patch positions.
        """
        C, H, W = ft_img.shape
        patch = 1 if self.partial_decoder else self.patch_size
        if patch == 1:
            tokens = ft_img.reshape(C, H * W).transpose(0, 1).contiguous()   # (H*W, C)
        else:
            # (1, C, H, W) -> (1, C*patch*patch, L) -> (L, C*patch*patch)
            unf = F.unfold(ft_img.unsqueeze(0), kernel_size=patch, stride=patch)
            tokens = unf.squeeze(0).transpose(0, 1).contiguous()             # (L, C*patch^2)
        return tokens

    # ------------------------------------------------------- per-image anchors
    def _anchor_indices(self, labels, mask, num_patches, device):
        """Return 1-D LongTensor of anchor indices into the flattened token grid."""
        if self.use_mask_sampling:
            idx = torch.nonzero(mask.reshape(-1) > 0, as_tuple=False).squeeze(-1)
            if idx.numel() > 1:
                idx = idx[torch.randperm(idx.numel(), device=device)]
            return idx
        # no mask: random interior patches, excluding a border of `exclude_border`
        b = self.exclude_border
        grid = torch.arange(num_patches * num_patches, device=device).reshape(num_patches, num_patches)
        if 2 * b < num_patches:
            grid = grid[b:-b, b:-b]
        flat = grid.reshape(-1)
        n = min(self.num_samples_loss_eval, flat.numel())
        sel = torch.randperm(flat.numel(), device=device)[:n]
        return flat[sel]

    # --------------------------------------------------------- per-image loss
    def _image_loss(self, ft_img, labels, mask):
        device = ft_img.device
        tokens = self._extract_tokens(ft_img)            # (N, D)
        N = tokens.shape[0]
        labels = labels.reshape(-1)                      # (N,)

        if labels.shape[0] != N:
            raise ValueError(f"constraint labels ({labels.shape[0]}) do not match "
                             f"token count ({N}); check patch_size / partial_decoder "
                             f"vs the constraint-map downsampling.")

        # guard: need more than topk+1 patches to form a neighborhood (matches TF)
        if N <= self.topk + 1:
            return ft_img.new_zeros(())

        anchor_idx = self._anchor_indices(labels, mask, int(round(N ** 0.5)), device)
        if self.use_mask_sampling:
            anchor_idx = anchor_idx[: self.num_samples_loss_eval]
        if anchor_idx.numel() == 0:
            return ft_img.new_zeros(())

        # cosine similarity of every anchor to every token: (A, N)
        tok_n = F.normalize(tokens, dim=-1)
        anc_n = tok_n[anchor_idx]                        # (A, D), already normalized rows
        sim = anc_n @ tok_n.transpose(0, 1)              # (A, N)
        probs = torch.exp(sim / self.temperature)        # (A, N)

        # background neighbors = top-(topk+1) most similar tokens per anchor
        k = min(self.topk + 1, N)
        topk_idx = torch.topk(sim, k=k, dim=1).indices   # (A, k) — non-differentiable selection
        bg = torch.zeros_like(sim, dtype=torch.bool)
        bg.scatter_(1, topk_idx, True)

        # same-cluster mask
        anc_labels = labels[anchor_idx].unsqueeze(1)     # (A, 1)
        same = labels.unsqueeze(0) == anc_labels         # (A, N)

        if self.exclude_self:
            self_mask = torch.zeros_like(bg)
            self_mask.scatter_(1, anchor_idx.unsqueeze(1), True)
            bg = bg & (~self_mask)

        pos_mask = bg & same                             # (A, N)
        neg_mask = bg & (~same)                          # (A, N)

        neg_sum = (probs * neg_mask).sum(dim=1)          # (A,)

        if self.contrastive_loss_type == 2:
            # pairwise (recommended): per positive p -> p / (p + sum_neg)
            denom = probs + neg_sum.unsqueeze(1)         # (A, N)
            rel = probs / denom
            log_rel = torch.log(rel + self.eps) * pos_mask
            n_pos = pos_mask.sum(dim=1).clamp_min(1)
            per_anchor = -(log_rel.sum(dim=1) / n_pos)   # (A,)
        elif self.contrastive_loss_type == 1:
            # setwise: sum_pos / (sum_pos + sum_neg)
            pos_sum = (probs * pos_mask).sum(dim=1)      # (A,)
            rel = pos_sum / (pos_sum + neg_sum)
            per_anchor = -torch.log(rel + self.eps)      # (A,)
        else:
            raise ValueError("contrastive_loss_type must be 1 (setwise) or 2 (pairwise)")

        # only count anchors that actually have positives (matches having self/same-label)
        valid = pos_mask.any(dim=1)
        if valid.any():
            return per_anchor[valid].mean()
        return ft_img.new_zeros(())

    # ----------------------------------------------------------------- forward
    def forward(self, features, y_true):
        """
        features : (B, C, H, W)
        y_true   : (B, Hc, Wc, K)  [..., 0] = labels, [..., -1] = sampling mask
        Returns scalar loss averaged over the batch.
        """
        if features.dim() != 4:
            raise ValueError("features must be (B, C, H, W)")
        B = features.shape[0]
        labels = y_true[..., 0]                          # (B, Hc, Wc)
        mask = y_true[..., -1] if self.use_mask_sampling else None

        total = features.new_zeros(())
        for b in range(B):
            m = mask[b] if mask is not None else None
            total = total + self._image_loss(features[b], labels[b], m)
        return total / B


# Convenience: build the loss from a config object with matching attribute names.
def build_ccl_loss(cfg):
    return ConstrainedContrastiveLoss(
        patch_size=getattr(cfg, "patch_size", 1),
        topk=getattr(cfg, "topk", 100),
        num_samples_loss_eval=getattr(cfg, "num_samples_loss_eval", 100),
        temperature=getattr(cfg, "temperature", 0.1),
        contrastive_loss_type=getattr(cfg, "contrastive_loss_type", 2),
        use_mask_sampling=getattr(cfg, "use_mask_sampling", True),
        partial_decoder=getattr(cfg, "partial_decoder", False),
        eps=getattr(cfg, "epsilon", 1e-7),
    )


if __name__ == "__main__":
    # Smoke test (runs only where torch is installed).
    torch.manual_seed(0)
    B, C, H, W = 2, 8, 16, 16          # partial_decoder-style: features already at cmap res
    feats = torch.randn(B, C, H, W, requires_grad=True)
    labels = torch.randint(0, 6, (B, H, W, 1)).float()
    mask = torch.zeros(B, H, W, 1)
    for b in range(B):                 # 30 random foreground anchors per image
        flat = torch.randperm(H * W)[:30]
        mask[b].view(-1)[flat] = 1
    y_true = torch.cat([labels, mask], dim=-1)

    loss_fn = ConstrainedContrastiveLoss(patch_size=1, topk=20,
                                         num_samples_loss_eval=30, temperature=0.1,
                                         contrastive_loss_type=2, use_mask_sampling=True,
                                         partial_decoder=True)
    loss = loss_fn(feats, y_true)
    loss.backward()
    print("loss:", float(loss), "| grad finite:", bool(torch.isfinite(feats.grad).all()))
