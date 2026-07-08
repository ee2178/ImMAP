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
    def _extract_tokens(self, features):
        """
        features : (B, C, H, W) feature maps.
        Returns (B, N, D) memory banks of patch tokens, row-major over patch
        positions — comparisons still happen strictly within each image's bank.
        """
        B, C, H, W = features.shape
        patch = 1 if self.partial_decoder else self.patch_size
        if patch == 1:
            tokens = features.reshape(B, C, H * W).transpose(1, 2).contiguous()   # (B, H*W, C)
        else:
            # (B, C, H, W) -> (B, C*patch*patch, L) -> (B, L, C*patch*patch)
            unf = F.unfold(features, kernel_size=patch, stride=patch)
            tokens = unf.transpose(1, 2).contiguous()                            # (B, L, C*patch^2)
        return tokens

    # ------------------------------------------------- per-image anchor masks
    def _valid_anchor_mask(self, labels, mask, B, N, Hc, Wc, device, dtype):
        """(B, N) float mask of patches eligible to be anchors, one row per image.

        * use_mask_sampling : the generator's sampling mask (foreground anchors).
        * otherwise         : interior patches, excluding a border of exclude_border
                              (shared across the batch).
        """
        if self.use_mask_sampling:
            return (mask.reshape(B, N) > 0).to(dtype)
        b = self.exclude_border
        interior = torch.zeros(Hc, Wc, device=device, dtype=dtype)
        if 2 * b < Hc and 2 * b < Wc:
            interior[b:Hc - b, b:Wc - b] = 1.0
        else:
            interior[:] = 1.0
        return interior.reshape(1, N).expand(B, N)

    # ----------------------------------------------------------------- forward
    def forward(self, features, y_true):
        """
        features : (B, C, H, W)
        y_true   : (B, Hc, Wc, K)  [..., 0] = labels, [..., -1] = sampling mask
        Returns scalar loss averaged over the batch.

        Fully batched: the former per-image Python loop is replaced by (B, A, N)
        tensor ops, where A = num_samples_loss_eval anchors are sampled per image
        (padded when an image has fewer eligible patches and masked out via
        `anchor_valid`). The result is identical to the per-image loop: each image's
        loss is the mean over its anchors-with-positives, and the batch loss is the
        sum of per-image losses divided by B (degenerate images contribute 0).
        """
        if features.dim() != 4:
            raise ValueError("features must be (B, C, H, W)")
        device, dtype = features.device, features.dtype
        B = features.shape[0]
        labels = y_true[..., 0]                          # (B, Hc, Wc)
        Hc, Wc = labels.shape[1], labels.shape[2]
        labels = labels.reshape(B, -1)                   # (B, N)
        N = labels.shape[1]
        mask = y_true[..., -1] if self.use_mask_sampling else None

        tokens = self._extract_tokens(features)          # (B, N, D)
        if tokens.shape[1] != N:
            raise ValueError(f"constraint labels ({N}) do not match token count "
                             f"({tokens.shape[1]}); check patch_size / partial_decoder "
                             f"vs the constraint-map downsampling.")

        # guard: need more than topk+1 patches to form a neighborhood (matches TF)
        if N <= self.topk + 1:
            return features.new_zeros(())

        # ---- anchor selection (sync-free): top-A of a randomized key over eligible
        # patches. Eligible patches all outrank padding; ties among them are broken
        # at random, so >A eligible -> a random A subset; <A eligible -> all of them
        # plus padding flagged by anchor_valid=False.
        A = min(self.num_samples_loss_eval, N)
        eligible = self._valid_anchor_mask(labels, mask, B, N, Hc, Wc, device, dtype)
        key = eligible * (1.0 + torch.rand(B, N, device=device, dtype=dtype))
        anchor_idx = torch.topk(key, k=A, dim=1).indices                 # (B, A)
        anchor_valid = torch.gather(eligible, 1, anchor_idx) > 0.5       # (B, A) bool

        # cosine similarity of every anchor to every token in its own image: (B, A, N)
        tok_n = F.normalize(tokens, dim=2)                               # (B, N, D)
        D = tok_n.shape[2]
        anc_n = torch.gather(tok_n, 1, anchor_idx.unsqueeze(-1).expand(B, A, D))
        sim = torch.bmm(anc_n, tok_n.transpose(1, 2))                    # (B, A, N)
        probs = torch.exp(sim / self.temperature)                       # (B, A, N)

        # background neighbors = top-(topk+1) most similar tokens per anchor
        k = min(self.topk + 1, N)
        topk_idx = torch.topk(sim, k=k, dim=2).indices                  # (B, A, k)
        bg = torch.zeros_like(sim, dtype=torch.bool)
        bg.scatter_(2, topk_idx, True)

        # same-cluster mask
        anc_labels = torch.gather(labels, 1, anchor_idx)                # (B, A)
        same = labels.unsqueeze(1) == anc_labels.unsqueeze(2)          # (B, A, N)

        if self.exclude_self:
            self_mask = torch.zeros_like(bg)
            self_mask.scatter_(2, anchor_idx.unsqueeze(2), True)
            bg = bg & (~self_mask)

        pos_mask = bg & same                                           # (B, A, N)
        neg_mask = bg & (~same)                                        # (B, A, N)

        neg_sum = (probs * neg_mask).sum(dim=2)                        # (B, A)

        if self.contrastive_loss_type == 2:
            # pairwise (recommended): per positive p -> p / (p + sum_neg)
            denom = probs + neg_sum.unsqueeze(2)                       # (B, A, N)
            rel = probs / denom
            log_rel = torch.log(rel + self.eps) * pos_mask
            n_pos = pos_mask.sum(dim=2).clamp_min(1)
            per_anchor = -(log_rel.sum(dim=2) / n_pos)                 # (B, A)
        elif self.contrastive_loss_type == 1:
            # setwise: sum_pos / (sum_pos + sum_neg)
            pos_sum = (probs * pos_mask).sum(dim=2)                    # (B, A)
            rel = pos_sum / (pos_sum + neg_sum)
            per_anchor = -torch.log(rel + self.eps)                   # (B, A)
        else:
            raise ValueError("contrastive_loss_type must be 1 (setwise) or 2 (pairwise)")

        # an anchor counts only if it is a real (non-padded) anchor with >=1 positive
        contributing = (pos_mask.any(dim=2) & anchor_valid).to(dtype)  # (B, A)
        n_anchors = contributing.sum(dim=1).clamp_min(1.0)            # (B,)
        per_image = (per_anchor * contributing).sum(dim=1) / n_anchors  # (B,)
        return per_image.sum() / B


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

