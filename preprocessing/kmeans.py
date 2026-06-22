from __future__ import annotations

import torch
from typing import Union

@torch.no_grad()
def _kmeanspp_init(x, K, gen):
    """k-means++ seeding on (M, D) -> (K, D)."""
    M = x.shape[0]
    centroids = torch.empty(K, x.shape[1], device=x.device, dtype=x.dtype)
    centroids[0] = x[torch.randint(0, M, (1,), generator=gen, device=x.device)]
    closest = torch.cdist(x, centroids[:1]).squeeze(1).pow(2)
    for i in range(1, K):
        probs = closest / closest.sum().clamp(min=1e-12)
        centroids[i] = x[torch.multinomial(probs, 1, generator=gen)]
        closest = torch.minimum(closest, torch.cdist(x, centroids[i:i+1]).squeeze(1).pow(2))
    return centroids


@torch.no_grad()
def minibatch_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    batch_size: int = 1024,
    max_iter: int = 100,
    tol: float = 0.0,
    max_no_improvement: int | None = 10,
    n_init: int = 3,
    init_size: int | None = None,
    reassignment_ratio: float = 0.01,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-compatible Mini-Batch K-Means, following sklearn.cluster.MiniBatchKMeans.

    Args mirror the sklearn names. Returns (labels (N,), centroids (K, D)).
    """
    N, D = x.shape
    K = n_clusters
    device, dtype = x.device, x.dtype
    bs = min(batch_size, N)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    if init_size is None:
        init_size = 3 * bs if 3 * bs >= K else 3 * K
    init_size = min(init_size, N)

    # --- multiple inits on a random subset, keep best by inertia on a validation subset ---
    valid_idx = torch.randint(0, N, (init_size,), generator=gen, device=device)
    x_valid = x[valid_idx]
    best_centroids, best_inertia = None, None
    for _ in range(n_init):
        sub = x[torch.randint(0, N, (init_size,), generator=gen, device=device)]
        c = _kmeanspp_init(sub, K, gen)
        inertia = torch.cdist(x_valid, c).amin(dim=1).pow(2).sum()
        if best_inertia is None or inertia < best_inertia:
            best_centroids, best_inertia = c, inertia
    centroids = best_centroids

    # --- streaming optimization ---
    counts = torch.zeros(K, device=device, dtype=dtype)      # running weight per center
    n_steps = (max_iter * N) // bs
    ewa = ewa_min = None
    no_improve = 0
    since_reassign = 0
    alpha = min(bs * 2.0 / (N + 1), 1.0)                     # EWA smoothing factor

    for step in range(n_steps):
        idx = torch.randint(0, N, (bs,), generator=gen, device=device)
        batch = x[idx]

        dists = torch.cdist(batch, centroids)                # (bs, K)
        labels = dists.argmin(dim=1)
        batch_inertia = dists.gather(1, labels[:, None]).squeeze(1).pow(2).sum() / bs

        # per-center batch sums and counts
        bcounts = torch.zeros(K, device=device, dtype=dtype)
        bcounts.index_add_(0, labels, torch.ones(bs, device=device, dtype=dtype))
        bsums = torch.zeros(K, D, device=device, dtype=dtype)
        bsums.index_add_(0, labels, batch)

        # incremental-mean update with decaying learning rate eta = 1 / new_count
        old = centroids.clone()
        active = bcounts > 0
        new_counts = counts + bcounts
        eta = torch.where(active, 1.0 / new_counts.clamp(min=1), torch.zeros_like(counts))
        centroids = centroids + eta.unsqueeze(1) * (bsums - bcounts.unsqueeze(1) * centroids)
        counts = new_counts

        # --- periodic reassignment of low-count centers ---
        since_reassign += bs
        if (counts == 0).any() or since_reassign >= 10 * K:
            since_reassign = 0
            to_reassign = counts < reassignment_ratio * counts.max().clamp(min=1)
            n_re = int(to_reassign.sum())
            if 0 < n_re <= bs // 2:
                # pick batch points far from their nearest center (distance-weighted)
                d = torch.cdist(batch, centroids).amin(dim=1).pow(2)
                pick = torch.multinomial(d / d.sum().clamp(min=1e-12), n_re, generator=gen)
                centroids[to_reassign] = batch[pick]
                counts[to_reassign] = 0

        # --- convergence checks (skip first step: it's just init inertia) ---
        if step == 0:
            continue
        if tol > 0.0 and (centroids - old).pow(2).sum() <= tol:
            break
        ewa = batch_inertia if ewa is None else ewa * (1 - alpha) + batch_inertia * alpha
        if ewa_min is None or ewa < ewa_min:
            ewa_min, no_improve = ewa, 0
        else:
            no_improve += 1
        if max_no_improvement is not None and no_improve >= max_no_improvement:
            break

    # final hard assignment over all points (sklearn computes labels_ at the end)
    labels = torch.cdist(x, centroids).argmin(dim=1)
    return labels, centroids


@torch.no_grad()
def kmeans(
    x: torch.Tensor,
    n_clusters: int,
    n_iters: int = 100,
    tol: float = 1e-4,
    seed: Union[int, None] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-compatible k-means with k-means++ initialization.

    Args:
        x:          Input tensor of shape (N, D) — N observations, D features.
        n_clusters: Number of clusters K.
        n_iters:    Maximum Lloyd iterations.
        tol:        Convergence threshold on centroid shift (mean squared movement).
        seed:       Optional RNG seed for reproducible init.

    Returns:
        labels:    Shape (N,)        — cluster index for each point (long tensor).
        centroids: Shape (K, D)      — final cluster centers.
    """
    N, D = x.shape
    K = n_clusters
    device, dtype = x.device, x.dtype

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    # --- k-means++ initialization (all on-device) ---
    centroids = torch.empty(K, D, device=device, dtype=dtype)
    first = torch.randint(0, N, (1,), generator=gen, device=device)
    centroids[0] = x[first]

    # Squared distance from every point to the nearest chosen centroid so far
    closest_sq = torch.cdist(x, centroids[:1]).squeeze(1).pow(2)  # (N,)
    for i in range(1, K):
        probs = closest_sq / closest_sq.sum().clamp(min=1e-12)
        idx = torch.multinomial(probs, 1, generator=gen)          # (1,)
        centroids[i] = x[idx]
        new_sq = torch.cdist(x, centroids[i:i+1]).squeeze(1).pow(2)
        closest_sq = torch.minimum(closest_sq, new_sq)

    # --- Lloyd iterations ---
    ones = torch.ones(N, device=device, dtype=dtype)
    for _ in range(n_iters):
        # Assign: (N, K) distances -> nearest centroid
        dists = torch.cdist(x, centroids)            # (N, K)
        labels = dists.argmin(dim=1)                 # (N,)

        # Update: vectorized mean per cluster via scatter-accumulate (no K-loop)
        counts = torch.zeros(K, device=device, dtype=dtype)
        counts.index_add_(0, labels, ones)           # (K,)
        sums = torch.zeros(K, D, device=device, dtype=dtype)
        sums.index_add_(0, labels, x)                # (K, D)

        new_centroids = sums / counts.clamp(min=1).unsqueeze(1)

        # Keep old centroid for any empty cluster (avoids NaNs / drift)
        empty = counts == 0
        if empty.any():
            new_centroids[empty] = centroids[empty]

        shift = (new_centroids - centroids).pow(2).mean()
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids
