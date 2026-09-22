import torch
import torch.fft as fft
import torch.nn.functional as F
import math
from typing import Tuple

### Sensitivity Map Estimation

def walsh(y: torch.Tensor, ks: int = 5, stride: int = 2):
    """
    Computes coil sensitivity maps using the Walsh method.
    Args:
        y: complex tensor of shape (B, C, H, W)
        ks: patch size
        stride: patch stride
    Returns:
        smaps: sensitivity maps of shape (B, C, H, W)
    """
    B, C, H, W = y.shape

    # Handle unfolding for complex tensors
    unfolded_real = F.unfold(y.real, kernel_size=(ks, ks), stride=stride)
    unfolded_imag = F.unfold(y.imag, kernel_size=(ks, ks), stride=stride)
    unfolded = torch.complex(unfolded_real, unfolded_imag)  # (B, C*ks*ks, Npatch)
    Npatch = unfolded.shape[-1]

    # (B, Npatch, C, ks*ks)
    Yp = unfolded.view(B, C, ks*ks, Npatch).permute(0, 3, 2, 1)  # (B, Npatch, ks*ks, C)

    # Covariance per patch
    X = torch.matmul(Yp.transpose(-1, -2).conj(), Yp)  # (B, Npatch, C, C)

    # SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = U[..., 0]  # (B, Npatch, C)

    # Reference coil alignment per batch
    power = y.abs().pow(2).sum(dim=(2, 3))  # (B, C)
    Cref = power.argmax(dim=1)
    for b in range(B):
        ref = Q[b, :, Cref[b]]
        Q[b] *= ref.conj().sgn().unsqueeze(-1)

    # Reshape to low-res maps
    Hp = (H - ks) // stride + 1
    Wp = (W - ks) // stride + 1
    smaps_p = Q.permute(0, 2, 1).reshape(B, C, Hp, Wp)

    # Upsample to full size
    smaps_real = F.interpolate(smaps_p.real, size=(H, W), mode='bilinear', align_corners=False)
    smaps_imag = F.interpolate(smaps_p.imag, size=(H, W), mode='bilinear', align_corners=False)
    smaps = torch.complex(smaps_real, smaps_imag)

    # Normalize
    norm = smaps.abs().pow(2).sum(dim=1, keepdim=True)
    smaps /= (norm.sqrt() + 1e-8)
    return smaps.conj()

def espirit(
    kspace: torch.Tensor,  # (B, C, Nx, Ny)
    acs_size: Tuple[int, int] = (24, 24),
    kernel_size: int = 8,
    thresh_rowspace: float = 0.05,
    thresh_eig: float = 0.95,
    rtol: float = 1e-3,
    maxit: int = 100,
    block_size: int = 8192,
):
    B, C, Nx, Ny = kspace.shape
    ks = kernel_size
    device, dtype = kspace.device, kspace.dtype
    # 1) ACS extraction
    ax, ay = acs_size
    cx, cy = Nx // 2, Ny // 2
    kspace_acs = kspace[:, :, cx - ax // 2 : cx + ax // 2, cy - ay // 2 : cy + ay // 2]

    # 2) Hankel matrix (patches)
    patches = (
        kspace_acs.permute(0, 2, 3, 1)
        .unfold(1, ks, 1)
        .unfold(2, ks, 1)
        .reshape(B, -1, ks * ks * C)
    )

    # 3) SVD
    _, S, Vh = torch.linalg.svd(patches, full_matrices=False)
    V = Vh.conj().transpose(-2, -1)

    # 4) Row-space truncation (vectorized; 1 sync total)
    #
    # PER SAMPLE, THOUGH THE STACK IS SHARED. `counts` differs from slice to
    # slice, so slicing everything to `counts.max()` gave every slice the
    # WIDEST retention in its batch -- and retaining extra basis vectors can
    # only raise the eigenvalue map (keep the complete basis and lam == 1
    # everywhere), so those slices came out with a DILATED support. It also
    # made the maps depend on which slices shared a chunk, i.e. on
    # make_espirit_smaps.py's `--chunk`, which is meant to be a pure memory
    # knob.
    #
    # Zeroing the tail is exactly equivalent to dropping it: every downstream
    # use of V is a sum over kernels (lam = sum_k |v_k^H q|^2 / ks^2, and
    # G = sum_k v_k v_k^H), so a zero kernel contributes nothing. Only the
    # padded width is shared now, not the retention.
    counts = (S >= thresh_rowspace * S[:, :1]).sum(dim=1)   # (B,)
    Nbasis = int(counts.max().clamp_min(1).item())
    V = V[:, :, :Nbasis]
    keep = (torch.arange(Nbasis, device=device)[None, :]
            < counts.clamp_min(1)[:, None])                # (B, Nbasis)
    V = V * keep[:, None, :]
    # 5) Kernels → image domain
    Vkernel = (
        V.permute(0, 2, 1)
        .reshape(B, Nbasis, C, ks, ks)
        .permute(0, 2, 1, 3, 4)
    )  # (B, C, K, ks, ks)

    # Avoid explicit pad allocation: implicit zero-pad via s=(Nx,Ny)
    # FLIP the kernels first. The rows of V are CORRELATION kernels (the
    # Hankel patches are raster-ordered k-space neighbourhoods), and the
    # image-domain operator is their CONVOLUTION -- k(-n), not k(n). Without
    # the flip every map comes out point-reflected through the FOV centre:
    # on a phantom with known maps, coherence with the truth was 0.54 and
    # the SENSE model residual 0.82; with it, 1.000 and 0.008, on odd and
    # even grids alike. (Flipping the OUTPUT instead is a pixel off on even
    # grids -- the reflection is about the fftshift centre.)
    Vk = torch.fft.ifft2(Vkernel.flip(-2, -1), s=(Nx, Ny), dim=(-2, -1))
    # Scale IN PLACE. `fftshift(Vk) * c` held three full-grid copies at once --
    # the ifft output, the shift and the product -- and this tensor, coils x
    # kernels x Nx x Ny, is the memory peak of the whole function.
    Vk = torch.fft.fftshift(Vk, dim=(-2, -1))
    Vk.mul_(Nx * Ny)

    # Vk: (B, C, K, Nx, Ny) -> (B, P, C, K)
    Vk = Vk.reshape(B, C, Nbasis, -1).permute(0, 3, 1, 2)
    P = Vk.shape[1]

    # 6) Power method
    Q = torch.randn(B, P, C, device=device, dtype=dtype)
    Q = Q / (Q.norm(dim=-1, keepdim=True) + 1e-12)

    for _ in range(maxit):
        Q_new = torch.empty_like(Q)

        for p0 in range(0, P, block_size):
            p1 = min(p0 + block_size, P)
            Vblk = Vk[:, p0:p1]   # (B, Pb, C, K)
            Qblk = Q[:, p0:p1]    # (B, Pb, C)

            tmp = torch.einsum("bpck,bpc->bpk", Vblk.conj(), Qblk)
            Q_new[:, p0:p1] = torch.einsum("bpck,bpk->bpc", Vblk, tmp)

        Q_new = Q_new / (Q_new.norm(dim=-1, keepdim=True) + 1e-12)

        # still a sync (Python if), but only once per iter
        if (Q_new - Q).norm() < rtol:
            Q = Q_new
            break
        Q = Q_new

    # 7) Eigenvalue estimate
    lam = torch.empty(B, P, device=device, dtype=torch.float32)

    for p0 in range(0, P, block_size):
        p1 = min(p0 + block_size, P)
        Vblk = Vk[:, p0:p1]
        Qblk = Q[:, p0:p1]

        tmp = torch.einsum("bpck,bpc->bpk", Vblk.conj(), Qblk)
        lam[:, p0:p1] = (tmp.abs() ** 2).sum(dim=-1).real / (ks ** 2)

    lam = lam.reshape(B, Nx, Ny)
    Q = Q.reshape(B, Nx, Ny, C)

    # 8) Threshold & normalize
    mask = lam > thresh_eig
    smaps = (mask[..., None] * Q).permute(0, 3, 1, 2).conj()

    # PHASE REFERENCE: THE STRONGEST COIL, not coil 0.
    #
    # Maps are defined only up to a per-pixel phase, and this fixes it by
    # rotating every map so the reference coil's is real and positive. That
    # choice propagates: the ground truth the nets are trained on is
    # `x = sum_c conj(s_c) c_c`, so wherever the reference coil is DARK its
    # phase is noise and x inherits it. A complex-valued unrolled net then has
    # to reproduce noise; E2E-VarNet, which outputs RSS magnitude, never sees
    # it -- so a bad reference penalises one side of that comparison only.
    #
    # `smaps[:, :1]` (coil 0, a fixed element of the array that is dark over
    # much of the head) was a porting slip. Sljiva picks the coil with the most
    # ACS energy -- `src/solver.jl`, `Cref = argmax(...)` -- and so does
    # `walsh` above, which is why those two agreed with each other and this did
    # not.
    #
    # PER SAMPLE, where Sljiva sums the energy over the batch too. Sharing one
    # reference across a batch would make a slice's maps depend on which slices
    # it was chunked with, which is exactly the coupling the row-space mask
    # above exists to avoid; the two differ only when slices of one volume
    # disagree on their strongest coil.
    energy = kspace_acs.abs().pow(2).sum(dim=(-2, -1))          # (B, C)
    cref = energy.argmax(dim=1)                                 # (B,)
    ref = smaps[torch.arange(B, device=device), cref][:, None]  # (B, 1, Nx, Ny)
    smaps = smaps * (ref / (ref.abs() + 1e-12)).conj()
    return smaps

def espirit_soft(
    kspace: torch.Tensor,  # (B, C, Nx, Ny)
    acs_size: Tuple[int, int] = (24, 24),
    kernel_size: int = 8,
    thresh_rowspace: float = 0.05,
    thresh_eig: float = 0.9,
    num_maps: int = 2,
    block_size: int = 8192,
):
    B, C, Nx, Ny = kspace.shape
    ks = kernel_size
    M = num_maps
    device, dtype = kspace.device, kspace.dtype
    assert 1 <= M <= C, "num_maps must be in [1, C]"

    # 1) ACS extraction
    ax, ay = acs_size
    cx, cy = Nx // 2, Ny // 2
    kspace_acs = kspace[:, :, cx - ax // 2 : cx + ax // 2, cy - ay // 2 : cy + ay // 2]

    # 2) Hankel matrix (patches)
    patches = (
        kspace_acs.permute(0, 2, 3, 1)
        .unfold(1, ks, 1)
        .unfold(2, ks, 1)
        .reshape(B, -1, ks * ks * C)
    )

    # 3) SVD
    _, S, Vh = torch.linalg.svd(patches, full_matrices=False)
    V = Vh.conj().transpose(-2, -1)

    # 4) Row-space truncation (vectorized; 1 sync total)
    # Per sample, with the tail ZEROED rather than the stack sliced to the
    # batch maximum -- see `espirit` for why sharing the retention dilates the
    # support of every slice but the widest one. G is a sum over kernels, so a
    # zero kernel contributes nothing.
    counts = (S >= thresh_rowspace * S[:, :1]).sum(dim=1)   # (B,)
    Nbasis = int(counts.max().clamp_min(1).item())
    V = V[:, :, :Nbasis]
    keep = (torch.arange(Nbasis, device=device)[None, :]
            < counts.clamp_min(1)[:, None])                # (B, Nbasis)
    V = V * keep[:, None, :]

    # 5) Kernels → image domain
    Vkernel = (
        V.permute(0, 2, 1)
        .reshape(B, Nbasis, C, ks, ks)
        .permute(0, 2, 1, 3, 4)
    )  # (B, C, K, ks, ks)
    # FLIP the kernels first. The rows of V are CORRELATION kernels (the
    # Hankel patches are raster-ordered k-space neighbourhoods), and the
    # image-domain operator is their CONVOLUTION -- k(-n), not k(n). Without
    # the flip every map comes out point-reflected through the FOV centre:
    # on a phantom with known maps, coherence with the truth was 0.54 and
    # the SENSE model residual 0.82; with it, 1.000 and 0.008, on odd and
    # even grids alike. (Flipping the OUTPUT instead is a pixel off on even
    # grids -- the reflection is about the fftshift centre.)
    Vk = torch.fft.ifft2(Vkernel.flip(-2, -1), s=(Nx, Ny), dim=(-2, -1))
    Vk = torch.fft.fftshift(Vk, dim=(-2, -1)) * (Nx * Ny)
    # Vk: (B, C, K, Nx, Ny) -> (B, P, C, K)
    Vk = Vk.reshape(B, C, Nbasis, -1).permute(0, 3, 1, 2)
    P = Vk.shape[1]

    # 6) Pointwise Hermitian eigendecomposition (replaces power method)
    #    G_q = (1/ks^2) Vk Vk^H is C×C, PSD; eigh gives ALL eigenpairs.
    maps = torch.empty(B, P, C, M, device=device, dtype=dtype)
    lams = torch.empty(B, P, M, device=device, dtype=torch.float32)
    for p0 in range(0, P, block_size):
        p1 = min(p0 + block_size, P)
        Vblk = Vk[:, p0:p1]                                      # (B, Pb, C, K)
        Gq = torch.einsum("bpck,bpdk->bpcd",
                          Vblk, Vblk.conj()) / (ks ** 2)         # (B, Pb, C, C)
        w, v = torch.linalg.eigh(Gq)                             # w ascending
        lams[:, p0:p1] = w[..., -M:].flip(-1)                    # [λ1, λ2, ...] desc
        maps[:, p0:p1] = v[..., -M:].flip(-1)                    # matching vectors

    maps = maps.reshape(B, Nx, Ny, C, M)
    lams = lams.reshape(B, Nx, Ny, M)

    # 7) Per-map threshold & normalize
    mask = (lams > thresh_eig)                                   # (B, Nx, Ny, M)
    smaps = maps.permute(0, 4, 3, 1, 2).conj()                   # (B, M, C, Nx, Ny)
    smaps = smaps * mask.permute(0, 3, 1, 2)[:, :, None]
    # Phase referenced to the STRONGEST COIL, per sample, as in `espirit` --
    # see the note there for why coil 0 put noise into the ground-truth phase.
    energy = kspace_acs.abs().pow(2).sum(dim=(-2, -1))           # (B, C)
    cref = energy.argmax(dim=1)                                  # (B,)
    ref = smaps[torch.arange(B, device=device), :, cref][:, :, None]
    smaps = smaps * (ref / (ref.abs() + 1e-12)).conj()
    return smaps                                                 # (B, M, C, Nx, Ny)
