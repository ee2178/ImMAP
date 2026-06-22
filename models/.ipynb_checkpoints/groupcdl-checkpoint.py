"""
GroupCDL: interpretable nonlocal unrolled network for image denoising and linear
inverse problems (CS-MRI primary), in PyTorch.

Direct parameterisation of proximal-gradient descent on the convolutional BPDN
problem with a group-sparsity prior (Janjusevic et al.). One network covers both
tasks via a pluggable linear forward operator E (Fig. 4 / Alg. 5 / Alg. 6):

    preprocess:  x_adj = E^H y ;  mu = mean(x_adj) ;  y~ = x_adj - mu
    z^(0) = 0 ,  Gamma^(0) = I
    for k = 0..K-1:
        Gamma^(k) = AdjUpdate(Gamma^(k-1), z^(k))           # Alg. 4 (every dK layers)
        v = A^(k)( E^H E ( B^(k) z^(k) ) - y~ )             # Gram operator per layer
        z^(k+1) = GT_{tau^(k)}( z^(k) - v ; Gamma^(k) )      # Eq. 11 (ST when Gamma=I)
    x_hat = D z^(K) + mu

  * E = Identity  -> denoising (Alg. 5), Gram = I.
  * E = MFR       -> CS-MRI (Alg. 6): masked multi-coil Fourier after sensitivities.
  * tau^(k) = relu(tau0^(k)) + sigma_hat * relu(tau1^(k))   (noise-adaptive; set
    tau1 = 0 / pass nothing for the noise-blind "-B" variants).

Latent z lives in the strided subband domain (Q = N / sc^2 spatial, M channels);
the adjacency / CircAtt act there. The model is complex-valued throughout.

Depends on circulant_attention.py (Circulant algebra + GT) and circulant_similarity.py.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from solvers.eigen import power_method

from operators import Identity
from models.base import BaseUnrolledModel
from models.components import Conv2d, ConvTranspose2d, RealPixelConvComplex
from models.circulant_attention import Circulant, circ_adjacency, _pixelwise, _abs2

from models.circulant_flex import image_to_seq, seq_to_image, build_block_mask, _distance_score_mod
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

# Lets matmuls happen in tf32
torch.set_float32_matmul_precision("high")

# =============================================================================
# GroupCDL
# =============================================================================
class GroupCDL(BaseUnrolledModel):
    def __init__(self, M=169, Mh=64, C=1, P=7, sc=2, K=30, W=35, dK=5,
                 sim_fun="distance", eps=1e-6,
                 attn_backend="gather", blend=True, flex_block_size=128, is_complex= True, init = True):
        super().__init__()
        assert W % 2 == 1, "window side W must be odd"
        assert attn_backend in ("gather", "flex")
        if attn_backend == "flex":
            assert sim_fun in ("distance", "realdot"), \
                "complex flex supports distance/realdot only (PI-sim -> gather path)"
        self.attn_backend, self.blend = attn_backend, blend
        self.flex_block_size = flex_block_size
        self._flex_mask = {}                # cache: (Q1,Q2) -> BlockMask
        self._flex_fn = None                # lazily torch.compile'd flex_attention
        self.M, self.Mh, self.C = M, Mh, C
        self.P, self.sc, self.K = P, sc, K
        self.W, self.dK = W, dK
        self.sim_fun, self.eps = sim_fun, eps
        self.complex = is_complex
        
        self.cdtype = torch.complex64 if self.complex else torch.float32
        
        self.A = nn.ModuleList([
            Conv2d(C, M, P, stride=sc, bias=False, complex = self.complex)
            for _ in range(K)
        ])

        self.B = nn.ModuleList([
            ConvTranspose2d(M, C, P, stride=sc, bias=False, complex = self.complex)
            for _ in range(K)
        ])
        self.D = self.B[0]  # alias D to B[0], otherwise unused as z0 is 0

        self.init_filters(dtype=self.cdtype)
        # Call spectral init to normalize filters
        if init is True:
            self.spectral_init()

        # shared pixel-wise NLSS transforms as 1x1 convs (Mh<<M compresses the attention domain).
        # Wtheta/Wphi/Walpha map M->Mh (complex); Wbeta maps Mh->M and is REAL-valued -- it acts
        # on the real magnitudes sqrt(e) inside the group threshold, so it must stay real.
        self.Wtheta = RealPixelConvComplex(M, Mh)
        self.Wphi   = RealPixelConvComplex(M, Mh)
        self.Walpha = RealPixelConvComplex(M, Mh)
        self.Wbeta  = nn.Conv2d(Mh, M, 1, bias=False)            # real-valued; >= 0
        self._init_nlss_weights()

        # adjacency blend gamma in [0,1]
        self.gamma = nn.Parameter(torch.tensor(0.8))

        # noise-adaptive thresholds tau^(k) = tau0 + sigma_hat * tau1  (real, per channel)
        self.tau0 = nn.Parameter(torch.full((K, M), 1e-3))
        self.tau1 = nn.Parameter(torch.zeros(K, M))
    
    def _init_nlss_weights(self):
        """Initialize the four 1x1-conv NLSS transforms exactly as the paper states
        (Sec. IV-A): a SINGLE matrix drawn from a standard uniform distribution
        U(0, 1), spectrally normalized, and SHARED across Wtheta/Wphi/Walpha/Wbeta.
 
        In the paper all four transforms are Mh x M (Wθ, Wφ, Wα ∈ R^{Mh×M},
        Wβ ∈ R^{Mh×M}_+). The conv weights store:
          Wtheta/Wphi/Walpha.weight : (Mh, M, 1, 1)  -> the shared matrix S
          Wbeta.weight              : (M, Mh, 1, 1)  -> Sᵀ  (Wbeta applies Wβᵀ: Mh->M)
        A 1x1 conv with weight (out, in, 1, 1) and no bias equals _pixelwise(W, x).
 
        Drawing from U(0, 1) keeps S real and non-negative, so the complex
        transforms (θ, φ, α) get real-valued (zero-imaginary) weights and Wβ stays
        >= 0 without an abs(); spectral normalization (a positive division) preserves
        both. The four share values only at init and untie during training (Wθ ≠ Wφ).
 
        Spectral norm is estimated with the power method -- the same helper used for
        the dictionary's B∘A Gram operator -- here on the Hermitian operator SᴴS,
        whose dominant eigenvalue is σ_max(S)², so σ_max(S) = sqrt(|L|)."""
        M, Mh = self.M, self.Mh
        with torch.no_grad():
            S = torch.empty(Mh, M).uniform_(0.0, 1.0)        # standard uniform U(0, 1)
            # Cast S to complex first
            Sc = S.to(self.cdtype)
            StS = lambda x: Sc.conj().t() @ (Sc @ x)           # Hermitian operator SᴴS
            L = power_method(
                StS,
                torch.rand(M, 1, dtype=self.cdtype),
                num_iter=200,
                verbose=False,
            )[0]
            scale = float(np.sqrt(np.abs(L)))                       # σ_max(S) = sqrt(λ_max(SᴴS))
            print(f"W matrix returns L = {L}")
            S = S / scale
            self.Wtheta.weight.copy_(S.view(Mh, M, 1, 1))
            self.Wphi.weight.copy_(  S.view(Mh, M, 1, 1))
            self.Walpha.weight.copy_(S.view(Mh, M, 1, 1))
            # Wbeta stores Wβᵀ (shape M×Mh), real and non-negative.
            self.Wbeta.weight.copy_(S.t().contiguous().view(M, Mh, 1, 1))


    # -- attention backends: both return an `apply_adj` callable energy->Γ·energy --
    #    plus updated state (gather: Circulant; flex: cached (q,k) seqs).
    def _is_update_layer(self, k):
        return (k == 1) or (k >= 1 and (k - 1) % self.dK == 0)

    def _flex_block_mask(self, Q1, Q2, device):
        key = (Q1, Q2)
        if key not in self._flex_mask:
            self._flex_mask[key] = build_block_mask(
                Q1, Q2, self.W, device, circular=True,
                BLOCK_SIZE=self.flex_block_size, compile=True)
        return self._flex_mask[key]

    def _flex_apply(self, q_img, k_img, energy):
        """Γ·energy via FlexAttention: q=Wθz, k=Wφz, v=energy (energy is real).
        Complex q,k are stacked to real [Re; Im] along channels (head_dim 2·Mh):
        Re(qᴴk)=q̃·k̃ gives realdot, and ‖k̃‖²=‖k‖² gives the distance bias -- so
        complex distance/realdot reduce exactly to this real flex call."""
        if q_img.is_complex():                       # [Re; Im] stacking (2·Mh channels)
            q_img = torch.cat([q_img.real, q_img.imag], dim=1)
            k_img = torch.cat([k_img.real, k_img.imag], dim=1)
        Q1, Q2 = q_img.shape[-2], q_img.shape[-1]
        q = image_to_seq(q_img); k = image_to_seq(k_img); v = image_to_seq(energy)
        mask = self._flex_block_mask(Q1, Q2, q.device)
        if self.sim_fun == "distance":
            kb = -0.5 * (k * k).sum(-1, keepdim=True)        # bias as an extra channel
            D = q.shape[-1]
            need = D + 1
            target = 1 << (need - 1).bit_length()            # -> 256 here
            zq = q.new_zeros(*q.shape[:-1], target - need)
            q = torch.cat([q, torch.ones_like(kb), zq], dim=-1)
            k = torch.cat([k, kb,                  zq], dim=-1)
            smod = None
        else:
            smod = None
        fn = self._flex_fn if self._flex_fn is not None else _flex_attention
        o = fn(q, k, v, score_mod=smod, block_mask=mask, scale=1.0,
               kernel_options={"num_stages": 2})
        return seq_to_image(o, Q1, Q2)                # (B,Mh,Q1,Q2) real energy

    def _update_attention(self, state, z, k):
        upd = self._is_update_layer(k)
        if self.attn_backend == "gather":
            Gamma = state
            if upd:
                new = circ_adjacency(self.sim_fun, self.Wtheta(z),
                                     self.Wphi(z), self.W)
                Gamma = new if (Gamma is None or not self.blend) else \
                        new.convex(Gamma, self.gamma.clamp(0.0, 1.0))
            apply_adj = (lambda e, G=Gamma: G.apply(e)) if Gamma is not None else None
            return apply_adj, Gamma
        else:  # flex: cache (q,k) images at update layers, recompute apply each layer
            qk = state
            if upd:
                qk = (self.Wtheta(z), self.Wphi(z))
            if qk is None:
                return None, None
            apply_adj = (lambda e, q=qk[0], kk=qk[1]: self._flex_apply(q, kk, e))
            return apply_adj, qk

    # -- threshold (Eq. 11 GT when apply_adj given; soft-threshold when None) -----
    def _threshold(self, u, apply_adj, tau_map):
        if apply_adj is None:                                # Gamma = I
            denom = u.abs().clamp_min(self.eps)
        else:
            a = self.Walpha(u)                                # Wα^T u  (B,Mh,..)
            e = apply_adj(_abs2(a))                           # (I⊗Γ)(|·|²)
            denom = self.Wbeta(torch.sqrt(torch.sqrt(e.clamp_min(self.eps))))
        factor = torch.clamp(1.0 - tau_map / (denom.clamp_min(self.eps)), min=0.0)
        return u * factor

    # -- forward (Alg. 5 / Alg. 6, unified) -------------------------------------
    def forward(self, y, sigma=0.0, E = None):
        # Quick padding preprocessing 
        H, W = y.shape[-2:]
        pad_h = (-H) % self.sc          # 481 -> 1, 480 -> 0
        pad_w = (-W) % self.sc          # 321 -> 1

        if pad_h or pad_w:
            # F.pad order is (left, right, top, bottom); pad at the far edge
            # so a top-left crop recovers the original.
            y = F.pad(y, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            # mode='circular' for periodic extension instead

        if E is None:
            E = Identity()

        x_adj = E.adjoint(y)                                 # E^H y  -> (B,C,N1,N2)
        mu = x_adj.mean(dim=(1, 2, 3), keepdim=True)         # scalar mean / image
        y_tilde = x_adj - mu

        B, _, N1, N2 = x_adj.shape
        # assert N1 % self.sc == 0 and N2 % self.sc == 0, "spatial dims must divide sc"
        Q1, Q2 = N1 // self.sc, N2 // self.sc
        z = torch.zeros(B, self.M, Q1, Q2, dtype=y.dtype, device=y.device)
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(float(sigma), device=y.device)
        sig = sigma.reshape(-1, 1, 1, 1) if sigma.ndim else sigma

        state = None                                         # Gamma^(0) = I
        for k in range(self.K):
            apply_adj, state = self._update_attention(state, z, k)
            Bz = self.B[k](z)                                # (B,C,N1,N2)
            grad = E.gram(Bz) - y_tilde                      # E^H E B z - y~
            v = self.A[k](grad)                              # (B,M,Q1,Q2)

            tau0 = self.tau0[k].view(1, self.M, 1, 1)
            tau1 = self.tau1[k].view(1, self.M, 1, 1)
            tau_map = tau0 + sig * tau1          # nonneg enforced by project_constraints
            z = self._threshold(z - v, apply_adj, tau_map)

        recon = self.D(z) + mu

        # Pad postprocessing 
        if pad_h or pad_w:
            recon = recon[..., :H, :W]         # crop back to 481 x 321
        return recon, z                                # x_hat

    def compile_flex(self):
        """Wrap flex_attention in torch.compile for the fused GPU kernel (call once)."""
        assert self.attn_backend == "flex"
        self._flex_fn = torch.compile(_flex_attention, dynamic=None)
        return self

    # -- constraint projection (call after each optimiser step; Eq. 16) ---------
    @torch.no_grad()
    def project(self):
        self.project_filters()
        self.gamma.clamp_(0.0, 1.0)
        self.Wbeta.weight.clamp_min_(1e-4)
        self.tau0.clamp_min_(0.0)
        self.tau1.clamp_min_(0.0)
