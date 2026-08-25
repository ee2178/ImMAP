from .cdlnet import CDLNet
from .dt_cdlnet import DTCDLNet
from .lpdsnet import LPDSNet
from .difflpdsnet import DiffLPDSNet
from .unet import Unet
from .normunet import NormUnet
from .ipalmnet import IPALMNet
from .groupcdl import GroupCDL
from .cclnet import CCLNet, Unet2D
from .multigrid import MGCDLNet, VCycle
from .lpds import LPDSLayer, LPDSStack
from .mg_lpds import MGLPDSNet, PDVCycle
from .ladmm import AltSplitCDLNet
from .sb_cdlnet import SBCDLNet
from .sb_groupcdl import SBGroupCDL


def build_model(cfg):

    model_type = cfg["model"]["type"]

    params = cfg["model"]["params"]

    if model_type == "CDLNet":
        return CDLNet(**params)

    elif model_type == "DTCDLNet":
        return DTCDLNet(**params)

    elif model_type == "LPDSNet":
        return LPDSNet(**params)

    elif model_type == "DiffLPDSNet":
        return DiffLPDSNet(**params)
    
    elif model_type == 'Unet':
        return Unet(**params)

    elif model_type == 'NormUnet':
        return NormUnet(**params)

    elif model_type == "IPALMNet":
        return IPALMNet(**params)
    
    elif model_type == "GroupCDL":
        return GroupCDL(**params)
    elif model_type == "CCLNet":
        return CCLNet(**params)
    elif model_type == "Unet2D":
        return Unet2D(**params)

    # Schrodinger-bridge CDLNet: two-fidelity unrolled ISTA over a shared sparse code
    # (target-domain + prior-domain dictionaries). Its schedule params must match cfg["i2sb"].
    elif model_type == "SBCDLNet":
        return SBCDLNet(**params)

    # Same two-fidelity bridge scaffold as SBCDLNet, with GroupCDL's nonlocal group-sparsity
    # prox in place of the soft threshold. Its schedule params must match cfg["i2sb"] too.
    elif model_type == "SBGroupCDL":
        return SBGroupCDL(**params)

    # Multigrid family, all one class:
    #   K = [K_outer, [iters_per_level...]]  -> V-cycle iterations
    #   K = int                              -> ordinary CDLNet, same blocks
    #   Mh + W > 1                           -> group (nonlocal) prox
    #   dual=True                            -> LPDS: z <- u - prox(u)
    # The four names differ only in the flags they pin, and exist so that a
    # config states its intent (and fails loudly if it contradicts it).
    elif model_type in ("MGCDLNet", "MGGroupCDL", "MGLPDS", "MGGroupLPDS"):
        params = dict(params)
        if model_type in ("MGLPDS", "MGGroupLPDS"):
            # Moreau's identity: the Fenchel prox of the same threshold, i.e. a
            # residual connection around it (clipping instead of shrinkage),
            # with the read-out becoming y~ - D z.
            params["dual"] = True
        if model_type in ("MGGroupCDL", "MGGroupLPDS"):
            if params.get("W", 1) <= 1 or params.get("Mh") is None:
                raise ValueError(
                    f"{model_type} needs a group prox: set W > 1 (attention "
                    f"window side, odd) and Mh (attention channels). "
                    f"Got W={params.get('W', 1)}, Mh={params.get('Mh')}.")
        return MGCDLNet(**params)

    # Multigrid Learned Primal-Dual Splitting -- the port of mg_lpds.jl. NOT the
    # same thing as `MGLPDS` above: that is MGCDLNet(dual=True), a LISTA layer
    # with a clipping prox (one iterate, no extrapolation). This propagates a
    # primal-dual PAIR with over-relaxation and a two-field FAS correction.
    # `K` follows the MGCDLNet convention, so K as a plain int gives an ordinary
    # LPDS stack from the same blocks and the multigrid ablation is one key.
    elif model_type == "MGLPDSNet":
        return MGLPDSNet(**params)

    # unrolled linearized ADMM with a learned CDL prox (+ optional joint coils)
    elif model_type == "AltSplitCDLNet":
        return AltSplitCDLNet(**params)

    raise ValueError
