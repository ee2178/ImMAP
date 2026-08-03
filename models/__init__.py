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
from .ladmm import AltSplitCDLNet
from .sb_cdlnet import SBCDLNet


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

    # unrolled linearized ADMM with a learned CDL prox (+ optional joint coils)
    elif model_type == "AltSplitCDLNet":
        return AltSplitCDLNet(**params)

    raise ValueError
