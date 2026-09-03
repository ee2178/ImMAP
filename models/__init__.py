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
from .guided_lpds import GuidedLPDSLayer, GuidedLPDSNet, LGGSNet
from .guided_cdl import GuidedGroupCDL, GuidedLISTALayer, LGGCDLNet
from .guided_prox import GuidedFenchelProx, GuidedGroupThreshold
from .ladmm import AltSplitCDLNet
from .sb_cdlnet import SBCDLNet
from .sb_groupcdl import SBGroupCDL
from .sb_unet import SBUnet


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

    # The I2SB paper's regressor: ADM's UNet conditioned on the bridge STEP (not sigma) via a
    # sinusoidal embedding + per-ResBlock FiLM. Like the SB* nets it inverts sigma through its
    # own schedule copy, so model.params must match cfg["i2sb"].
    elif model_type == "SBUnet":
        return SBUnet(**params)

    # Multigrid family, all one class:
    #   K = [K_outer, [iters_per_level...]]  -> V-cycle iterations
    #   K = int                              -> ordinary CDLNet, same blocks
    #   Mh + W > 1                           -> group (nonlocal) prox
    #   dual=True                            -> LPDS: z <- u - prox(u)
    # The four names differ only in the flags they pin, and exist so that a
    # config states its intent (and fails loudly if it contradicts it).
    elif model_type in ("MGCDLNet", "MGGroupCDL"):
        params = dict(params)
        if params.get("dual"):
            raise ValueError(
                f"{model_type} was given dual=true. `dual` on a CDLNet is the "
                f"LISTA-layer imitation of LPDS -- a Fenchel prox and a "
                f"`y~ - D z` read-out, with ONE variable and no tau/theta "
                f"extrapolation. Use MGLPDS (or MGGroupLPDS for the nonlocal "
                f"prox), which build the real primal-dual net.")
        if model_type == "MGGroupCDL":
            if params.get("W", 1) <= 1 or params.get("Mh") is None:
                raise ValueError(
                    f"{model_type} needs a group prox: set W > 1 (attention "
                    f"window side, odd) and Mh (attention channels). "
                    f"Got W={params.get('W', 1)}, Mh={params.get('Mh')}.")
        return MGCDLNet(**params)

    # Multigrid Learned Primal-Dual Splitting -- the port of mg_lpds.jl.
    # Propagates a primal-dual PAIR with over-relaxation and a two-field FAS
    # correction. `K` follows the MGCDLNet convention, so K as a plain int
    # gives an ordinary LPDS stack from the same blocks and the multigrid
    # ablation is one key.
    elif model_type in ("MGLPDSNet", "MGLPDS", "MGGroupLPDS"):
        # The REAL primal-dual nets: an (x, z) pair, tau/theta extrapolation
        # and two FAS corrections. `MGGroupLPDS` differs only in what sits
        # in the prox slot -- a GroupThreshold instead of a SoftThreshold --
        # so nonlocal self-similarity is applied at EVERY grid level rather
        # than once at full resolution.
        #
        # ALL THREE NAMES BUILD THE REAL NET. There is deliberately no way
        # to ask for the old MGCDLNet(dual=True) imitation any more: it
        # shares a name family with LPDS but is a different algorithm.
        params = dict(params)
        # MGGroupLPDS USED to mean MGCDLNet(dual=True, W>1, Mh) -- a LISTA
        # layer with a Fenchel prox, no primal variable, no extrapolation. It
        # now means the real primal-dual net. The two take different keys, so
        # catch an old config by name rather than letting it fail on an
        # unexpected kwarg deep in a constructor.
        # `MGLPDS` / `MGGroupLPDS` used to build MGCDLNet(dual=True): a LISTA
        # layer with a Fenchel prox, ONE variable, no tau/theta. That reading
        # is gone -- dual now always means the real thing -- so translate the
        # one key that carries over and refuse the ones that do not.
        if "W" in params and "window" not in params:
            params["window"] = params.pop("W")
        params.pop("dual", None)              # implied by the name; not a kwarg
        orphans = [k for k in ("eta0", "eta_degrees") if k in params]
        if orphans:
            raise ValueError(
                f"{model_type} does not take {orphans}: `eta` scales the FAS "
                f"correction in a LISTA layer, and the primal-dual sweep "
                f"subtracts pi_x / pi_z unscaled. Drop them. (These keys mean "
                f"the config was written for the old MGCDLNet(dual=True) "
                f"reading of this name.)")
        if model_type == "MGGroupLPDS":
            if params.get("window", 1) <= 1 or params.get("Mh") is None:
                raise ValueError(
                    f"MGGroupLPDS needs a group prox: set window > 1 "
                    f"(attention window side, odd) and Mh (attention "
                    f"channels). Got window={params.get('window', 1)}, "
                    f"Mh={params.get('Mh')}. For the plain soft-threshold "
                    f"prox use MGLPDSNet.")
        elif params.get("window", 1) > 1:
            raise ValueError(
                "MGLPDSNet was given window > 1, which builds a GROUP prox. "
                "Use MGGroupLPDS so the config states that intent.")
        return MGLPDSNet(**params)

    # LGGS -- Longitudinally-Guided Group-Sparse reconstruction: the LPDS unrolling
    # whose dual prox is a GUIDED group threshold, so a fully-sampled prior study
    # shapes the nonlocal grouping without contributing its own intensities.
    # `window` is the SELF window and is 1 in every published LGGS config (the
    # guide window carries the nonlocality), so unlike MGGroupLPDS it is not an
    # error here.
    elif model_type in ("LGGS", "GuidedLPDSNet", "LGGSNet"):
        params = dict(params)
        if params.get("guide_window", 1) <= 1:
            raise ValueError(
                f"{model_type} needs guide_window > 1 (odd): with a 1x1 guide "
                f"window each guide adjacency has a single neighbour and the "
                f"guide contributes only its co-located pixel, which is not a "
                f"nonlocal guide at all. The published setting is "
                f"guide_window=15.")
        if params.get("Mh") is None:
            raise ValueError(
                f"{model_type} needs Mh (the compressed attention channel "
                f"count): the guided prox builds its similarity from "
                f"W_theta/W_phi, which only exist when Mh is set.")
        return LGGSNet(**params)

    # LGGS's ISTA sibling, for REAL-valued data: the same guided group threshold,
    # applied as SHRINKAGE rather than through its Fenchel conjugate. Clipping
    # (what the LPDS dual step applies) pins the modulus at tau for every
    # |z| > tau, so on real features the gradient there is exactly zero and the
    # iterate stops learning wherever the prox is active; complex features keep
    # a live gradient through the phase, which is why LGGS is the complex-valued
    # member of the pair and this is the real-valued one.
    elif model_type in ("GuidedGroupCDL", "LGGCDL", "LGGCDLNet"):
        params = dict(params)
        if params.get("guide_window", 1) <= 1:
            raise ValueError(
                f"{model_type} needs guide_window > 1 (odd): with a 1x1 guide "
                f"window each guide adjacency has a single neighbour and the "
                f"guide contributes only its co-located pixel, which is not a "
                f"nonlocal guide at all. The published setting is "
                f"guide_window=15.")
        if params.get("Mh") is None:
            raise ValueError(
                f"{model_type} needs Mh (the compressed attention channel "
                f"count): the guided prox builds its similarity from "
                f"W_theta/W_phi, which only exist when Mh is set.")
        if params.get("is_complex"):
            raise ValueError(
                f"{model_type} is the REAL-valued member of the guided family "
                f"and was given is_complex=true. On complex data use LGGS "
                f"(model type 'LGGS'), whose primal-dual iteration has the "
                f"better-conditioned data term; the clipping pathology this "
                f"class exists to avoid does not bite there.")
        return GuidedGroupCDL(**params)

    # unrolled linearized ADMM with a learned CDL prox (+ optional joint coils)
    elif model_type == "E2EVarNet":
        # fastMRI's End-to-End VarNet, vendored unmodified under
        # models/e2evarnet/_fastmri/. It estimates its OWN coil maps (that is
        # the method) and returns an RSS MAGNITUDE image, so it is comparable
        # on the magnitude metrics this grid reports and on nothing else.
        from .e2evarnet import E2EVarNet
        return E2EVarNet(**params)

    elif model_type == "AltSplitCDLNet":
        return AltSplitCDLNet(**params)

    raise ValueError
