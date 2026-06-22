from .cdlnet import CDLNet
from .lpdsnet import LPDSNet
from .difflpdsnet import DiffLPDSNet
from .unet import Unet
from .normunet import NormUnet
from .ipalmnet import IPALMNet
from .groupcdl import GroupCDL
from .cclnet import CCLNet, Unet2D


def build_model(cfg):

    model_type = cfg["model"]["type"]

    params = cfg["model"]["params"]

    if model_type == "CDLNet":
        return CDLNet(**params)

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
    raise ValueError
