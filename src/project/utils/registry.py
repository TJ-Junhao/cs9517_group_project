from collections.abc import Callable

import numpy as np
from torch import nn

from project.models.cnn import UNet, ResUNet, ASPPResUNet
from project.utils.constant import DEVICE
from project.processing.classic_cv import exessive_green_method, hsv_segmentation

MODELS: dict[str, Callable[[int], nn.Module]] = {
    "unet": lambda cin: UNet(cin).to(DEVICE),
    "resunet": lambda cin: ResUNet(cin).to(DEVICE),
    "aspp_resunet": lambda cin: ASPPResUNet(cin).to(DEVICE),
}

FEATURE_BUILDERS = {
    "exg": lambda pipe, **_: exessive_green_method(pipe),
    "hsv": lambda pipe, **kwargs: hsv_segmentation(
        pipe,
        lower=np.array(kwargs["lower"]),
        upper=np.array(kwargs["upper"]),
        kernel_size=tuple(kwargs["kernel_size"]),
        iters=kwargs["iters"],
    ),
}
