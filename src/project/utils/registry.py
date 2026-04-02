from collections.abc import Callable

import numpy as np
from torch import nn

from project.models.cnn import UNet, ResUNet, ASPPResUNet, AttentionGateASPPResUNet
from project.utils.constant import (
    DEVICE,
    NOISE_LEVELS,
    BLUR_LEVELS,
    BRIGHTNESS_LEVELS,
    ROTATION_LEVELS,
    JPEG_COMPRESSION_LEVEL,
)
from project.processing.classic_cv import exessive_green_method, hsv_segmentation

MODELS: dict[str, Callable[[int], nn.Module]] = {
    "unet": lambda cin: UNet(cin).to(DEVICE),
    "resunet": lambda cin: ResUNet(cin).to(DEVICE),
    "aspp_resunet": lambda cin: ASPPResUNet(cin).to(DEVICE),
    "attention_gate_aspp_resunet": lambda cin: AttentionGateASPPResUNet(cin).to(DEVICE),
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

CORRUPTIONS: dict[str, list[dict]] = {
    "gaussian_noise": [{"var": v} for v in NOISE_LEVELS],
    "gaussian_blur": [{"kernel_size": k, "sigma_x": 0} for k in BLUR_LEVELS],
    "brightness_shift": [{"beta": b} for b in BRIGHTNESS_LEVELS],
    "warp_affine": [{"angle": a, "scale": 1.0} for a in ROTATION_LEVELS],
    "jpeg_compression": [{"quality": q} for q in JPEG_COMPRESSION_LEVEL],
}
