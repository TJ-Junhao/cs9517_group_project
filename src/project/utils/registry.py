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
from project.processing.traditional_cv import (
    excessive_green_method,
    hsv_segmentation_method,
    edge_method,
    watershed_method,
    kmeans_method,
    grabcut_method,
    crf_method,
)
from project.processing.pipeline import ImagePipeline

MODELS: dict[str, Callable[[int], nn.Module]] = {
    "unet": lambda cin: UNet(cin).to(DEVICE),
    "resunet": lambda cin: ResUNet(cin).to(DEVICE),
    "aspp_resunet": lambda cin: ASPPResUNet(cin).to(DEVICE),
    "attention_gate_aspp_resunet": lambda cin: AttentionGateASPPResUNet(cin).to(DEVICE),
}

FEATURE_BUILDERS = {
    "exg": lambda pipe, **_: excessive_green_method(pipe),
    "hsv": lambda pipe, **kwargs: hsv_segmentation_method(
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

TRADITIONAL_CV_METHODS: dict[str, Callable[..., ImagePipeline]] = {
    "kmeans_method": kmeans_method,
    "edge_method": edge_method,
    "watershed_method": watershed_method,
    "grabcut_method": grabcut_method,
    "excessive_green_method": excessive_green_method,
    "hsv_segmentation_method": hsv_segmentation_method,
    "crf_method": crf_method,
}
