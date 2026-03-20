from pathlib import Path
import cv2 as cv
from cv2.typing import MatLike
import numpy as np


def read_images(path: Path) -> tuple[np.ndarray, np.ndarray]:
    images: list[MatLike] = []
    ground_truthes: list[MatLike] = []
    for im_path in path.iterdir():
        if im_path.name.endswith("_mask.png"):
            mask = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
            image = cv.imread(
                im_path.parent / Path(im_path.stem.removesuffix("_mask") + ".png"),
                cv.IMREAD_COLOR_RGB,
            )
            if mask is None or image is None:
                raise IOError("IO Error When reading the image")

            images.append(cv.resize(image, (352, 352)))
            ground_truthes.append(
                cv.resize(mask, (352, 352), interpolation=cv.INTER_NEAREST)
            )

    return (np.array(images), np.array(ground_truthes))
