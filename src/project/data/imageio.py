from pathlib import Path
from typing import overload

import cv2 as cv
from cv2.typing import MatLike
import numpy as np

from project.processing.pipeline import ImagePipeline, ImageState


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


@overload
def load_data(
    train_path: None, val_path: Path, test_path: Path
) -> tuple[None, ImagePipeline, ImagePipeline]: ...


@overload
def load_data(
    train_path: Path, val_path: None, test_path: Path
) -> tuple[ImagePipeline, None, ImagePipeline]: ...


@overload
def load_data(
    train_path: Path, val_path: Path, test_path: None
) -> tuple[ImagePipeline, ImagePipeline, None]: ...


@overload
def load_data(
    train_path: None, val_path: None, test_path: Path
) -> tuple[None, None, ImagePipeline]: ...


@overload
def load_data(
    train_path: Path, val_path: None, test_path: None
) -> tuple[ImagePipeline, None, None]: ...


@overload
def load_data(
    train_path: None, val_path: Path, test_path: None
) -> tuple[None, ImagePipeline, None]: ...


@overload
def load_data(
    train_path: Path, val_path: Path, test_path: Path
) -> tuple[ImagePipeline, ImagePipeline, ImagePipeline]: ...


def load_data(
    train_path: Path | None, val_path: Path | None, test_path: Path | None
) -> tuple[ImagePipeline | None, ImagePipeline | None, ImagePipeline | None]:
    pipe_train = pipe_val = pipe_test = None

    if train_path is not None:
        train_x, train_y = read_images(train_path)
        assert len(train_x) == len(train_y)
        pipe_train = ImagePipeline(train_x, train_y, ImageState.RGB, "Training Set")

    if val_path is not None:
        val_x, val_y = read_images(val_path)
        assert len(val_x) == len(val_y)
        pipe_val = ImagePipeline(val_x, val_y, ImageState.RGB, "Validation Set")

    if test_path is not None:
        test_x, test_y = read_images(test_path)
        assert len(test_x) == len(test_y)
        pipe_test = ImagePipeline(test_x, test_y, ImageState.RGB, "Test Set")
    return pipe_train, pipe_val, pipe_test
