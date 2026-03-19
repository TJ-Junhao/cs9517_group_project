from pathlib import Path
import cv2 as cv
from cv2.typing import MatLike


def read_images(path: Path) -> tuple[list[MatLike], list[MatLike]]:
    images: list[MatLike] = []
    ground_truthes: list[MatLike] = []
    for im_path in path.iterdir():
        if im_path.name.endswith("_mask.png"):
            mask = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
            im = cv.imread(
                im_path.parent / Path(im_path.stem.removesuffix("_mask") + ".png"),
                cv.IMREAD_COLOR_RGB,
            )
            if mask is None or im is None:
                raise IOError("IO Error When reading the image")

            ground_truthes.append(mask)
            images.append(im)

    return (images, ground_truthes)
