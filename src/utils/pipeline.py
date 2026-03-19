from typing import Self, Callable, Any
from copy import copy
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from cv2.typing import MatLike, TermCriteria


class ImageState(Enum):
    RGB = auto()
    GRAY = auto()
    HSV = auto()
    BINARY = auto()


class ImagePipeline:
    images: list[MatLike]
    title: str
    image_state: ImageState

    def __init__(
        self: Self, images: list[MatLike], state: ImageState, title: str = "undefined"
    ) -> None:
        self.images = images
        self.title = title
        self.image_state = state

    def __getitem__(self: Self, key: int):
        return self.images[key]

    def __len__(self: Self) -> int:
        return len(self.images)

    def __str__(self) -> str:
        return f"<class ImagePipeline> - with {len(self.images)} images"

    def copy(self) -> Self:
        return self.__class__(
            [img.copy() for img in self.images],
            self.image_state,
            self.title,
        )

    def _cmap(self: Self) -> str | None:
        return {
            ImageState.GRAY: "gray",
            ImageState.RGB: None,
            ImageState.HSV: None,
            ImageState.BINARY: "gray",
        }.get(self.image_state)

    def show(self: Self, index: int) -> None:
        _, ax = plt.subplots(1, 1, dpi=150)
        ax.axis("off")
        im = self.images[index]
        if self.image_state == ImageState.HSV:
            im = cv.cvtColor(im, cv.COLOR_HSV2RGB)
        ax.imshow(im, cmap=self._cmap())
        ax.set_title(f"{self.title}: index = {index}")
        plt.show()

    def rgb_to_gray(self: Self) -> Self:
        assert self.image_state == ImageState.RGB
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.cvtColor(self.images[i], cv.COLOR_RGB2GRAY))
        state = ImageState.GRAY
        return self.__class__(
            ims,
            state,
            self.title,
        )

    def gray_to_rgb(self: Self) -> Self:
        assert self.image_state == ImageState.GRAY
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.cvtColor(self.images[i], cv.COLOR_GRAY2RGB))
        return self.__class__(
            ims,
            ImageState.RGB,
            self.title,
        )

    def rgb_to_hsv(self: Self) -> Self:
        assert self.image_state == ImageState.RGB
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.cvtColor(self.images[i], cv.COLOR_RGB2HSV))
        return self.__class__(
            ims,
            ImageState.HSV,
            self.title,
        )

    def hsv_to_rgb(self: Self) -> Self:
        assert self.image_state == ImageState.HSV
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.cvtColor(self.images[i], cv.COLOR_HSV2RGB))
        return self.__class__(
            ims,
            ImageState.RGB,
            self.title,
        )

    def k_means_clustering(self: Self, k: int, criteria: TermCriteria) -> Self:
        ims = []
        for i in range(len(self.images)):
            im = self.images[i]
            h, w, c = im.shape
            z = im.reshape((-1, c))
            z = np.float32(z)
            _, labels, centers = cv.kmeans(
                z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS  # type: ignore
            )  # type: ignore
            labels = labels.reshape((h, w))
            centers = centers.astype(np.uint8)

            # center -> (label, channel[r, g, b])
            # choose the label with the highest green value as plant
            green_idx = np.argmax(centers[:, 1])

            # plant is white
            mask = (labels == green_idx).astype(np.uint8) * 255

            ims.append(mask)
        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def in_color_range(self: Self, lower: MatLike, upper: MatLike) -> Self:
        assert self.image_state == ImageState.HSV
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.inRange(self.images[i], lower, upper))
        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def otsu(self: Self) -> Self:
        ims = []
        for i in range(len(self.images)):
            _, im = cv.threshold(
                self.images[i], 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY
            )
            ims.append(im)

        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def fill_contours(self: Self) -> Self:
        assert self.image_state == ImageState.BINARY
        ims = []
        for i in range(len(self.images)):
            contour, _ = cv.findContours(
                self.images[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            filled_img = np.zeros_like(self.images[i])
            ims.append(
                cv.drawContours(
                    filled_img, contour, -1, (255, 255, 255), thickness=cv.FILLED
                )
            )

        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def gaussian_blur(self: Self, kernel_size: tuple[int, int], sigma_x: float) -> Self:
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.GaussianBlur(self.images[i], kernel_size, sigma_x))
        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def canny_edge_detect(self: Self, th1: float, th2: float) -> Self:
        ims = []
        for i in range(len(self.images)):
            ims.append(cv.Canny(self.images[i], threshold1=th1, threshold2=th2))
        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def remove_small_object(self: Self, min_area: int = 1000) -> Self:
        ims = []
        for i in range(len(self.images)):
            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
                self.images[i], connectivity=8
            )
            resulting_image = np.zeros_like(self.images[i])

            for j in range(1, num_labels):
                area = stats[j, cv.CC_STAT_AREA]
                if area >= min_area:
                    resulting_image[labels == j] = 255
            ims.append(resulting_image)

        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def excessive_green_mask(self: Self, threshold: int) -> Self:
        assert self.image_state == ImageState.RGB
        ims = []
        for i in range(len(self.images)):
            image = self.images[i].astype(np.int16)
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            processed = (2 * g - r - b > threshold).astype(np.uint8) * 255
            ims.append(processed)
        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def opening(self: Self, kernel_size: tuple[int, int], iters: int = 4) -> Self:
        ims = []
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        for i in range(len(self.images)):
            ims.append(
                cv.morphologyEx(self.images[i], cv.MORPH_OPEN, kernel, iterations=iters)
            )
        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def closing(self: Self, kernel_size: tuple[int, int], iters: int = 4) -> Self:
        ims = []
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        for i in range(len(self.images)):
            ims.append(
                cv.morphologyEx(
                    self.images[i], cv.MORPH_CLOSE, kernel, iterations=iters
                )
            )
        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def apply(self: Self, func: Callable, *args: Any, **kwargs: Any) -> Self:
        ims = []
        for i in range(len(self.images)):
            ims.append(func(self.images[i], *args, **kwargs))
        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def get(self, key: int, color: ImageState) -> MatLike | None:
        image = self.images[key]
        if image is None:
            return None

        if color == self.image_state:
            return image

        if self.image_state == ImageState.RGB and color == ImageState.HSV:
            return cv.cvtColor(image, cv.COLOR_RGB2HSV)
        if self.image_state == ImageState.RGB and color == ImageState.GRAY:
            return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        if self.image_state == ImageState.HSV and color == ImageState.RGB:
            return cv.cvtColor(image, cv.COLOR_HSV2RGB)
        if self.image_state == ImageState.GRAY and color == ImageState.RGB:
            return cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        if self.image_state == ImageState.HSV and color == ImageState.GRAY:
            rgb = cv.cvtColor(image, cv.COLOR_HSV2RGB)
            return cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

        if self.image_state == ImageState.GRAY and color == ImageState.HSV:
            rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            return cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

        raise ValueError(f"Unsupported conversion: {self.image_state} -> {color}")
