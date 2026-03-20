from typing import Self, Callable, Any
from copy import copy
from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from cv2.typing import MatLike, TermCriteria
from sklearn import svm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from utils.cnn import UNet

INTENSITY_LEVEL = 256
INTENSITY_RANGE = (0, 256)


class ImageState(Enum):
    RGB = auto()
    GRAY = auto()
    HSV = auto()
    BINARY = auto()


class ImagePipeline:
    images: np.ndarray
    title: str
    image_state: ImageState
    svm_clf: svm.LinearSVC | None
    unet_clf: UNet | None

    def __init__(
        self: Self,
        images: list[MatLike] | np.ndarray,
        state: ImageState,
        title: str = "undefined",
        svm_clf: svm.LinearSVC | None = None,
        unet_clf: UNet | None = None,
    ) -> None:
        self.images = np.array([cv.resize(image, (352, 352)) for image in images])
        self.title = title
        self.image_state = state
        self.svm_clf = svm_clf
        self.unet_clf = unet_clf

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

    # def _partition_image(
    #     self: Self, image: MatLike, size: tuple[int, int] = (10, 10)
    # ) -> list[MatLike]:
    #     h = image.shape[0]
    #     w = image.shape[1]
    #     py = size[0]
    #     px = size[1]
    #     patches = []
    #     assert py <= h and px <= w and h % py == 0 and w % px == 0
    #     for y in range(0, h, py):
    #         for x in range(0, w, px):
    #             patches.append(image[y : y + py, x : x + px])
    #     return patches

    def get_data_loader(
        self: Self,
        ground_truth: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        X = torch.tensor(
            (self.images.astype(np.float32) / 255), dtype=torch.float32
        ).permute((0, 3, 1, 2))
        y = torch.tensor(ground_truth > 0, dtype=torch.float32).unsqueeze(1)
        generator = torch.Generator()
        generator = generator.manual_seed(seed)
        return DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,
        )

    @staticmethod
    def get_svm_feature(images: np.ndarray, threshold: int) -> np.ndarray:
        X = []
        for i in range(len(images)):
            image = images[i].astype(np.int16)
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            exg = (2 * g - r - b > threshold).astype(np.float32) * 255
            exg = exg[..., np.newaxis]
            feature = np.concatenate((image, exg), axis=2)
            X.append(feature.reshape(-1, 4))
        return np.concatenate(X, axis=0).astype(np.float32)

    def train_svm_rgb_clf(self: Self, ground_truth: np.ndarray, threshold: int) -> Self:
        assert self.image_state == ImageState.RGB

        X = self.get_svm_feature(self.images, threshold)
        y = [y_i.reshape(-1) for y_i in ground_truth]

        clf = svm.LinearSVC()

        y = np.concatenate(y, axis=0).astype(np.int32)

        clf.fit(X, y)

        return self.__class__(self.images.copy(), ImageState.RGB, self.title, clf)

    def svm_predict(self: Self, images: np.ndarray, threshold: int = 20):
        assert self.svm_clf is not None

        X = self.get_svm_feature(images, threshold)

        return self.svm_clf.predict(X)

    def set_unet_clf(self: Self, clf: UNet) -> Self:
        self.unet_clf = clf
        return self

    def unet_predict(self: Self, criteria: float, device: torch.device) -> Self:
        assert 0 <= criteria <= 1
        assert self.unet_clf is not None

        self.unet_clf.eval()

        ims = []
        X = torch.tensor(
            (self.images.astype(np.float32) / 255), dtype=torch.float32
        ).permute((0, 3, 1, 2))

        with torch.no_grad():
            for x in X:
                x = x.unsqueeze(0).to(device)
                logits = self.unet_clf(x)

                probs = torch.sigmoid(logits)
                preds = (probs > criteria).int()

                preds = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
                ims.append(preds)
        return self.__class__(ims, ImageState.BINARY, self.title)

    def invert(self: Self) -> Self:
        assert self.image_state == ImageState.BINARY
        ims = [cv.bitwise_not(im, None, None) for im in self.images]
        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def rgb_to_gray(self: Self) -> Self:
        assert self.image_state == ImageState.RGB
        ims = [cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in self.images]

        state = ImageState.GRAY
        return self.__class__(
            ims,
            state,
            self.title,
        )

    def gray_to_rgb(self: Self) -> Self:
        assert self.image_state == ImageState.GRAY
        ims = [cv.cvtColor(im, cv.COLOR_GRAY2RGB) for im in self.images]
        return self.__class__(
            ims,
            ImageState.RGB,
            self.title,
        )

    def rgb_to_hsv(self: Self) -> Self:
        assert self.image_state == ImageState.RGB
        ims = [cv.cvtColor(im, cv.COLOR_RGB2HSV) for im in self.images]
        return self.__class__(
            ims,
            ImageState.HSV,
            self.title,
        )

    def hsv_to_rgb(self: Self) -> Self:
        assert self.image_state == ImageState.HSV
        ims = [cv.cvtColor(im, cv.COLOR_HSV2RGB) for im in self.images]
        return self.__class__(
            ims,
            ImageState.RGB,
            self.title,
        )

    def k_means_clustering(self: Self, k: int, criteria: TermCriteria) -> Self:
        ims = []
        for im in self.images:
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
        ims = [cv.inRange(im, lower, upper) for im in self.images]
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
        ims = [cv.GaussianBlur(im, kernel_size, sigma_x) for im in self.images]
        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def canny_edge_detect(self: Self, th1: float, th2: float) -> Self:
        ims = [cv.Canny(im, threshold1=th1, threshold2=th2) for im in self.images]
        return self.__class__(
            ims,
            ImageState.BINARY,
            self.title,
        )

    def remove_small_object(self: Self, min_area: int = 1000) -> Self:
        ims = []
        for im in self.images:
            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
                im, connectivity=8
            )
            resulting_image = np.zeros_like(im)

            for i in range(1, num_labels):
                area = stats[i, cv.CC_STAT_AREA]
                if area >= min_area:
                    resulting_image[labels == i] = 255
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
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        ims = [
            cv.morphologyEx(im, cv.MORPH_OPEN, kernel, iterations=iters)
            for im in self.images
        ]

        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def closing(self: Self, kernel_size: tuple[int, int], iters: int = 4) -> Self:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        ims = [
            cv.morphologyEx(im, cv.MORPH_CLOSE, kernel, iterations=iters)
            for im in self.images
        ]

        return self.__class__(
            ims,
            copy(self.image_state),
            self.title,
        )

    def apply(self: Self, func: Callable, *args: Any, **kwargs: Any) -> Self:
        ims = [func(im, *args, **kwargs) for im in self.images]

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
