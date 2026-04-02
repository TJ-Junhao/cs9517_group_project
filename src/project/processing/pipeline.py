from __future__ import annotations
from typing import Self, Callable, Any
from copy import copy
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from cv2.typing import MatLike, TermCriteria
from skimage.util import random_noise

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn


class ImageFlag(Enum):
    PROCESS_IM = auto()
    GROUND_TRUTH = auto()


class ImageState(Enum):
    RGB = auto()
    GRAY = auto()
    HSV = auto()
    BINARY = auto()
    MULTICHANNEL = auto()


class ImagePipeline:
    images: np.ndarray
    gt: np.ndarray
    title: str
    image_state: ImageState
    nn_clf: nn.Module | None

    def __init__(
        self: Self,
        images: list[MatLike] | np.ndarray,
        gt: list[MatLike] | np.ndarray,
        state: ImageState = ImageState.RGB,
        title: str = "undefined",
        nn_clf: nn.Module | None = None,
        resize: tuple[int, int] = (352, 352),
    ) -> None:
        assert len(images) == len(gt)
        self.images = np.array([cv.resize(image, resize) for image in images])
        if any(g.shape[:2] != resize[::-1] for g in gt) or isinstance(gt, list):
            self.gt = np.array([cv.resize(true, resize) for true in gt])
        else:
            self.gt = gt
        self.title = title
        self.image_state = state
        self.nn_clf = nn_clf

    def __getitem__(self: Self, key: tuple[ImageFlag, int] | int) -> MatLike:
        if isinstance(key, tuple):
            if key[0] == ImageFlag.PROCESS_IM:
                return self.images[key[1]]
            else:
                return self.gt[key[1]]
        else:
            return self.images[key]

    def __len__(self: Self) -> int:
        return len(self.images)

    def __str__(self) -> str:
        return f"<class ImagePipeline> - with {len(self.images)} images and {len(self.gt)} ground truth images"

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        result = self.images[self.current]
        self.current += 1
        return result

    def copy(self) -> Self:
        return self.__class__(self.images.copy(), self.gt, self.image_state, self.title)

    def _cmap(self: Self) -> str | None:
        return {
            ImageState.GRAY: "gray",
            ImageState.RGB: None,
            ImageState.HSV: None,
            ImageState.BINARY: "gray",
        }.get(self.image_state)

    def show(self: Self, index: int, dpi: int = 150) -> None:
        _, ax = plt.subplots(1, 2, figsize=(14, 7), dpi=dpi)

        im = self.images[index]
        if self.image_state == ImageState.HSV:
            im = cv.cvtColor(im, cv.COLOR_HSV2RGB)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(im, cmap=self._cmap())
        ax[1].imshow(self.gt[index], cmap="gray")
        plt.suptitle(f"{self.title}: index = {index}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def normalize(self: Self) -> Self:
        ims = self.images.astype(np.float32) / 255
        return self.__class__(
            ims.copy(), self.gt, self.image_state, self.title, self.nn_clf
        )

    def get_data_loader(
        self: Self,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        # Pytorch: (batch, channel, height, width)
        # OpenCv: (batch, height, width, channel)
        X_np = self.images.astype(np.float32) / 255
        if X_np.ndim == 3:
            X_np = X_np[..., None]

        X = torch.tensor(X_np, dtype=torch.float32).permute((0, 3, 1, 2))
        y = torch.tensor(self.gt == 0, dtype=torch.float32).unsqueeze(1)
        generator = torch.Generator()
        generator = generator.manual_seed(seed)
        return DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=0,
        )

    def save(self: Self, save_to: Path, save_gt: bool = False):
        save_to.mkdir(parents=True, exist_ok=True)

        for i, (im, t) in enumerate(zip(self.images, self.gt)):
            cv.imwrite(str(save_to / f"predicted_{i}.png"), im)

            if save_gt:
                gt = np.asarray(t)
                if gt.dtype != np.uint8:
                    gt = (gt > 0).astype(np.uint8) * 255
                cv.imwrite(str(save_to / f"ground_truth_{i}.png"), gt)

    def set_nn_clf(self: Self, clf: nn.Module) -> Self:
        self.nn_clf = clf
        return self

    def nn_predict(self: Self, criteria: float, device: torch.device) -> Self:
        assert 0 <= criteria <= 1
        assert self.nn_clf is not None

        self.nn_clf.eval()

        ims = []
        X_np = self.images.astype(np.float32) / 255
        if X_np.ndim == 3:
            X_np = X_np[..., None]

        X = torch.tensor(X_np, dtype=torch.float32).permute((0, 3, 1, 2))

        with torch.no_grad():
            for x in X:
                x = x.unsqueeze(0).to(device)
                logits = self.nn_clf(x)
                probs = torch.sigmoid(logits)

                preds = (probs <= criteria).to(torch.uint8) * 255
                preds = preds.squeeze().cpu().numpy()

                ims.append(preds)

        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def select_failures(self: Self, bottom_n: int) -> Self:
        assert self.image_state == ImageState.BINARY
        scores = []
        for i, (gt, pred) in enumerate(zip(self.gt, self.images)):
            scores.append((i, self.per_image_iou(gt, pred), pred, gt))

        worst = sorted(scores, key=lambda x: x[1])[:bottom_n]
        return self.__class__(
            np.array([w[2] for w in worst]),
            np.array([w[3] for w in worst]),
            ImageState.BINARY,
            self.title,
        )

    @staticmethod
    def per_image_iou(gt: np.ndarray, pred: np.ndarray) -> float:

        gt_bin = gt > 0
        pred_bin = pred > 0
        union = np.logical_or(gt_bin, pred_bin).sum()
        if union == 0:
            return 1.0
        inter = np.logical_and(gt_bin, pred_bin).sum()
        return inter / union

    def concat(self: Self, other: np.ndarray | ImagePipeline) -> Self:
        ims = []
        assert len(self) == len(other)
        for im, o in zip(self.images, other):
            if im.ndim == 2:
                im = im[..., None]
            if o.ndim == 2:
                o = o[..., None]
            ims.append(np.concatenate((im, o), axis=2))

        return self.__class__(ims, self.gt, ImageState.MULTICHANNEL, self.title)

    def invert(self: Self) -> Self:
        assert self.image_state == ImageState.BINARY
        ims = [cv.bitwise_not(im, None, None) for im in self.images]
        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def rgb_to_gray(self: Self) -> Self:
        assert self.image_state == ImageState.RGB
        ims = [cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in self.images]

        state = ImageState.GRAY
        return self.__class__(ims, self.gt, state, self.title, self.nn_clf)

    def gray_to_rgb(self: Self) -> Self:
        assert self.image_state == ImageState.GRAY
        ims = [cv.cvtColor(im, cv.COLOR_GRAY2RGB) for im in self.images]
        return self.__class__(ims, self.gt, ImageState.RGB, self.title, self.nn_clf)

    def rgb_to_hsv(self: Self) -> Self:
        assert self.image_state == ImageState.RGB
        ims = [cv.cvtColor(im, cv.COLOR_RGB2HSV) for im in self.images]
        return self.__class__(ims, self.gt, ImageState.HSV, self.title, self.nn_clf)

    def hsv_to_rgb(self: Self) -> Self:
        assert self.image_state == ImageState.HSV
        ims = [cv.cvtColor(im, cv.COLOR_HSV2RGB) for im in self.images]
        return self.__class__(ims, self.gt, ImageState.RGB, self.title, self.nn_clf)

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
        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def in_color_range(self: Self, lower: MatLike, upper: MatLike) -> Self:
        assert self.image_state == ImageState.HSV
        ims = [cv.inRange(im, lower, upper) for im in self.images]
        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def otsu(self: Self) -> Self:
        ims = []
        for i in range(len(self.images)):
            _, im = cv.threshold(
                self.images[i], 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY
            )
            ims.append(im)

        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

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

        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def gaussian_blur(self: Self, kernel_size: tuple[int, int], sigma_x: float) -> Self:
        ims = [cv.GaussianBlur(im, kernel_size, sigma_x) for im in self.images]
        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def canny_edge_detect(self: Self, th1: float, th2: float) -> Self:
        ims = [cv.Canny(im, threshold1=th1, threshold2=th2) for im in self.images]
        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

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
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def brightness_shift(self: Self, beta: float) -> Self:
        ims = [
            np.clip(im.astype(np.int16) + beta, 0, 255).astype(np.uint8)
            for im in self.images
        ]
        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def contrast_shift(self: Self, alpha: float) -> Self:
        ims = [
            np.clip(im.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            for im in self.images
        ]
        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def jpeg_compression(self: Self, quality: int) -> Self:
        ims = []
        for im in self.images:
            _, encoded = cv.imencode(".jpg", im, [cv.IMWRITE_JPEG_QUALITY, quality])
            ims.append(cv.imdecode(encoded, cv.IMREAD_COLOR))
        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
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
        return self.__class__(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def opening(self: Self, kernel_size: tuple[int, int], iters: int = 4) -> Self:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        ims = [
            cv.morphologyEx(im, cv.MORPH_OPEN, kernel, iterations=iters)
            for im in self.images
        ]

        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def closing(self: Self, kernel_size: tuple[int, int], iters: int = 4) -> Self:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        ims = [
            cv.morphologyEx(im, cv.MORPH_CLOSE, kernel, iterations=iters)
            for im in self.images
        ]

        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def apply(self: Self, func: Callable, *args: Any, **kwargs: Any) -> Self:
        ims = [func(im, *args, **kwargs) for im in self.images]

        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
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

    def gaussian_noise(self: Self, var: float = 0.01) -> Self:
        ims = np.array(
            [
                (random_noise(im / 255.0, mode="gaussian", var=var) * 255)
                for im in self.images
            ]
        ).astype(np.uint8)

        return self.__class__(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    @staticmethod
    def _get_dst_pts(distortion_intensity: float, shape: tuple[int, int]):
        h, w = shape
        dst_pts = np.array(
            [
                [
                    np.random.uniform(0, w * distortion_intensity),
                    np.random.uniform(0, h * distortion_intensity),
                ],
                [
                    w - np.random.uniform(0, w * distortion_intensity),
                    np.random.uniform(0, h * distortion_intensity),
                ],
                [
                    w - np.random.uniform(0, w * distortion_intensity),
                    h - np.random.uniform(0, h * distortion_intensity),
                ],
                [
                    np.random.uniform(0, w * distortion_intensity),
                    h - np.random.uniform(0, h * distortion_intensity),
                ],
            ]
        )
        return dst_pts

    def warp_perspective(self: Self, distortion_intensity: float) -> Self:
        ims = []
        gts = []

        for im, t in zip(self.images, self.gt):
            h, w = im.shape[:2]

            src_pts = np.array(
                [
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1],
                ],
                dtype=np.float32,
            )

            dst_pts = self._get_dst_pts(distortion_intensity, (h, w)).astype(np.float32)

            M = cv.getPerspectiveTransform(src_pts, dst_pts)

            ims.append(
                cv.warpPerspective(
                    im,
                    M,
                    (w, h),
                    flags=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REFLECT_101,
                )
            )
            gts.append(
                cv.warpPerspective(
                    t,
                    M,
                    (w, h),
                    flags=cv.INTER_NEAREST,
                    borderMode=cv.BORDER_CONSTANT,
                    borderValue=0,
                )
            )

        return self.__class__(
            np.array(ims),
            np.array(gts),
            copy(self.image_state),
            self.title,
            self.nn_clf,
        )

    def warp_affine(self: Self, angle: float, scale: float) -> Self:
        ims = []
        gts = []

        for im, t in zip(self.images, self.gt):
            h, w = im.shape[:2]

            M = cv.getRotationMatrix2D((w // 2, h // 2), angle, scale)
            ims.append(
                cv.warpAffine(
                    im,
                    M,
                    (w, h),
                    flags=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REFLECT_101,
                )
            )
            gts.append(
                cv.warpAffine(
                    t,
                    M,
                    (w, h),
                    flags=cv.INTER_NEAREST,
                    borderMode=cv.BORDER_REFLECT_101,
                )
            )

        return self.__class__(
            np.array(ims),
            np.array(gts),
            copy(self.image_state),
            self.title,
            self.nn_clf,
        )
