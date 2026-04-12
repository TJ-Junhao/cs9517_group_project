from __future__ import annotations
from typing import Self, Callable, Any, overload
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
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from project.data.imageio import read_images


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

    @staticmethod
    def read_from_path(
        data_path: Path, title: str = "undefined", resize: tuple[int, int] = (352, 352)
    ) -> ImagePipeline:
        ims, gts = read_images(data_path)
        return ImagePipeline(ims, gts, title=title, resize=resize)

    @staticmethod
    @overload
    def load_data(
        train_path: None, val_path: Path, test_path: Path
    ) -> tuple[None, ImagePipeline, ImagePipeline]: ...

    @staticmethod
    @overload
    def load_data(
        train_path: Path, val_path: None, test_path: Path
    ) -> tuple[ImagePipeline, None, ImagePipeline]: ...

    @staticmethod
    @overload
    def load_data(
        train_path: Path, val_path: Path, test_path: None
    ) -> tuple[ImagePipeline, ImagePipeline, None]: ...

    @staticmethod
    @overload
    def load_data(
        train_path: None, val_path: None, test_path: Path
    ) -> tuple[None, None, ImagePipeline]: ...

    @staticmethod
    @overload
    def load_data(
        train_path: Path, val_path: None, test_path: None
    ) -> tuple[ImagePipeline, None, None]: ...

    @staticmethod
    @overload
    def load_data(
        train_path: None, val_path: Path, test_path: None
    ) -> tuple[None, ImagePipeline, None]: ...

    @staticmethod
    @overload
    def load_data(
        train_path: Path, val_path: Path, test_path: Path
    ) -> tuple[ImagePipeline, ImagePipeline, ImagePipeline]: ...

    @staticmethod
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

    def copy(self) -> ImagePipeline:
        return ImagePipeline(
            self.images.copy(), self.gt, self.image_state, self.title, self.nn_clf
        )

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

    def flatten(self: Self) -> tuple[np.ndarray, np.ndarray]:
        pred = (self.images.ravel() == 0).astype(np.uint8)
        gt = (self.gt.ravel() == 0).astype(np.uint8)
        return pred, gt

    def normalize(self: Self) -> ImagePipeline:
        ims = self.images.astype(np.float32) / 255
        return ImagePipeline(
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
            out = im
            if self.image_state == ImageState.RGB:
                out = cv.cvtColor(im, cv.COLOR_RGB2BGR)
            elif self.image_state == ImageState.HSV:
                out = cv.cvtColor(cv.cvtColor(im, cv.COLOR_HSV2RGB), cv.COLOR_RGB2BGR)
            cv.imwrite(str(save_to / f"predicted_{i}.png"), out)

            if save_gt:
                gt = np.asarray(t)
                if gt.dtype != np.uint8:
                    gt = (gt > 0).astype(np.uint8) * 255
                cv.imwrite(str(save_to / f"ground_truth_{i}.png"), gt)

    def set_nn_clf(self: Self, clf: nn.Module) -> ImagePipeline:
        self.nn_clf = clf
        return self

    def nn_predict(self: Self, criteria: float, device: torch.device) -> ImagePipeline:
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

        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    @staticmethod
    def from_arrays(
        gt: np.ndarray,
        predicted: np.ndarray,
        title: str = "",
    ) -> ImagePipeline:
        N, H, W = gt.shape[0], gt.shape[1], gt.shape[2]
        pred_maps = predicted.reshape(N, H, W).astype(np.uint8) * 255

        return ImagePipeline(
            pred_maps,
            gt,
            ImageState.BINARY,
            title,
        )

    def select_failures(self: Self, bottom_n: int) -> ImagePipeline:
        assert self.image_state == ImageState.BINARY
        scores = []
        for i, (gt, pred) in enumerate(zip(self.gt, self.images)):
            scores.append((i, self.per_image_iou(gt, pred), pred, gt))

        worst = sorted(scores, key=lambda x: x[1])[:bottom_n]
        return ImagePipeline(
            np.array([w[2] for w in worst]),
            np.array([w[3] for w in worst]),
            ImageState.BINARY,
            self.title,
        )

    @staticmethod
    def per_image_iou(gt: np.ndarray, pred: np.ndarray) -> float:
        gt_bin = gt > 0
        pred_bin = pred > 0

        fg_union = np.logical_or(gt_bin, pred_bin).sum()
        fg_iou = (
            1.0
            if fg_union == 0
            else (np.logical_and(gt_bin, pred_bin).sum() / fg_union)
        )

        bg_union = np.logical_or(~gt_bin, ~pred_bin).sum()
        bg_iou = (
            1.0
            if bg_union == 0
            else (np.logical_and(~gt_bin, ~pred_bin).sum() / bg_union)
        )

        return (fg_iou + bg_iou) / 2

    def concat(self: Self, other: np.ndarray | ImagePipeline) -> ImagePipeline:
        ims = []
        assert len(self) == len(other)
        for im, o in zip(self.images, other):
            if im.ndim == 2:
                im = im[..., None]
            if o.ndim == 2:
                o = o[..., None]
            ims.append(np.concatenate((im, o), axis=2))

        return ImagePipeline(ims, self.gt, ImageState.MULTICHANNEL, self.title)

    def invert(self: Self) -> ImagePipeline:
        assert self.image_state == ImageState.BINARY
        ims = [cv.bitwise_not(im, None, None) for im in self.images]
        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def rgb_to_gray(self: Self) -> ImagePipeline:
        assert self.image_state == ImageState.RGB
        ims = [cv.cvtColor(im, cv.COLOR_RGB2GRAY) for im in self.images]

        state = ImageState.GRAY
        return ImagePipeline(ims, self.gt, state, self.title, self.nn_clf)

    def gray_to_rgb(self: Self) -> ImagePipeline:
        assert self.image_state == ImageState.GRAY
        ims = [cv.cvtColor(im, cv.COLOR_GRAY2RGB) for im in self.images]
        return ImagePipeline(ims, self.gt, ImageState.RGB, self.title, self.nn_clf)

    def rgb_to_hsv(self: Self) -> ImagePipeline:
        assert self.image_state == ImageState.RGB
        ims = [cv.cvtColor(im, cv.COLOR_RGB2HSV) for im in self.images]
        return ImagePipeline(ims, self.gt, ImageState.HSV, self.title, self.nn_clf)

    def hsv_to_rgb(self: Self) -> ImagePipeline:
        assert self.image_state == ImageState.HSV
        ims = [cv.cvtColor(im, cv.COLOR_HSV2RGB) for im in self.images]
        return ImagePipeline(ims, self.gt, ImageState.RGB, self.title, self.nn_clf)

    def k_means_clustering(self: Self, k: int, criteria: TermCriteria) -> ImagePipeline:
        assert self.image_state == ImageState.RGB
        ims = []
        for im in self.images:
            h, w, c = im.shape
            z = im.reshape((-1, c))
            z = np.float32(z)
            _, labels, centers = cv.kmeans(
                z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS  # type: ignore
            )  # type: ignore
            labels = labels.reshape((h, w))
            exg_per_center = (
                2 * centers[:, 1].astype(int)
                - centers[:, 0].astype(int)
                - centers[:, 2].astype(int)
            )
            plant_idx = np.argmax(exg_per_center)

            # plant is white
            mask = (labels == plant_idx).astype(np.uint8) * 255

            ims.append(mask)
        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def in_color_range(self: Self, lower: MatLike, upper: MatLike) -> ImagePipeline:
        assert self.image_state == ImageState.HSV
        ims = [cv.inRange(im, lower, upper) for im in self.images]
        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def otsu(self: Self) -> ImagePipeline:
        ims = []
        for i in range(len(self.images)):
            _, im = cv.threshold(
                self.images[i], 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY
            )
            ims.append(im)

        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def fill_contours(self: Self) -> ImagePipeline:
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

        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def gaussian_blur(
        self: Self, kernel_size: tuple[int, int], sigma_x: float
    ) -> ImagePipeline:
        ims = [cv.GaussianBlur(im, kernel_size, sigma_x) for im in self.images]
        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def canny_edge_detect(self: Self, th1: float, th2: float) -> ImagePipeline:
        ims = [cv.Canny(im, threshold1=th1, threshold2=th2) for im in self.images]
        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def remove_small_object(self: Self, min_area: int = 1000) -> ImagePipeline:
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

        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def brightness_shift(self: Self, beta: float) -> ImagePipeline:
        ims = [
            np.clip(im.astype(np.int16) + beta, 0, 255).astype(np.uint8)
            for im in self.images
        ]
        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def contrast_shift(self: Self, alpha: float) -> ImagePipeline:
        ims = [
            np.clip(im.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            for im in self.images
        ]
        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def jpeg_compression(self: Self, quality: int) -> ImagePipeline:
        ims = []
        for im in self.images:
            _, encoded = cv.imencode(".jpg", im, [cv.IMWRITE_JPEG_QUALITY, quality])
            ims.append(cv.imdecode(encoded, cv.IMREAD_COLOR))
        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def excessive_green_mask(self: Self, threshold: int) -> ImagePipeline:
        assert self.image_state == ImageState.RGB
        ims = []
        for i in range(len(self.images)):
            image = self.images[i].astype(np.int16)
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            processed = (2 * g - r - b > threshold).astype(np.uint8) * 255
            ims.append(processed)
        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def opening(
        self: Self, kernel_size: tuple[int, int], iters: int = 4
    ) -> ImagePipeline:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        ims = [
            cv.morphologyEx(im, cv.MORPH_OPEN, kernel, iterations=iters)
            for im in self.images
        ]

        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def closing(
        self: Self, kernel_size: tuple[int, int], iters: int = 4
    ) -> ImagePipeline:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=kernel_size)
        ims = [
            cv.morphologyEx(im, cv.MORPH_CLOSE, kernel, iterations=iters)
            for im in self.images
        ]

        return ImagePipeline(
            ims, self.gt, copy(self.image_state), self.title, self.nn_clf
        )

    def apply(self: Self, func: Callable, *args: Any, **kwargs: Any) -> ImagePipeline:
        ims = [func(im, *args, **kwargs) for im in self.images]

        return ImagePipeline(
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

    def gaussian_noise(self: Self, var: float = 0.01) -> ImagePipeline:
        ims = np.array(
            [
                (random_noise(im / 255.0, mode="gaussian", var=var) * 255)
                for im in self.images
            ]
        ).astype(np.uint8)

        return ImagePipeline(
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

    def warp_perspective(self: Self, distortion_intensity: float) -> ImagePipeline:
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

        return ImagePipeline(
            np.array(ims),
            np.array(gts),
            copy(self.image_state),
            self.title,
            self.nn_clf,
        )

    def warp_affine(self: Self, angle: float, scale: float) -> ImagePipeline:
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

        return ImagePipeline(
            np.array(ims),
            np.array(gts),
            copy(self.image_state),
            self.title,
            self.nn_clf,
        )

    def watershed(
        self: Self,
        exg_low: int = -20,
        exg_high: int = 20,
    ) -> ImagePipeline:
        assert self.image_state == ImageState.RGB
        ims = []
        for im in self.images:
            img_i = im.astype(np.int16)
            exg = 2 * img_i[:, :, 1] - img_i[:, :, 0] - img_i[:, :, 2]

            markers = np.zeros(im.shape[:2], dtype=np.int32)
            markers[exg > exg_high] = 2  # plant seed
            markers[exg < exg_low] = 1  # soil seed

            # cv.watershed requires BGR
            bgr = cv.cvtColor(im, cv.COLOR_RGB2BGR)
            cv.watershed(bgr, markers)

            # markers == 2 → plant (255), boundaries (-1) and soil → 0
            mask = (markers == 2).astype(np.uint8) * 255
            ims.append(mask)

        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def grabcut(
        self: Self,
        exg_threshold: int = 10,
        iters: int = 5,
    ) -> ImagePipeline:
        assert self.image_state == ImageState.RGB
        ims = []
        for im in self.images:
            img_i = im.astype(np.int16)
            exg = 2 * img_i[:, :, 1] - img_i[:, :, 0] - img_i[:, :, 2]

            gc_mask = np.full(im.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)
            gc_mask[exg > exg_threshold] = cv.GC_PR_FGD
            gc_mask[exg > exg_threshold * 2] = cv.GC_FGD
            gc_mask[exg < -exg_threshold] = cv.GC_BGD

            bgr = cv.cvtColor(im, cv.COLOR_RGB2BGR)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, iters, cv.GC_INIT_WITH_MASK)  # type: ignore

            mask = np.where(
                (gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            ims.append(mask)

        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)

    def dense_crf(
        self: Self,
        exg_threshold: int = 10,
        gt_prob: float = 0.7,
        iters: int = 5,
        sxy_gaussian: int = 3,
        compat_gaussian: int = 3,
        sxy_bilateral: int = 60,
        srgb_bilateral: int = 13,
        compat_bilateral: int = 10,
    ) -> ImagePipeline:
        assert self.image_state == ImageState.RGB

        ims = []
        for im in self.images:
            img_i = im.astype(np.int16)
            exg = 2 * img_i[:, :, 1] - img_i[:, :, 0] - img_i[:, :, 2]
            labels = (exg > exg_threshold).astype(np.int32)

            h, w = im.shape[:2]
            d = dcrf.DenseCRF2D(w, h, 2)
            U = unary_from_labels(labels, 2, gt_prob=gt_prob, zero_unsure=False)
            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
            d.addPairwiseBilateral(
                sxy=sxy_bilateral,
                srgb=srgb_bilateral,
                rgbim=np.ascontiguousarray(im),
                compat=compat_bilateral,
            )

            Q = d.inference(iters)
            refined = np.argmax(Q, axis=0).reshape(h, w)
            mask = (refined == 1).astype(np.uint8) * 255
            ims.append(mask)

        return ImagePipeline(ims, self.gt, ImageState.BINARY, self.title, self.nn_clf)
