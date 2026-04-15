from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2 as cv
from sklearn.linear_model import LogisticRegression

from project.processing.pipeline import ImagePipeline, ImageState


@dataclass
class LRConfig:
    max_iter: int = 1000
    random_state: int = 42
    samples_per_class: int = 500


def extract_features_from_image(
    image: np.ndarray,
    feature_mode: str = "rgb_hsv_exg",
) -> np.ndarray:
    rgb = image.astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    if feature_mode == "rgb":
        feat = np.dstack([r, g, b])
        return feat.reshape(-1, 3)

    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if feature_mode == "rgb_hsv":
        feat = np.dstack([r, g, b, h, s, v])
        return feat.reshape(-1, 6)

    if feature_mode == "rgb_hsv_exg":
        exg = 2.0 * g - r - b
        feat = np.dstack([r, g, b, h, s, v, exg])
        return feat.reshape(-1, 7)

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def mask_to_labels(mask: np.ndarray) -> np.ndarray:
    return (mask.reshape(-1) == 0).astype(np.uint8)


def sample_balanced_pixels(
    features: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    plant_idx = np.where(labels == 1)[0]
    soil_idx = np.where(labels == 0)[0]

    if len(plant_idx) == 0 or len(soil_idx) == 0:
        return features, labels

    n_plant = min(samples_per_class, len(plant_idx))
    n_soil = min(samples_per_class, len(soil_idx))

    plant_sel = rng.choice(plant_idx, size=n_plant, replace=False)
    soil_sel = rng.choice(soil_idx, size=n_soil, replace=False)

    idx = np.concatenate([plant_sel, soil_sel])
    rng.shuffle(idx)

    return features[idx], labels[idx]


def build_training_set(
    pipe: ImagePipeline,
    samples_per_class: int = 500,
    random_state: int = 42,
    feature_mode: str = "rgb_hsv_exg",
) -> tuple[np.ndarray, np.ndarray]:
    assert pipe.image_state == ImageState.RGB

    rng = np.random.default_rng(random_state)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for image, mask in zip(pipe.images, pipe.gt):
        feat = extract_features_from_image(image, feature_mode=feature_mode)
        label = mask_to_labels(mask)
        x_i, y_i = sample_balanced_pixels(feat, label, samples_per_class, rng)
        xs.append(x_i)
        ys.append(y_i)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y


def train_logistic_regression(
    pipe_train: ImagePipeline,
    config: LRConfig,
    feature_mode: str = "rgb_hsv_exg",
) -> LogisticRegression:
    x_train, y_train = build_training_set(
        pipe_train,
        samples_per_class=config.samples_per_class,
        random_state=config.random_state,
        feature_mode=feature_mode,
    )

    model = LogisticRegression(
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    model.fit(x_train, y_train)
    return model


def predict_mask(
    model: LogisticRegression,
    image: np.ndarray,
    feature_mode: str = "rgb_hsv_exg",
) -> np.ndarray:
    h, w, _ = image.shape
    feat = extract_features_from_image(image, feature_mode=feature_mode)
    pred = model.predict(feat).reshape(h, w).astype(np.uint8)
    return pred * 255


def predict_pipeline(
    model: LogisticRegression,
    pipe: ImagePipeline,
    title: str = "LR Prediction",
    feature_mode: str = "rgb_hsv_exg",
) -> ImagePipeline:
    preds = [
        predict_mask(model, image, feature_mode=feature_mode) for image in pipe.images
    ]
    return ImagePipeline(
        np.array(preds),
        pipe.gt.copy(),
        ImageState.BINARY,
        title,
    )
