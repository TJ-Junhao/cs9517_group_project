from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2 as cv
from sklearn.ensemble import RandomForestClassifier

from project.processing.pipeline import ImagePipeline, ImageState


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_depth: int | None = 15
    samples_per_class: int = 2000
    random_state: int = 42
    n_jobs: int = -1


def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    """
    Input: RGB image, shape (H, W, 3)
    Output: features, shape (H*W, 7)
            [R, G, B, H, S, V, ExG]
    """
    rgb = image.astype(np.float32)
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV).astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    exg = 2.0 * g - r - b

    feat = np.dstack(
        [
            rgb[:, :, 0],
            rgb[:, :, 1],
            rgb[:, :, 2],
            hsv[:, :, 0],
            hsv[:, :, 1],
            hsv[:, :, 2],
            exg,
        ]
    )
    return feat.reshape(-1, 7)


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
    samples_per_class: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    assert pipe.image_state == ImageState.RGB

    rng = np.random.default_rng(random_state)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for image, mask in zip(pipe.images, pipe.gt):
        feat = extract_features_from_image(image)
        label = mask_to_labels(mask)
        x_i, y_i = sample_balanced_pixels(feat, label, samples_per_class, rng)
        xs.append(x_i)
        ys.append(y_i)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y


def train_random_forest(
    pipe_train: ImagePipeline,
    config: RFConfig,
) -> RandomForestClassifier:
    x_train, y_train = build_training_set(
        pipe_train,
        samples_per_class=config.samples_per_class,
        random_state=config.random_state,
    )

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def predict_mask(model: RandomForestClassifier, image: np.ndarray) -> np.ndarray:
    """
    Return binary mask:
    plant -> 255
    soil  -> 0
    """
    h, w, _ = image.shape
    feat = extract_features_from_image(image)
    pred = model.predict(feat).reshape(h, w).astype(np.uint8)
    return pred * 255


def predict_pipeline(
    model: RandomForestClassifier,
    pipe: ImagePipeline,
    title: str = "RF Prediction",
) -> ImagePipeline:
    preds = [predict_mask(model, image) for image in pipe.images]
    return ImagePipeline(
        np.array(preds),
        pipe.gt.copy(),
        ImageState.BINARY,
        title,
    )