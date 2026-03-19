import cv2 as cv
from utils.types import ArrayLike
import numpy as np
from textwrap import dedent


def evaluate_metrics(
    predicted: list[np.ndarray],
    ground_truthes: list[np.ndarray],
    print_metrics: bool = False,
    custom_title: str = "",
) -> dict[str, float]:
    assert len(predicted) == len(ground_truthes)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for pred, true in zip(predicted, ground_truthes):
        # pred > 0 => plant
        pred_bin = (pred > 0).astype(np.uint8)
        # true == 0 => plant
        true_bin = (true == 0).astype(np.uint8)

        tp += np.logical_and(pred_bin == 1, true_bin == 1).sum()
        tn += np.logical_and(pred_bin == 0, true_bin == 0).sum()
        fp += np.logical_and(pred_bin == 1, true_bin == 0).sum()
        fn += np.logical_and(pred_bin == 0, true_bin == 1).sum()

    eps = 1e-20
    accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))
    precision = float((tp) / (tp + fp + eps))
    recall = float((tp) / (tp + fn + eps))
    f1 = float((2 * precision * recall) / (precision + recall + eps))

    if print_metrics:
        print(
            dedent(
                f"""
    Evaluation Result for **{custom_title}**:

    True Positive: {tp}
    True Negative: {tn}
    False Positive: {fp}
    False Negative: {fn}
    
    Accuracy: {accuracy:.2f}
    Precision: {precision:.2f}
    Recall: {recall:.2f}
    F1: {f1:.2f}
    """
            )
        )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
