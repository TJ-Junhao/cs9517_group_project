import numpy as np
from textwrap import dedent
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch import device
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    jaccard_score,
)


def evaluate_metrics(
    predicted: np.ndarray,
    ground_truthes: np.ndarray,
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


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: device,
    criteria: float,
) -> tuple[np.ndarray, np.ndarray]:
    assert 0 <= criteria <= 1
    model.eval()
    expected_list, predicted_list = [], []

    for x, trues in loader:
        x = x.to(device)
        preds = (torch.sigmoid(model(x)) > criteria).int()

        expected_list.append(trues.view(-1).cpu().numpy())
        predicted_list.append(preds.view(-1).cpu().numpy())

    return np.concatenate(expected_list), np.concatenate(predicted_list)


def compute_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, dict]:
    class_names = ["soil", "plant"]
    confusion = confusion_matrix(expected, predicted, labels=[0, 1])
    accuracy = accuracy_score(expected, predicted)
    report = classification_report(
        expected,
        predicted,
        output_dict=True,
        labels=[0, 1],
        target_names=class_names,
    )
    jaccard = jaccard_score(expected, predicted, average=None)

    assert isinstance(jaccard, np.ndarray)
    assert isinstance(report, dict)

    for i, name in enumerate(class_names):
        report[name]["iou"] = float(jaccard[i])
    report["accuracy"] = accuracy

    return confusion, report


def evaluate_neural_network(model, loader, device, criteria):
    expected, predicted = predict(model, loader, device, criteria)
    return compute_metrics(expected, predicted)
