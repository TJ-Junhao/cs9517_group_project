import cv2 as cv
from utils.types import ArrayLike
import numpy as np
from textwrap import dedent
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch import device
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


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
def evaluate_neural_network(
    model: nn.Module, loader: DataLoader, device: device, criteria: float
):
    assert 0 <= criteria <= 1
    model.eval()
    expected_list, predicted_list = [], []

    for x, trues in loader:
        x = x.to(device)
        logits = model(x)

        probs = torch.sigmoid(logits)
        preds = (probs > criteria).int()

        trues = trues.view(-1).cpu().numpy()
        preds = preds.view(-1).cpu().numpy()

        expected_list.append(trues)
        predicted_list.append(preds)

    expected_out = np.concatenate(expected_list)

    predicted_out = np.concatenate(predicted_list)

    # Create confusion matrix
    confusion = confusion_matrix(expected_out, predicted_out, labels=[0, 1])
    # Calculate accuracy
    accuracy = accuracy_score(expected_out, predicted_out)
    # Get classification report
    report = classification_report(expected_out, predicted_out, output_dict=True)

    return accuracy, confusion, report
