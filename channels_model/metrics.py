"""
Metrics for the multi-label bit-plane classification task.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

PRED_THRESHOLD = 0.5
"""Fixed decision threshold on sigmoid(logits) -- not a tunable config knob,
since no validation-driven threshold tuning was requested by the spec."""


def probs_to_preds(probs: np.ndarray) -> np.ndarray:
    return (probs > PRED_THRESHOLD).astype(np.float32)


def confusion_and_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    y_true, y_pred: (N, 8) 0/1 arrays.

    Returns (per_class_acc (8,), confusion (8,2,2)) -- confusion[i] is
    [[TN, FP], [FN, TP]] for bit-plane i (sklearn's multilabel convention).
    """
    conf = multilabel_confusion_matrix(y_true, y_pred)
    per_class_acc = np.array([(m[0, 0] + m[1, 1]) / m.sum() for m in conf])
    return per_class_acc, conf


def super_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of images where ALL 8 bit-plane predictions are exactly
    correct -- the metric of record for picking the final model's epoch
    budget (see train.py)."""
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def format_confusion_matrices(confusion: np.ndarray, indent: str = '  ') -> str:
    """
    Pretty-printed, aligned table for per-epoch logging -- rows TN/FP/FN/TP,
    one column per bit-plane, e.g.:

      Confusion matrices (rows=actual/predicted counts, cols=bit-plane):
              bit1  bit2  bit3  bit4  bit5  bit6  bit7  bit8
        TN       0    12     0     0     0     0    15     7
        FP      14     0    12    10    13    14     0     0
        FN       0     8     0    10     0     0     5    13
        TP       6     0     8     0     7     6     0     0
    """
    num_classes = confusion.shape[0]
    headers = [f"bit{i}" for i in range(1, num_classes + 1)]
    rows = {
        'TN': [int(m[0, 0]) for m in confusion],
        'FP': [int(m[0, 1]) for m in confusion],
        'FN': [int(m[1, 0]) for m in confusion],
        'TP': [int(m[1, 1]) for m in confusion],
    }

    all_values = [v for vals in rows.values() for v in vals]
    col_width = max([len(h) for h in headers] + [len(str(v)) for v in all_values]) + 2
    row_label_width = 4

    lines = [indent + ' ' * row_label_width + ''.join(h.rjust(col_width) for h in headers)]
    for label, vals in rows.items():
        lines.append(indent + label.rjust(row_label_width) + ''.join(str(v).rjust(col_width) for v in vals))
    return "\n".join(lines)
