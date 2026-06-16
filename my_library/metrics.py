"""
Evaluation metrics — built from scratch using NumPy only.

Author : Sajid Ansari
Project: Handwritten Digit Recognizer
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def precision_recall_f1_per_class(y_true, y_pred, n_classes=10):
    results = {}
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[c] = {"precision": p, "recall": r, "f1": f1,
                      "support": int(np.sum(y_true == c))}
    return results
