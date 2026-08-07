from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import log2

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def collision_conditioned_ceiling(
    features: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError("features and labels must describe a non-empty row-aligned matrix")
    _, inverse = np.unique(features, axis=0, return_inverse=True)
    correct = 0
    balanced_correct = Fraction(0, 1)
    collision_rows = 0
    pure_rows = 0
    singleton_classes = 0
    singleton_rows = 0
    mixed_classes = 0
    conditional_entropy = 0.0
    class_counts: list[tuple[int, int]] = []
    observed_labels = np.unique(labels)
    label_counts = {
        int(label): int(np.sum(labels == label)) for label in observed_labels
    }
    for group in range(int(inverse.max()) + 1):
        group_labels = labels[inverse == group]
        counts = np.bincount(
            group_labels,
            minlength=max(2, int(observed_labels.max()) + 1),
        )
        class_counts.append((int(counts[0]), int(counts[1])))
        group_size = len(group_labels)
        correct += int(counts.max())
        balanced_correct += max(
            Fraction(int(counts[int(label)]), label_counts[int(label)])
            for label in observed_labels
        )
        if group_size == 1:
            singleton_classes += 1
            singleton_rows += 1
        if np.count_nonzero(counts) > 1:
            collision_rows += group_size
            mixed_classes += 1
        else:
            pure_rows += group_size
        for count in counts:
            if count:
                probability = float(count / group_size)
                conditional_entropy -= (group_size / len(labels)) * probability * log2(
                    probability
                )
    roc_auc_ceiling = None
    if set(map(int, np.unique(labels))) == {0, 1}:
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        by_posterior: dict[Fraction, list[int]] = {}
        for negative_count, positive_count in class_counts:
            posterior = Fraction(
                positive_count,
                positive_count + negative_count,
            )
            totals = by_posterior.setdefault(posterior, [0, 0])
            totals[0] += negative_count
            totals[1] += positive_count
        concordant = 0.0
        lower_negative_count = 0
        for posterior in sorted(by_posterior):
            negative_count, positive_count = by_posterior[posterior]
            concordant += positive_count * lower_negative_count
            concordant += 0.5 * positive_count * negative_count
            lower_negative_count += negative_count
        roc_auc_ceiling = float(concordant / (positives * negatives))

    prior = float(np.bincount(labels, minlength=2).max() / len(labels))
    label_probabilities = np.bincount(labels, minlength=2) / len(labels)
    label_entropy = float(
        -sum(probability * log2(probability) for probability in label_probabilities if probability)
    )
    histogram = Counter(class_counts)
    return {
        "ceiling": float(correct / len(labels)),
        "balanced_accuracy_ceiling": float(balanced_correct / len(observed_labels)),
        "roc_auc_ceiling": roc_auc_ceiling,
        "majority_prior": prior,
        "rows": int(len(labels)),
        "unique_representations": int(inverse.max()) + 1,
        "singleton_classes": int(singleton_classes),
        "singleton_rows": int(singleton_rows),
        "singleton_fraction": float(singleton_rows / len(labels)),
        "mixed_classes": int(mixed_classes),
        "collision_rows": int(collision_rows),
        "pure_rows": int(pure_rows),
        "collision_coverage": float(collision_rows / len(labels)),
        "conditional_label_entropy_bits": float(conditional_entropy),
        "label_entropy_bits": label_entropy,
        "equivalence_class_histogram": [
            {
                "negative_count": negative_count,
                "positive_count": positive_count,
                "classes": multiplicity,
            }
            for (negative_count, positive_count), multiplicity in sorted(histogram.items())
        ],
    }


def empirical_representation_ceiling(features: np.ndarray, labels: np.ndarray) -> float:
    return float(collision_conditioned_ceiling(features, labels)["ceiling"])


def extraction_efficiency(metric: float, ceiling: float, prior: float = 0.5) -> float | None:
    denominator = ceiling - prior
    if denominator <= 0.0:
        return None
    return float((metric - prior) / denominator)


def multiclass_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64)
    if probabilities.shape != (len(labels), len(classes)):
        raise ValueError("multiclass probabilities are not row/class aligned")
    if set(map(int, np.unique(labels))) - set(map(int, classes)):
        raise ValueError("evaluation contains a class absent from decoder probabilities")
    predictions = classes[np.argmax(probabilities, axis=1)]
    one_hot = (labels[:, None] == classes[None, :]).astype(np.float64)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            recall_score(
                labels,
                predictions,
                labels=np.unique(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }
    if len(classes) >= 3:
        metrics["top3_accuracy"] = float(
            top_k_accuracy_score(labels, probabilities, k=3, labels=classes)
        )
    return metrics
