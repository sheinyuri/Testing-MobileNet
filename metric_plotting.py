from __future__ import annotations

import argparse
import csv
import heapq
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


METRIC_COLUMNS = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Top-3 Acc",
    "Macro Recall@3",
]

PROJECT_SEEDS = ["seed_2", "seed_18", "seed_33", "seed_45", "seed_255"]
PROJECT_LOSSES = {
    "balanced": ["ce", "focal_loss", "label_smoothing_ce"],
    "imbalanced": [
        "ce",
        "class_balanced_loss",
        "focal_loss",
        "label_smoothing_ce",
        "weighted_ce",
    ],
}


def _metric_columns(k: int = 3) -> list[str]:
    return [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        f"Top-{k} Acc",
        f"Macro Recall@{k}",
    ]


def _normalise_class_counts(
    class_sample_counts: Mapping[int, int] | Sequence[int],
    class_labels: Sequence[int],
) -> dict[int, int]:
    if isinstance(class_sample_counts, Mapping):
        counts = {int(label): int(count) for label, count in class_sample_counts.items()}
    else:
        counts = {
            int(label): int(count)
            for label, count in zip(class_labels, class_sample_counts, strict=True)
        }

    missing_labels = set(class_labels) - set(counts)
    if missing_labels:
        raise ValueError(f"Missing class sample counts for labels: {sorted(missing_labels)}")

    if any(counts[label] < 0 for label in class_labels):
        raise ValueError("Class sample counts must be non-negative.")

    if sum(counts[label] for label in class_labels) == 0:
        raise ValueError("At least one class sample count must be positive.")

    return counts


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0

    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_confusion_metrics(
    results_folder: str | Path,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    confusion_filename: str = "testing_confusion_matrices.csv",
) -> list[dict[str, float | int | str]]:
    """Compute accuracy, macro metrics, and train-distribution weighted F1.

    The weighted F1 uses ``class_sample_counts`` as the class weights. This is
    useful when the evaluation split is balanced but the training split should
    be treated as imbalanced for reporting.
    """
    results_folder = Path(results_folder)
    confusion_path = results_folder / confusion_filename
    grouped_counts: dict[tuple[str, int], dict[tuple[int, int], float]] = {}
    class_label_set: set[int] = set()

    with confusion_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            criterion = record["criterion"]
            epoch = int(record["epoch"])
            true_class = int(record["true_class"])
            predicted_class = int(record["predicted_class"])
            count = float(record["count"])

            class_label_set.update([true_class, predicted_class])
            grouped_counts.setdefault((criterion, epoch), {})[
                (true_class, predicted_class)
            ] = count

    class_labels = sorted(class_label_set)
    train_counts = _normalise_class_counts(class_sample_counts, class_labels)
    total_train_count = sum(train_counts[label] for label in class_labels)

    rows = []
    for criterion, epoch in sorted(grouped_counts, key=lambda item: (item[0], item[1])):
        counts = grouped_counts[(criterion, epoch)]
        total = sum(counts.values())

        true_positive = {
            label: counts.get((label, label), 0.0) for label in class_labels
        }
        predicted_total = {
            label: sum(counts.get((true_label, label), 0.0) for true_label in class_labels)
            for label in class_labels
        }
        actual_total = {
            label: sum(
                counts.get((label, predicted_label), 0.0)
                for predicted_label in class_labels
            )
            for label in class_labels
        }

        precision = {
            label: (
                true_positive[label] / predicted_total[label]
                if predicted_total[label]
                else 0.0
            )
            for label in class_labels
        }
        recall = {
            label: (
                true_positive[label] / actual_total[label] if actual_total[label] else 0.0
            )
            for label in class_labels
        }
        f1 = {
            label: (
                2 * precision[label] * recall[label] / (precision[label] + recall[label])
                if precision[label] + recall[label]
                else 0.0
            )
            for label in class_labels
        }

        rows.append(
            {
                "criterion": criterion,
                "epoch": int(epoch),
                "Accuracy": sum(true_positive.values()) / total if total else 0.0,
                "Macro Precision": sum(precision.values()) / len(class_labels),
                "Macro Recall": sum(recall.values()) / len(class_labels),
                "Macro F1": sum(f1.values()) / len(class_labels),
                "Weighted F1": (
                    sum(f1[label] * train_counts[label] for label in class_labels)
                    / total_train_count
                ),
            }
        )

    return rows


def compute_topk_metrics(
    results_folder: str | Path,
    k: int = 3,
    logits_filename: str = "test_logits.csv",
) -> list[dict[str, float | int | str]]:
    """Compute Top-k accuracy and macro Recall@k from saved test logits."""
    results_folder = Path(results_folder)
    logits_path = results_folder / logits_filename
    grouped_hits: dict[tuple[str, int], dict[str, object]] = {}

    with logits_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{logits_path} is empty.")

        logit_columns = sorted(
            [column for column in reader.fieldnames if column.startswith("logit_class_")],
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if not logit_columns:
            raise ValueError(f"No logit_class_* columns found in {logits_path}")
        if k < 1 or k > len(logit_columns):
            raise ValueError(f"k must be between 1 and {len(logit_columns)}.")

        class_labels = [int(column.rsplit("_", 1)[1]) for column in logit_columns]

        for record in reader:
            criterion = record["criterion"]
            epoch = int(record["epoch"])
            true_label = int(record["true_label"])
            logits = [
                (float(record[column]), int(column.rsplit("_", 1)[1]))
                for column in logit_columns
            ]
            predicted_topk = {label for _, label in heapq.nlargest(k, logits)}
            hit = true_label in predicted_topk

            grouped = grouped_hits.setdefault(
                (criterion, epoch),
                {
                    "hits": 0,
                    "total": 0,
                    "class_hits": {label: 0 for label in class_labels},
                    "class_totals": {label: 0 for label in class_labels},
                },
            )
            grouped["hits"] = int(grouped["hits"]) + int(hit)
            grouped["total"] = int(grouped["total"]) + 1
            grouped["class_hits"][true_label] += int(hit)
            grouped["class_totals"][true_label] += 1

    rows = []
    for criterion, epoch in sorted(grouped_hits, key=lambda item: (item[0], item[1])):
        grouped = grouped_hits[(criterion, epoch)]
        class_hits = grouped["class_hits"]
        class_totals = grouped["class_totals"]
        per_class_recall_at_k = [
            class_hits[label] / class_totals[label] if class_totals[label] else 0.0
            for label in class_labels
        ]

        rows.append(
            {
                "criterion": criterion,
                "epoch": int(epoch),
                f"Top-{k} Acc": grouped["hits"] / grouped["total"]
                if grouped["total"]
                else 0.0,
                f"Macro Recall@{k}": sum(per_class_recall_at_k) / len(class_labels),
            }
        )

    return rows


def _softmax_confidence(logits: Sequence[tuple[float, int]]) -> tuple[int, float]:
    """Return the predicted label and max softmax probability for one logit row."""
    max_logit, predicted_label = max(logits, key=lambda item: item[0])
    exp_sum = sum(math.exp(logit - max_logit) for logit, _ in logits)
    return predicted_label, 1.0 / exp_sum if exp_sum else 0.0


def compute_reliability_rows(
    results_folder: str | Path,
    n_bins: int = 10,
    logits_filename: str = "test_logits.csv",
) -> list[dict[str, float | int | str]]:
    """Bin epoch-level predictions by confidence for reliability diagrams.

    Confidence is computed from saved logits as max(softmax(logits)). Accuracy
    is the fraction of predictions in that confidence bin that match true_label.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    results_folder = Path(results_folder)
    logits_path = results_folder / logits_filename
    grouped_bins: dict[tuple[str, int, int], dict[str, float | int]] = {}

    with logits_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{logits_path} is empty.")

        logit_columns = sorted(
            [column for column in reader.fieldnames if column.startswith("logit_class_")],
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if not logit_columns:
            raise ValueError(f"No logit_class_* columns found in {logits_path}")

        for record in reader:
            criterion = record["criterion"]
            epoch = int(record["epoch"])
            true_label = int(record["true_label"])
            logits = [
                (float(record[column]), int(column.rsplit("_", 1)[1]))
                for column in logit_columns
            ]
            predicted_label, confidence = _softmax_confidence(logits)
            bin_index = min(n_bins - 1, int(confidence * n_bins))
            grouped = grouped_bins.setdefault(
                (criterion, epoch, bin_index),
                {"count": 0, "correct": 0, "confidence_sum": 0.0},
            )
            grouped["count"] = int(grouped["count"]) + 1
            grouped["correct"] = int(grouped["correct"]) + int(
                predicted_label == true_label
            )
            grouped["confidence_sum"] = float(grouped["confidence_sum"]) + confidence

    rows = []
    for criterion, epoch, bin_index in sorted(
        grouped_bins, key=lambda item: (item[0], item[1], item[2])
    ):
        grouped = grouped_bins[(criterion, epoch, bin_index)]
        count = int(grouped["count"])
        rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "bin": bin_index,
                "bin_low": bin_index / n_bins,
                "bin_high": (bin_index + 1) / n_bins,
                "confidence": float(grouped["confidence_sum"]) / count if count else 0.0,
                "accuracy": int(grouped["correct"]) / count if count else 0.0,
                "count": count,
            }
        )

    return rows


def compute_epoch_confidence_accuracy_rows(
    results_folder: str | Path,
    logits_filename: str = "test_logits.csv",
) -> list[dict[str, float | int | str]]:
    """Compute one confidence/accuracy point per epoch from saved logits."""
    results_folder = Path(results_folder)
    logits_path = results_folder / logits_filename
    grouped_epochs: dict[tuple[str, int], dict[str, float | int]] = {}

    with logits_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{logits_path} is empty.")

        logit_columns = sorted(
            [column for column in reader.fieldnames if column.startswith("logit_class_")],
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if not logit_columns:
            raise ValueError(f"No logit_class_* columns found in {logits_path}")

        for record in reader:
            criterion = record["criterion"]
            epoch = int(record["epoch"])
            true_label = int(record["true_label"])
            logits = [
                (float(record[column]), int(column.rsplit("_", 1)[1]))
                for column in logit_columns
            ]
            predicted_label, confidence = _softmax_confidence(logits)
            grouped = grouped_epochs.setdefault(
                (criterion, epoch),
                {"count": 0, "correct": 0, "confidence_sum": 0.0},
            )
            grouped["count"] = int(grouped["count"]) + 1
            grouped["correct"] = int(grouped["correct"]) + int(
                predicted_label == true_label
            )
            grouped["confidence_sum"] = float(grouped["confidence_sum"]) + confidence

    rows = []
    for criterion, epoch in sorted(grouped_epochs, key=lambda item: (item[0], item[1])):
        grouped = grouped_epochs[(criterion, epoch)]
        count = int(grouped["count"])
        rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "confidence": float(grouped["confidence_sum"]) / count if count else 0.0,
                "accuracy": int(grouped["correct"]) / count if count else 0.0,
                "count": count,
            }
        )

    return rows


def compute_metrics(
    results_folder: str | Path,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    k: int = 3,
) -> list[dict[str, float | int | str]]:
    """Compute all requested metrics for one training-results folder."""
    confusion_metrics = compute_confusion_metrics(results_folder, class_sample_counts)
    topk_metrics = compute_topk_metrics(results_folder, k=k)
    topk_by_key = {
        (row["criterion"], row["epoch"]): row
        for row in topk_metrics
    }

    merged_rows = []
    for confusion_row in confusion_metrics:
        key = (confusion_row["criterion"], confusion_row["epoch"])
        if key not in topk_by_key:
            continue
        merged_rows.append({**confusion_row, **topk_by_key[key]})

    return sorted(merged_rows, key=lambda row: (str(row["criterion"]), int(row["epoch"])))


def aggregate_reliability_rows(
    seed_reliability_rows: Sequence[Mapping[str, float | int | str]],
    confidence_z: float = 1.96,
) -> list[dict[str, float | int | str]]:
    """Aggregate reliability rows across seeds for each loss/epoch/bin."""
    grouped: dict[tuple[str, int, int], list[Mapping[str, float | int | str]]] = {}
    for row in seed_reliability_rows:
        grouped.setdefault(
            (str(row["criterion"]), int(row["epoch"]), int(row["bin"])),
            [],
        ).append(row)

    aggregate_rows = []
    for criterion, epoch, bin_index in sorted(
        grouped, key=lambda item: (item[0], item[1], item[2])
    ):
        rows = grouped[(criterion, epoch, bin_index)]
        accuracies = [float(row["accuracy"]) for row in rows]
        confidences = [float(row["confidence"]) for row in rows]
        counts = [int(row["count"]) for row in rows]
        accuracy_mean = _mean(accuracies)
        accuracy_std = _sample_std(accuracies)
        ci_margin = (
            confidence_z * accuracy_std / math.sqrt(len(accuracies))
            if accuracies
            else 0.0
        )

        aggregate_rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "bin": bin_index,
                "bin_low": rows[0]["bin_low"],
                "bin_high": rows[0]["bin_high"],
                "confidence": _mean(confidences),
                "accuracy": accuracy_mean,
                "accuracy_std": accuracy_std,
                "accuracy_ci_low": max(0.0, accuracy_mean - ci_margin),
                "accuracy_ci_high": min(1.0, accuracy_mean + ci_margin),
                "count": sum(counts),
                "n_seeds": len({str(row["seed"]) for row in rows if "seed" in row}),
            }
        )

    return aggregate_rows


def aggregate_epoch_confidence_accuracy_rows(
    seed_rows: Sequence[Mapping[str, float | int | str]],
    confidence_z: float = 1.96,
) -> list[dict[str, float | int | str]]:
    """Aggregate epoch-level confidence/accuracy points across seeds."""
    grouped: dict[tuple[str, int], list[Mapping[str, float | int | str]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["criterion"]), int(row["epoch"])), []).append(row)

    aggregate_rows = []
    for criterion, epoch in sorted(grouped, key=lambda item: (item[0], item[1])):
        rows = grouped[(criterion, epoch)]
        accuracies = [float(row["accuracy"]) for row in rows]
        confidences = [float(row["confidence"]) for row in rows]
        accuracy_mean = _mean(accuracies)
        confidence_mean = _mean(confidences)
        accuracy_std = _sample_std(accuracies)
        confidence_std = _sample_std(confidences)
        accuracy_ci = (
            confidence_z * accuracy_std / math.sqrt(len(accuracies))
            if accuracies
            else 0.0
        )
        confidence_ci = (
            confidence_z * confidence_std / math.sqrt(len(confidences))
            if confidences
            else 0.0
        )

        aggregate_rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "confidence": confidence_mean,
                "confidence_std": confidence_std,
                "confidence_ci_low": max(0.0, confidence_mean - confidence_ci),
                "confidence_ci_high": min(1.0, confidence_mean + confidence_ci),
                "accuracy": accuracy_mean,
                "accuracy_std": accuracy_std,
                "accuracy_ci_low": max(0.0, accuracy_mean - accuracy_ci),
                "accuracy_ci_high": min(1.0, accuracy_mean + accuracy_ci),
                "count": sum(int(row["count"]) for row in rows),
                "n_seeds": len({str(row["seed"]) for row in rows if "seed" in row}),
            }
        )

    return aggregate_rows


def aggregate_seed_metrics(
    seed_metric_rows: Sequence[Mapping[str, float | int | str]],
    metric_columns: Sequence[str] = METRIC_COLUMNS,
    confidence_z: float = 1.96,
) -> list[dict[str, float | int | str]]:
    """Aggregate seed-level metric rows into mean/std/CI rows.

    Confidence intervals are normal-approximation 95% intervals by default:
    mean +/- 1.96 * sample_std / sqrt(number_of_seeds).
    """
    grouped: dict[tuple[str, int], list[Mapping[str, float | int | str]]] = {}
    for row in seed_metric_rows:
        grouped.setdefault((str(row["criterion"]), int(row["epoch"])), []).append(row)

    aggregate_rows = []
    for criterion, epoch in sorted(grouped, key=lambda item: (item[0], item[1])):
        rows = grouped[(criterion, epoch)]
        aggregate_row: dict[str, float | int | str] = {
            "criterion": criterion,
            "epoch": epoch,
            "n_seeds": len({str(row["seed"]) for row in rows if "seed" in row}),
        }

        for metric in metric_columns:
            values = [float(row[metric]) for row in rows]
            mean = _mean(values)
            std = _sample_std(values)
            ci_margin = confidence_z * std / math.sqrt(len(values)) if values else 0.0

            aggregate_row[f"{metric} Mean"] = mean
            aggregate_row[f"{metric} Std"] = std
            aggregate_row[f"{metric} CI Low"] = max(0.0, mean - ci_margin)
            aggregate_row[f"{metric} CI High"] = min(1.0, mean + ci_margin)

        aggregate_rows.append(aggregate_row)

    return aggregate_rows


def plot_metrics(
    metrics_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path | None = None,
    show: bool = False,
    title: str = "Evaluation Metrics",
    metric_columns: Sequence[str] = METRIC_COLUMNS,
) -> None:
    """Plot the requested metrics over epochs for each criterion in the data."""
    if not metrics_rows:
        raise ValueError("No metrics rows to plot.")

    available_columns = set(metrics_rows[0])
    missing = [column for column in metric_columns if column not in available_columns]
    if missing:
        raise ValueError(f"Metrics dataframe is missing columns: {missing}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    criteria = sorted({str(row["criterion"]) for row in metrics_rows})

    for index, metric in enumerate(metric_columns):
        ax = axes[index]
        for criterion in criteria:
            criterion_rows = sorted(
                [row for row in metrics_rows if str(row["criterion"]) == criterion],
                key=lambda row: int(row["epoch"]),
            )
            ax.plot(
                [int(row["epoch"]) for row in criterion_rows],
                [float(row[metric]) for row in criterion_rows],
                marker="o",
                linewidth=1.8,
                label=criterion,
            )
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)

    for unused_ax in axes[len(metric_columns) :]:
        unused_ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_aggregate_metrics(
    aggregate_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    show: bool = False,
    title: str = "Aggregate Evaluation Metrics",
    metric_columns: Sequence[str] = METRIC_COLUMNS,
    include_confidence_intervals: bool = True,
    zoom_to_data: bool = False,
) -> None:
    """Plot aggregate metric means, optionally with confidence intervals."""
    if not aggregate_rows:
        raise ValueError("No aggregate rows to plot.")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    criteria = sorted({str(row["criterion"]) for row in aggregate_rows})

    for index, metric in enumerate(metric_columns):
        ax = axes[index]
        y_values = []
        for criterion in criteria:
            criterion_rows = sorted(
                [row for row in aggregate_rows if str(row["criterion"]) == criterion],
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in criterion_rows]
            means = [float(row[f"{metric} Mean"]) for row in criterion_rows]
            y_values.extend(means)

            ax.plot(epochs, means, marker="o", linewidth=1.8, label=criterion)

            if include_confidence_intervals:
                ci_low = [float(row[f"{metric} CI Low"]) for row in criterion_rows]
                ci_high = [float(row[f"{metric} CI High"]) for row in criterion_rows]
                y_values.extend(ci_low)
                y_values.extend(ci_high)
                ax.fill_between(epochs, ci_low, ci_high, alpha=0.18)

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        if zoom_to_data:
            y_min = min(y_values)
            y_max = max(y_values)
            if y_min == y_max:
                y_min -= 0.01
                y_max += 0.01
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)

    for unused_ax in axes[len(metric_columns) :]:
        unused_ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _select_reliability_epochs(
    reliability_rows: Sequence[Mapping[str, float | int | str]],
    epochs: Sequence[int] | None,
    max_epoch_curves: int,
) -> list[int]:
    available_epochs = sorted({int(row["epoch"]) for row in reliability_rows})
    if epochs:
        requested = [int(epoch) for epoch in epochs]
        missing = sorted(set(requested) - set(available_epochs))
        if missing:
            raise ValueError(f"Requested epochs not found in reliability rows: {missing}")
        return requested

    if len(available_epochs) <= max_epoch_curves:
        return available_epochs

    positions = [
        round(index * (len(available_epochs) - 1) / (max_epoch_curves - 1))
        for index in range(max_epoch_curves)
    ]
    return [available_epochs[position] for position in positions]


def plot_reliability_diagram(
    reliability_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    show: bool = False,
    title: str = "Reliability Diagram",
    epochs: Sequence[int] | None = None,
    max_epoch_curves: int = 5,
) -> None:
    """Plot accuracy vs confidence curves, with separate lines per epoch."""
    if not reliability_rows:
        raise ValueError("No reliability rows to plot.")

    selected_epochs = _select_reliability_epochs(
        reliability_rows,
        epochs=epochs,
        max_epoch_curves=max_epoch_curves,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.2, label="Perfect calibration")

    for epoch in selected_epochs:
        epoch_rows = sorted(
            [row for row in reliability_rows if int(row["epoch"]) == epoch],
            key=lambda row: int(row["bin"]),
        )
        ax.plot(
            [float(row["confidence"]) for row in epoch_rows],
            [float(row["accuracy"]) for row in epoch_rows],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=f"Epoch {epoch}",
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_epoch_confidence_accuracy(
    epoch_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    show: bool = False,
    title: str = "Accuracy vs Confidence Over Epochs",
    annotate_epochs: bool = True,
) -> None:
    """Plot one line where each point is one epoch's mean confidence and accuracy."""
    if not epoch_rows:
        raise ValueError("No epoch confidence/accuracy rows to plot.")

    criteria = sorted({str(row["criterion"]) for row in epoch_rows})
    if len(criteria) != 1:
        raise ValueError(
            "plot_epoch_confidence_accuracy expects rows for exactly one criterion."
        )

    rows = sorted(epoch_rows, key=lambda row: int(row["epoch"]))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.2)
    ax.plot(
        [float(row["confidence"]) for row in rows],
        [float(row["accuracy"]) for row in rows],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=str(criteria[0]),
    )

    if annotate_epochs:
        for row in rows:
            ax.annotate(
                str(int(row["epoch"])),
                (float(row["confidence"]), float(row["accuracy"])),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
                alpha=0.75,
            )

    ax.set_xlabel("Average Confidence")
    ax.set_ylabel("Average Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_epoch_confidence_accuracy_comparison(
    epoch_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    show: bool = False,
    title: str = "Accuracy vs Confidence Over Epochs",
) -> None:
    """Plot one epoch confidence/accuracy line for each criterion."""
    if not epoch_rows:
        raise ValueError("No epoch confidence/accuracy rows to plot.")

    criteria = sorted({str(row["criterion"]) for row in epoch_rows})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.2)

    for criterion in criteria:
        rows = sorted(
            [row for row in epoch_rows if str(row["criterion"]) == criterion],
            key=lambda row: int(row["epoch"]),
        )
        ax.plot(
            [float(row["confidence"]) for row in rows],
            [float(row["accuracy"]) for row in rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=criterion,
        )

    ax.set_xlabel("Average Confidence")
    ax.set_ylabel("Average Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_terminal_loss_comparison(
    aggregate_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    show: bool = False,
    title: str = "Terminal Loss Comparison",
    metric_columns: Sequence[str] = METRIC_COLUMNS,
) -> None:
    """Plot final-epoch metric means and confidence intervals by loss function."""
    if not aggregate_rows:
        raise ValueError("No aggregate rows to plot.")

    terminal_rows = []
    for criterion in sorted({str(row["criterion"]) for row in aggregate_rows}):
        criterion_rows = [
            row for row in aggregate_rows if str(row["criterion"]) == criterion
        ]
        terminal_rows.append(max(criterion_rows, key=lambda row: int(row["epoch"])))

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    labels = [str(row["criterion"]) for row in terminal_rows]
    x_positions = list(range(len(labels)))

    for index, metric in enumerate(metric_columns):
        ax = axes[index]
        means = [float(row[f"{metric} Mean"]) for row in terminal_rows]
        ci_low = [float(row[f"{metric} CI Low"]) for row in terminal_rows]
        ci_high = [float(row[f"{metric} CI High"]) for row in terminal_rows]
        lower_errors = [mean - low for mean, low in zip(means, ci_low, strict=True)]
        upper_errors = [high - mean for mean, high in zip(means, ci_high, strict=True)]

        ax.errorbar(
            x_positions,
            means,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=1.8,
            capsize=4,
            capthick=1.2,
        )
        ax.set_title(metric)
        y_min = min(ci_low)
        y_max = max(ci_high)
        y_range = y_max - y_min
        y_padding = y_range * 0.05 if y_range > 0 else 0.01
        ax.set_ylim(max(0.0, y_min - y_padding), min(1.0, y_max + y_padding))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.margins(x=0.02)
        ax.grid(True, linestyle="--", alpha=0.35)

    for unused_ax in axes[len(metric_columns) :]:
        unused_ax.axis("off")

    final_epochs = sorted({int(row["epoch"]) for row in terminal_rows})
    epoch_note = (
        f"Terminal Epoch {final_epochs[0]}"
        if len(final_epochs) == 1
        else f"Terminal Epochs {final_epochs}"
    )
    fig.suptitle(f"{title} ({epoch_note})", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def write_rows_csv(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_fieldnames(metric_columns: Sequence[str] = METRIC_COLUMNS) -> list[str]:
    fieldnames = ["criterion", "epoch", "n_seeds"]
    for metric in metric_columns:
        fieldnames.extend(
            [
                f"{metric} Mean",
                f"{metric} Std",
                f"{metric} CI Low",
                f"{metric} CI High",
            ]
        )
    return fieldnames


def find_missing_project_metric_files(
    split: str,
    project_root: str | Path = ".",
    seeds: Sequence[str] = PROJECT_SEEDS,
) -> list[Path]:
    """Return required metric files missing from the fixed project layout."""
    project_root = Path(project_root)
    if split not in PROJECT_LOSSES:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(PROJECT_LOSSES)}.")

    missing_files = []
    for loss_name in PROJECT_LOSSES[split]:
        for seed in seeds:
            results_folder = project_root / "training_data" / split / loss_name / seed
            for filename in ["testing_confusion_matrices.csv", "test_logits.csv"]:
                path = results_folder / filename
                if not path.exists():
                    missing_files.append(path)

    return missing_files


def compute_project_split_metrics(
    split: str,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    project_root: str | Path = ".",
    seeds: Sequence[str] = PROJECT_SEEDS,
    k: int = 3,
) -> list[dict[str, float | int | str]]:
    """Compute seed-level metrics for one fixed project split."""
    project_root = Path(project_root)
    if split not in PROJECT_LOSSES:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(PROJECT_LOSSES)}.")

    missing_files = find_missing_project_metric_files(split, project_root, seeds)
    if missing_files:
        formatted_missing = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Cannot compute {split} aggregate metrics; missing files:\n"
            f"{formatted_missing}"
        )

    seed_rows = []
    for loss_name in PROJECT_LOSSES[split]:
        for seed in seeds:
            results_folder = project_root / "training_data" / split / loss_name / seed
            rows = compute_metrics(results_folder, class_sample_counts, k=k)
            for row in rows:
                row["criterion"] = loss_name
                row["seed"] = seed
                row["split"] = split
                seed_rows.append(row)

    return seed_rows


def plot_project_split_metrics(
    split: str,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    project_root: str | Path = ".",
    output_dir: str | Path = "plots/aggregate_metrics",
    seeds: Sequence[str] = PROJECT_SEEDS,
    k: int = 3,
    confidence_z: float = 1.96,
    show: bool = False,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Create per-loss CI plots plus one all-loss mean-only plot for a split."""
    metric_columns = _metric_columns(k)
    output_dir = Path(output_dir) / split

    seed_rows = compute_project_split_metrics(
        split,
        class_sample_counts,
        project_root=project_root,
        seeds=seeds,
        k=k,
    )
    aggregate_rows = aggregate_seed_metrics(
        seed_rows,
        metric_columns=metric_columns,
        confidence_z=confidence_z,
    )

    write_rows_csv(
        seed_rows,
        output_dir / f"{split}_seed_metrics.csv",
        fieldnames=["split", "criterion", "seed", "epoch", *metric_columns],
    )
    write_rows_csv(
        aggregate_rows,
        output_dir / f"{split}_aggregate_metrics.csv",
        fieldnames=aggregate_fieldnames(metric_columns),
    )

    for loss_name in PROJECT_LOSSES[split]:
        loss_rows = [
            row for row in aggregate_rows if str(row["criterion"]) == loss_name
        ]
        plot_aggregate_metrics(
            loss_rows,
            output_path=output_dir / f"{loss_name}_mean_ci.png",
            show=show,
            title=f"{split.title()} {loss_name}: Mean with 95% CI",
            metric_columns=metric_columns,
            include_confidence_intervals=True,
        )

    plot_aggregate_metrics(
        aggregate_rows,
        output_path=output_dir / f"{split}_all_losses_mean.png",
        show=show,
        title=f"{split.title()} Loss Comparison: Mean with 95% CI",
        metric_columns=metric_columns,
        include_confidence_intervals=True,
        zoom_to_data=True,
    )
    plot_terminal_loss_comparison(
        aggregate_rows,
        output_path=output_dir / f"{split}_terminal_loss_comparison.png",
        show=show,
        title=f"{split.title()} Terminal Loss Comparison",
        metric_columns=metric_columns,
    )

    return seed_rows, aggregate_rows


def compute_project_split_reliability(
    split: str,
    project_root: str | Path = ".",
    seeds: Sequence[str] = PROJECT_SEEDS,
    n_bins: int = 10,
) -> list[dict[str, float | int | str]]:
    """Compute seed-level reliability rows for one fixed project split."""
    project_root = Path(project_root)
    if split not in PROJECT_LOSSES:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(PROJECT_LOSSES)}.")

    missing_files = []
    for loss_name in PROJECT_LOSSES[split]:
        for seed in seeds:
            logits_path = (
                project_root
                / "training_data"
                / split
                / loss_name
                / seed
                / "test_logits.csv"
            )
            if not logits_path.exists():
                missing_files.append(logits_path)
    if missing_files:
        formatted_missing = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Cannot compute {split} reliability plots; missing files:\n"
            f"{formatted_missing}"
        )

    reliability_rows = []
    for loss_name in PROJECT_LOSSES[split]:
        for seed in seeds:
            results_folder = project_root / "training_data" / split / loss_name / seed
            rows = compute_reliability_rows(results_folder, n_bins=n_bins)
            for row in rows:
                row["criterion"] = loss_name
                row["seed"] = seed
                row["split"] = split
                reliability_rows.append(row)

    return reliability_rows


def plot_project_split_reliability(
    split: str,
    project_root: str | Path = ".",
    output_dir: str | Path = "plots/reliability",
    seeds: Sequence[str] = PROJECT_SEEDS,
    n_bins: int = 10,
    epochs: Sequence[int] | None = None,
    max_epoch_curves: int = 5,
    confidence_z: float = 1.96,
    show: bool = False,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Create loss-wise reliability diagrams for a fixed project split."""
    output_dir = Path(output_dir) / split
    seed_rows = compute_project_split_reliability(
        split,
        project_root=project_root,
        seeds=seeds,
        n_bins=n_bins,
    )
    aggregate_rows = aggregate_reliability_rows(
        seed_rows,
        confidence_z=confidence_z,
    )

    reliability_fieldnames = [
        "split",
        "criterion",
        "seed",
        "epoch",
        "bin",
        "bin_low",
        "bin_high",
        "confidence",
        "accuracy",
        "count",
    ]
    write_rows_csv(
        seed_rows,
        output_dir / f"{split}_seed_reliability.csv",
        fieldnames=reliability_fieldnames,
    )
    write_rows_csv(
        aggregate_rows,
        output_dir / f"{split}_aggregate_reliability.csv",
        fieldnames=[
            "criterion",
            "epoch",
            "bin",
            "bin_low",
            "bin_high",
            "confidence",
            "accuracy",
            "accuracy_std",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "count",
            "n_seeds",
        ],
    )

    for loss_name in PROJECT_LOSSES[split]:
        loss_rows = [
            row for row in aggregate_rows if str(row["criterion"]) == loss_name
        ]
        plot_reliability_diagram(
            loss_rows,
            output_path=output_dir / f"{loss_name}_reliability.png",
            show=show,
            title=f"{split.title()} {loss_name}: Accuracy vs Confidence",
            epochs=epochs,
            max_epoch_curves=max_epoch_curves,
        )

    return seed_rows, aggregate_rows


def compute_project_split_epoch_confidence_accuracy(
    split: str,
    project_root: str | Path = ".",
    seeds: Sequence[str] = PROJECT_SEEDS,
) -> list[dict[str, float | int | str]]:
    """Compute seed-level epoch confidence/accuracy rows for one project split."""
    project_root = Path(project_root)
    if split not in PROJECT_LOSSES:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(PROJECT_LOSSES)}.")

    missing_files = []
    for loss_name in PROJECT_LOSSES[split]:
        for seed in seeds:
            logits_path = (
                project_root
                / "training_data"
                / split
                / loss_name
                / seed
                / "test_logits.csv"
            )
            if not logits_path.exists():
                missing_files.append(logits_path)
    if missing_files:
        formatted_missing = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Cannot compute {split} epoch confidence/accuracy plots; missing files:\n"
            f"{formatted_missing}"
        )

    seed_rows = []
    for loss_name in PROJECT_LOSSES[split]:
        for seed in seeds:
            results_folder = project_root / "training_data" / split / loss_name / seed
            rows = compute_epoch_confidence_accuracy_rows(results_folder)
            for row in rows:
                row["criterion"] = loss_name
                row["seed"] = seed
                row["split"] = split
                seed_rows.append(row)

    return seed_rows


def plot_project_split_epoch_confidence_accuracy(
    split: str,
    project_root: str | Path = ".",
    output_dir: str | Path = "plots/epoch_confidence_accuracy",
    seeds: Sequence[str] = PROJECT_SEEDS,
    confidence_z: float = 1.96,
    show: bool = False,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Create one-line epoch confidence/accuracy plots for each loss in a split."""
    output_dir = Path(output_dir) / split
    seed_rows = compute_project_split_epoch_confidence_accuracy(
        split,
        project_root=project_root,
        seeds=seeds,
    )
    aggregate_rows = aggregate_epoch_confidence_accuracy_rows(
        seed_rows,
        confidence_z=confidence_z,
    )

    write_rows_csv(
        seed_rows,
        output_dir / f"{split}_seed_epoch_confidence_accuracy.csv",
        fieldnames=[
            "split",
            "criterion",
            "seed",
            "epoch",
            "confidence",
            "accuracy",
            "count",
        ],
    )
    write_rows_csv(
        aggregate_rows,
        output_dir / f"{split}_aggregate_epoch_confidence_accuracy.csv",
        fieldnames=[
            "criterion",
            "epoch",
            "confidence",
            "confidence_std",
            "confidence_ci_low",
            "confidence_ci_high",
            "accuracy",
            "accuracy_std",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "count",
            "n_seeds",
        ],
    )

    for loss_name in PROJECT_LOSSES[split]:
        loss_rows = [
            row for row in aggregate_rows if str(row["criterion"]) == loss_name
        ]
        plot_epoch_confidence_accuracy(
            loss_rows,
            output_path=output_dir / f"{loss_name}_epoch_confidence_accuracy.png",
            show=show,
            title=f"{split.title()} {loss_name}: Accuracy vs Confidence Over Epochs",
        )

    plot_epoch_confidence_accuracy_comparison(
        aggregate_rows,
        output_path=output_dir / f"{split}_all_losses_epoch_confidence_accuracy.png",
        show=show,
        title=f"{split.title()}: Accuracy vs Confidence Over Epochs",
    )

    return seed_rows, aggregate_rows


def plot_project_metrics(
    balanced_class_counts: Mapping[int, int] | Sequence[int],
    imbalanced_class_counts: Mapping[int, int] | Sequence[int],
    project_root: str | Path = ".",
    output_dir: str | Path = "plots/aggregate_metrics",
    seeds: Sequence[str] = PROJECT_SEEDS,
    k: int = 3,
    confidence_z: float = 1.96,
    show: bool = False,
) -> dict[str, tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]]:
    """Generate all fixed project plots for balanced and imbalanced folders."""
    split_counts = {
        "balanced": balanced_class_counts,
        "imbalanced": imbalanced_class_counts,
    }
    all_missing_files = []
    for split in split_counts:
        all_missing_files.extend(find_missing_project_metric_files(split, project_root, seeds))
    if all_missing_files:
        formatted_missing = "\n".join(f"  - {path}" for path in all_missing_files)
        raise FileNotFoundError(
            "Cannot generate all project aggregate plots; missing files:\n"
            f"{formatted_missing}"
        )

    return {
        "balanced": plot_project_split_metrics(
            "balanced",
            balanced_class_counts,
            project_root=project_root,
            output_dir=output_dir,
            seeds=seeds,
            k=k,
            confidence_z=confidence_z,
            show=show,
        ),
        "imbalanced": plot_project_split_metrics(
            "imbalanced",
            imbalanced_class_counts,
            project_root=project_root,
            output_dir=output_dir,
            seeds=seeds,
            k=k,
            confidence_z=confidence_z,
            show=show,
        ),
    }


def plot_results_folder(
    results_folder: str | Path,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    output_path: str | Path | None = None,
    k: int = 3,
    show: bool = False,
) -> list[dict[str, float | int | str]]:
    """Compute and plot metrics for a folder such as training_data/balanced/ce/seed_2."""
    results_folder = Path(results_folder)
    metrics_rows = compute_metrics(results_folder, class_sample_counts, k=k)

    if output_path is None:
        output_path = Path("plots") / f"{results_folder.name}_metrics.png"

    plot_metrics(
        metrics_rows,
        output_path=output_path,
        show=show,
        title=f"Evaluation Metrics: {results_folder}",
        metric_columns=_metric_columns(k),
    )
    return metrics_rows


def plot_results_folder_reliability(
    results_folder: str | Path,
    output_path: str | Path | None = None,
    n_bins: int = 10,
    epochs: Sequence[int] | None = None,
    max_epoch_curves: int = 5,
    show: bool = False,
) -> list[dict[str, float | int | str]]:
    """Compute and plot reliability rows for one results folder."""
    results_folder = Path(results_folder)
    reliability_rows = compute_reliability_rows(results_folder, n_bins=n_bins)

    if output_path is None:
        output_path = Path("plots") / f"{results_folder.name}_reliability.png"

    criteria = sorted({str(row["criterion"]) for row in reliability_rows})
    title_criterion = criteria[0] if len(criteria) == 1 else "Model"
    plot_reliability_diagram(
        reliability_rows,
        output_path=output_path,
        show=show,
        title=f"{title_criterion}: Accuracy vs Confidence",
        epochs=epochs,
        max_epoch_curves=max_epoch_curves,
    )
    return reliability_rows


def plot_results_folder_epoch_confidence_accuracy(
    results_folder: str | Path,
    output_path: str | Path | None = None,
    show: bool = False,
) -> list[dict[str, float | int | str]]:
    """Compute and plot one epoch confidence/accuracy line for one results folder."""
    results_folder = Path(results_folder)
    rows = compute_epoch_confidence_accuracy_rows(results_folder)

    if output_path is None:
        output_path = Path("plots") / f"{results_folder.name}_epoch_confidence_accuracy.png"

    criteria = sorted({str(row["criterion"]) for row in rows})
    title_criterion = criteria[0] if len(criteria) == 1 else "Model"
    plot_epoch_confidence_accuracy(
        rows,
        output_path=output_path,
        show=show,
        title=f"{title_criterion}: Accuracy vs Confidence Over Epochs",
    )
    return rows


def _parse_class_counts(raw_counts: str) -> list[int]:
    counts = [int(value.strip()) for value in raw_counts.split(",") if value.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("Provide comma-separated class sample counts.")
    return counts


def _parse_epochs(raw_epochs: str) -> list[int]:
    epochs = [int(value.strip()) for value in raw_epochs.split(",") if value.strip()]
    if not epochs:
        raise argparse.ArgumentTypeError("Provide comma-separated epoch numbers.")
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accuracy, macro metrics, weighted F1, and top-3 metrics."
    )
    parser.add_argument(
        "results_folder",
        nargs="?",
        help="Folder containing the training CSV files.",
    )
    parser.add_argument(
        "--class-counts",
        type=_parse_class_counts,
        help="Comma-separated training sample counts by class label order, e.g. 5000,500,50.",
    )
    parser.add_argument(
        "--project-plots",
        action="store_true",
        help="Generate fixed balanced/imbalanced aggregate plots for all project losses.",
    )
    parser.add_argument(
        "--reliability",
        action="store_true",
        help="Plot accuracy vs confidence reliability diagrams from test logits.",
    )
    parser.add_argument(
        "--epoch-confidence-accuracy",
        action="store_true",
        help="Plot one confidence/accuracy point per epoch from test logits.",
    )
    parser.add_argument(
        "--project-split",
        choices=["balanced", "imbalanced", "both"],
        default="both",
        help="Which fixed project split to plot. Defaults to both.",
    )
    parser.add_argument(
        "--balanced-class-counts",
        type=_parse_class_counts,
        default=[5000] * 10,
        help="Comma-separated training counts for balanced folders. Defaults to CIFAR-10 5000 each.",
    )
    parser.add_argument(
        "--imbalanced-class-counts",
        type=_parse_class_counts,
        help="Comma-separated training counts for imbalanced folders.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root for fixed training_data folders. Defaults to current directory.",
    )
    parser.add_argument(
        "--aggregate-output-dir",
        default="plots/aggregate_metrics",
        help="Directory for fixed project aggregate plots and CSVs.",
    )
    parser.add_argument(
        "--confidence-z",
        type=float,
        default=1.96,
        help="Z value for confidence intervals. Defaults to 1.96 for about 95%%.",
    )
    parser.add_argument("--output", default=None, help="Output image path.")
    parser.add_argument("--k", type=int, default=3, help="Top-k value. Defaults to 3.")
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of confidence bins for reliability plots. Defaults to 10.",
    )
    parser.add_argument(
        "--epochs",
        type=_parse_epochs,
        default=None,
        help="Comma-separated epoch curves to show in reliability plots.",
    )
    parser.add_argument(
        "--max-epoch-curves",
        type=int,
        default=5,
        help="Maximum epoch curves to auto-select for reliability plots. Defaults to 5.",
    )
    parser.add_argument(
        "--reliability-output-dir",
        default="plots/reliability",
        help="Directory for fixed project reliability plots and CSVs.",
    )
    parser.add_argument(
        "--epoch-confidence-output-dir",
        default="plots/epoch_confidence_accuracy",
        help="Directory for epoch confidence/accuracy plots and CSVs.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional path to save the computed metrics as CSV.",
    )
    args = parser.parse_args()

    if args.project_plots:
        imbalanced_counts = args.imbalanced_class_counts or args.class_counts
        needs_imbalanced_counts = args.project_split in {"imbalanced", "both"}
        if (
            needs_imbalanced_counts
            and imbalanced_counts is None
            and not args.reliability
            and not args.epoch_confidence_accuracy
        ):
            parser.error(
                "--project-plots requires --imbalanced-class-counts "
                "or --class-counts for the imbalanced training distribution."
            )

        if args.epoch_confidence_accuracy:
            if args.project_split in {"balanced", "both"}:
                plot_project_split_epoch_confidence_accuracy(
                    "balanced",
                    project_root=args.project_root,
                    output_dir=args.epoch_confidence_output_dir,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
            if args.project_split in {"imbalanced", "both"}:
                plot_project_split_epoch_confidence_accuracy(
                    "imbalanced",
                    project_root=args.project_root,
                    output_dir=args.epoch_confidence_output_dir,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
        elif args.reliability:
            if args.project_split in {"balanced", "both"}:
                plot_project_split_reliability(
                    "balanced",
                    project_root=args.project_root,
                    output_dir=args.reliability_output_dir,
                    n_bins=args.bins,
                    epochs=args.epochs,
                    max_epoch_curves=args.max_epoch_curves,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
            if args.project_split in {"imbalanced", "both"}:
                plot_project_split_reliability(
                    "imbalanced",
                    project_root=args.project_root,
                    output_dir=args.reliability_output_dir,
                    n_bins=args.bins,
                    epochs=args.epochs,
                    max_epoch_curves=args.max_epoch_curves,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
        else:
            if args.project_split == "balanced":
                plot_project_split_metrics(
                    "balanced",
                    args.balanced_class_counts,
                    project_root=args.project_root,
                    output_dir=args.aggregate_output_dir,
                    k=args.k,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
            elif args.project_split == "imbalanced":
                plot_project_split_metrics(
                    "imbalanced",
                    imbalanced_counts,
                    project_root=args.project_root,
                    output_dir=args.aggregate_output_dir,
                    k=args.k,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
            else:
                plot_project_metrics(
                    balanced_class_counts=args.balanced_class_counts,
                    imbalanced_class_counts=imbalanced_counts,
                    project_root=args.project_root,
                    output_dir=args.aggregate_output_dir,
                    k=args.k,
                    confidence_z=args.confidence_z,
                    show=args.show,
                )
        return

    if args.results_folder is None:
        parser.error("results_folder is required unless using --project-plots.")

    if args.epoch_confidence_accuracy:
        epoch_rows = plot_results_folder_epoch_confidence_accuracy(
            args.results_folder,
            output_path=args.output,
            show=args.show,
        )
        if args.metrics_csv:
            write_rows_csv(
                epoch_rows,
                args.metrics_csv,
                fieldnames=[
                    "criterion",
                    "epoch",
                    "confidence",
                    "accuracy",
                    "count",
                ],
            )
        return

    if args.reliability:
        reliability_rows = plot_results_folder_reliability(
            args.results_folder,
            output_path=args.output,
            n_bins=args.bins,
            epochs=args.epochs,
            max_epoch_curves=args.max_epoch_curves,
            show=args.show,
        )
        if args.metrics_csv:
            write_rows_csv(
                reliability_rows,
                args.metrics_csv,
                fieldnames=[
                    "criterion",
                    "epoch",
                    "bin",
                    "bin_low",
                    "bin_high",
                    "confidence",
                    "accuracy",
                    "count",
                ],
            )
        return

    if args.class_counts is None:
        parser.error("--class-counts is required unless using --reliability.")

    metrics_rows = plot_results_folder(
        args.results_folder,
        class_sample_counts=args.class_counts,
        output_path=args.output,
        k=args.k,
        show=args.show,
    )

    if args.metrics_csv:
        metrics_csv_path = Path(args.metrics_csv)
        metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["criterion", "epoch", *_metric_columns(args.k)],
            )
            writer.writeheader()
            writer.writerows(metrics_rows)


if __name__ == "__main__":
    main()
