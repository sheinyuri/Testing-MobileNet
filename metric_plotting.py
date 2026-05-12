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
) -> None:
    """Plot aggregate metric means, optionally with confidence intervals."""
    if not aggregate_rows:
        raise ValueError("No aggregate rows to plot.")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    criteria = sorted({str(row["criterion"]) for row in aggregate_rows})

    for index, metric in enumerate(metric_columns):
        ax = axes[index]
        for criterion in criteria:
            criterion_rows = sorted(
                [row for row in aggregate_rows if str(row["criterion"]) == criterion],
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in criterion_rows]
            means = [float(row[f"{metric} Mean"]) for row in criterion_rows]

            ax.plot(epochs, means, marker="o", linewidth=1.8, label=criterion)

            if include_confidence_intervals:
                ci_low = [float(row[f"{metric} CI Low"]) for row in criterion_rows]
                ci_high = [float(row[f"{metric} CI High"]) for row in criterion_rows]
                ax.fill_between(epochs, ci_low, ci_high, alpha=0.18)

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
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

        ax.bar(
            x_positions,
            means,
            yerr=[lower_errors, upper_errors],
            capsize=4,
            alpha=0.85,
        )
        ax.set_title(metric)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

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
    )
    plot_terminal_loss_comparison(
        aggregate_rows,
        output_path=output_dir / f"{split}_terminal_loss_comparison.png",
        show=show,
        title=f"{split.title()} Terminal Loss Comparison",
        metric_columns=metric_columns,
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


def _parse_class_counts(raw_counts: str) -> list[int]:
    counts = [int(value.strip()) for value in raw_counts.split(",") if value.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("Provide comma-separated class sample counts.")
    return counts


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
        if needs_imbalanced_counts and imbalanced_counts is None:
            parser.error(
                "--project-plots requires --imbalanced-class-counts "
                "or --class-counts for the imbalanced training distribution."
            )

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

    if args.results_folder is None or args.class_counts is None:
        parser.error("results_folder and --class-counts are required unless using --project-plots.")

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
