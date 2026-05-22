from __future__ import annotations

import csv
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image, ImageDraw


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_utc")


def create_run_dir(base_dir: Path, label: str) -> Path:
    run_dir = base_dir / f"{utc_timestamp()}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_training_config(fold_dir: Path) -> dict:
    config_path = fold_dir / "training_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _parse_yolo_label_file(label_path: Path, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []
    if not label_path.exists():
        return detections

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id, cx, cy, bw, bh = parts[:5]
        cx_f, cy_f, bw_f, bh_f = map(float, (cx, cy, bw, bh))
        x1 = (cx_f - bw_f / 2.0) * width
        y1 = (cy_f - bh_f / 2.0) * height
        x2 = (cx_f + bw_f / 2.0) * width
        y2 = (cy_f + bh_f / 2.0) * height
        detections.append(
            {
                "class_id": int(float(class_id)),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return detections


def _select_checkpoint(fold_dir: Path) -> Path | None:
    for name in ["checkpoint_best_total.pth", "checkpoint_best_regular.pth", "checkpoint_best_ema.pth"]:
        path = fold_dir / name
        if path.exists():
            return path
    return None


def _draw_legend(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((12, 12, 270, 70), outline="white", width=2, fill=(0, 0, 0, 180))
    draw.text((22, 20), "Green: ground truth", fill="lime")
    draw.text((22, 42), "Red: prediction", fill="red")


def _render_overlay(
    image: Image.Image,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    class_names: list[str],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for gt in gt_boxes:
        x1, y1, x2, y2 = gt["xyxy"]
        class_name = class_names[gt["class_id"]] if gt["class_id"] < len(class_names) else str(gt["class_id"])
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"GT {class_name}", fill="lime")

    for pred in pred_boxes:
        x1, y1, x2, y2 = pred["xyxy"]
        confidence = pred.get("confidence")
        class_id = pred.get("class_id", 0)
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        label = f"PR {class_name}"
        if confidence is not None:
            label += f" {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), label, fill="red")

    _draw_legend(draw)
    return canvas


def _generate_visual_samples(fold_dir: Path, training_config: dict) -> None:
    checkpoint = _select_checkpoint(fold_dir)
    dataset_dir_value = training_config.get("train_config", {}).get("dataset_dir")
    if checkpoint is None or not dataset_dir_value:
        return

    dataset_dir = Path(dataset_dir_value)
    valid_dir = dataset_dir / "valid"
    image_dir = valid_dir / "images"
    label_dir = valid_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return

    png_images = sorted(image_dir.glob("*.png"))[:4]
    if not png_images:
        return

    from rfdetr import RFDETRSmall

    class_names = training_config.get("class_names") or ["damage"]
    num_classes = training_config.get("num_classes") or len(class_names)
    model = RFDETRSmall.from_checkpoint(str(checkpoint), device="cpu", num_classes=num_classes)

    samples_dir = fold_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[list[object]] = []

    for idx, image_path in enumerate(png_images, start=1):
        image = Image.open(image_path).convert("RGB")
        gt_boxes = _parse_yolo_label_file(label_dir / f"{image_path.stem}.txt", image.size)

        detections = model.predict(image, threshold=0.01)
        pred_boxes = []
        for box_idx in range(len(detections.xyxy)):
            class_id = 0
            if detections.class_id is not None:
                class_id = int(detections.class_id[box_idx])
            confidence = None
            if detections.confidence is not None:
                confidence = float(detections.confidence[box_idx])
            pred_boxes.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "xyxy": [float(v) for v in detections.xyxy[box_idx]],
                }
            )

        overlay = _render_overlay(image, gt_boxes, pred_boxes, class_names)
        output_name = f"sample_{idx:02d}_{image_path.stem}.png"
        overlay.save(samples_dir / output_name)
        sample_rows.append([output_name, len(gt_boxes), len(pred_boxes)])

    index_lines = [
        "# Visual Samples",
        "",
        _write_markdown_table(["File", "GT Boxes", "Pred Boxes"], sample_rows),
    ]
    (samples_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def collect_environment(run_type: str, data_dir: Path) -> dict:
    versions: dict[str, str | None] = {}
    for package in ["rfdetr", "torch", "pytorch-lightning", "matplotlib"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dataset_dir": str(data_dir.resolve()),
        "package_versions": versions,
    }


def _split_epoch_rows(rows: list[dict[str, str]]) -> list[dict]:
    merged: dict[int, dict] = {}
    for row in rows:
        epoch = int(row["epoch"])
        entry = merged.setdefault(epoch, {"epoch": epoch})

        if safe_float(row.get("train/loss")) is not None:
            entry.update(
                {
                    "step": int(row["step"]),
                    "train_loss": safe_float(row.get("train/loss")),
                    "train_loss_bbox": safe_float(row.get("train/loss_bbox")),
                    "train_loss_ce": safe_float(row.get("train/loss_ce")),
                    "train_loss_giou": safe_float(row.get("train/loss_giou")),
                    "train_cardinality_error": safe_float(row.get("train/cardinality_error")),
                    "train_class_error": safe_float(row.get("train/class_error")),
                }
            )

        if safe_float(row.get("val/mAP_50_95")) is not None:
            entry.update(
                {
                    "step": int(row["step"]),
                    "val_f1": safe_float(row.get("val/F1")),
                    "val_loss": safe_float(row.get("val/loss")),
                    "val_map50": safe_float(row.get("val/mAP_50")),
                    "val_map50_95": safe_float(row.get("val/mAP_50_95")),
                    "val_map75": safe_float(row.get("val/mAP_75")),
                    "val_mar": safe_float(row.get("val/mAR")),
                    "val_precision": safe_float(row.get("val/precision")),
                    "val_recall": safe_float(row.get("val/recall")),
                }
            )

    return [merged[idx] for idx in sorted(merged)]


def _best_epoch_entry(epoch_rows: list[dict]) -> dict:
    candidates = [row for row in epoch_rows if row.get("val_map50_95") is not None]
    if not candidates:
        raise ValueError("No validation rows found in metrics.csv")
    return max(candidates, key=lambda row: (row.get("val_map50_95") or -math.inf, -(row["epoch"])))


def write_epoch_metrics_csv(path: Path, epoch_rows: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "step",
        "train_loss",
        "train_loss_bbox",
        "train_loss_ce",
        "train_loss_giou",
        "train_cardinality_error",
        "train_class_error",
        "val_loss",
        "val_f1",
        "val_map50",
        "val_map50_95",
        "val_map75",
        "val_mar",
        "val_precision",
        "val_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in epoch_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_key_value_csv(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in payload.items():
            writer.writerow([key, value])


def _write_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _maybe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_fold_curves(fold_dir: Path, epoch_rows: list[dict]) -> None:
    plots_dir = fold_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in epoch_rows]
    plt = _maybe_import_matplotlib()

    def save_plot(filename: str, title: str, series: list[tuple[str, list[float | None]]], ylabel: str) -> None:
        plt.figure(figsize=(8, 5))
        has_any = False
        for label, values in series:
            numeric = [value if value is not None else math.nan for value in values]
            if any(value is not None for value in values):
                has_any = True
                plt.plot(epochs, numeric, marker="o", label=label)
        if not has_any:
            plt.close()
            return
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if len(series) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=150)
        plt.close()

    save_plot(
        "loss_curve.png",
        "Fold Loss Curves",
        [
            ("train_loss", [row.get("train_loss") for row in epoch_rows]),
            ("val_loss", [row.get("val_loss") for row in epoch_rows]),
        ],
        "Loss",
    )
    save_plot(
        "map_curve.png",
        "Fold mAP Curves",
        [
            ("mAP50", [row.get("val_map50") for row in epoch_rows]),
            ("mAP50_95", [row.get("val_map50_95") for row in epoch_rows]),
        ],
        "mAP",
    )
    save_plot(
        "precision_curve.png",
        "Fold Precision / Recall / F1",
        [
            ("precision", [row.get("val_precision") for row in epoch_rows]),
            ("recall", [row.get("val_recall") for row in epoch_rows]),
            ("f1", [row.get("val_f1") for row in epoch_rows]),
        ],
        "Score",
    )


def summarize_fold(fold_dir: Path) -> dict:
    metrics_csv = fold_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found in {fold_dir}")

    rows = load_csv_rows(metrics_csv)
    epoch_rows = _split_epoch_rows(rows)
    best = _best_epoch_entry(epoch_rows)
    training_config = read_training_config(fold_dir)
    class_names = training_config.get("class_names") or []

    summary = {
        "fold": fold_dir.name,
        "best_epoch": best["epoch"],
        "best_step": best.get("step"),
        "train_loss_last": next((row.get("train_loss") for row in reversed(epoch_rows) if row.get("train_loss") is not None), None),
        "val_loss_best_epoch": best.get("val_loss"),
        "map50": best.get("val_map50"),
        "map50_95": best.get("val_map50_95"),
        "map75": best.get("val_map75"),
        "mar": best.get("val_mar"),
        "precision": best.get("val_precision"),
        "recall": best.get("val_recall"),
        "f1": best.get("val_f1"),
        "epochs_logged": len(epoch_rows),
        "checkpoint_best_regular_exists": (fold_dir / "checkpoint_best_regular.pth").exists(),
        "checkpoint_best_total_exists": (fold_dir / "checkpoint_best_total.pth").exists(),
        "checkpoint_best_ema_exists": (fold_dir / "checkpoint_best_ema.pth").exists(),
    }

    write_json(fold_dir / "metrics.json", summary)
    _write_key_value_csv(fold_dir / "metrics_summary.csv", summary)
    write_epoch_metrics_csv(fold_dir / "epoch_metrics.csv", epoch_rows)

    if len(class_names) == 1:
        with (fold_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["class_name", "map50", "map50_95", "map75", "mar", "precision", "recall", "f1"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "class_name": class_names[0],
                    "map50": summary["map50"],
                    "map50_95": summary["map50_95"],
                    "map75": summary["map75"],
                    "mar": summary["mar"],
                    "precision": summary["precision"],
                    "recall": summary["recall"],
                    "f1": summary["f1"],
                }
            )

    _plot_fold_curves(fold_dir, epoch_rows)
    try:
        _generate_visual_samples(fold_dir, training_config)
        visual_samples_created = True
    except Exception as exc:
        visual_samples_created = False
        (fold_dir / "visual_samples_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    report_lines = [
        f"# {fold_dir.name} Report",
        "",
        "## Best Epoch Summary",
        "",
        _write_markdown_table(
            ["Metric", "Value"],
            [[key, value] for key, value in summary.items()],
        ),
        "",
        "## Artifacts",
        "",
        "- `metrics.json`: machine-readable fold summary",
        "- `epoch_metrics.csv`: merged epoch-level train/validation metrics",
        "- `plots/loss_curve.png`: train vs validation loss",
        "- `plots/map_curve.png`: validation mAP curves",
        "- `plots/precision_curve.png`: validation precision/recall/F1 curves",
    ]
    if visual_samples_created:
        report_lines.extend(
            [
                "- `samples/index.md`: visual sample index",
                "- `samples/*.png`: GT vs prediction overlays",
            ]
        )
    else:
        report_lines.append("- Visual samples could not be generated; see `visual_samples_error.txt` if present.")
    (fold_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def _aggregate_metric(values: list[float | None]) -> dict[str, float | None]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(numeric),
        "std": pstdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


def _plot_run_summary(run_dir: Path, fold_summaries: list[dict]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = _maybe_import_matplotlib()

    folds = [item["fold"] for item in fold_summaries]

    def save_bar(filename: str, metric_key: str, title: str) -> None:
        values = [item.get(metric_key) for item in fold_summaries]
        if not any(value is not None for value in values):
            return
        numeric = [value if value is not None else 0.0 for value in values]
        plt.figure(figsize=(8, 5))
        plt.bar(folds, numeric)
        plt.title(title)
        plt.ylabel(metric_key)
        plt.xlabel("Fold")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=150)
        plt.close()

    save_bar("folds_map50.png", "map50", "Fold Comparison: mAP50")
    save_bar("folds_map50_95.png", "map50_95", "Fold Comparison: mAP50-95")
    save_bar("folds_f1.png", "f1", "Fold Comparison: F1")

    plt.figure(figsize=(8, 5))
    precision = [item.get("precision") or 0.0 for item in fold_summaries]
    recall = [item.get("recall") or 0.0 for item in fold_summaries]
    plt.plot(folds, precision, marker="o", label="precision")
    plt.plot(folds, recall, marker="o", label="recall")
    plt.title("Fold Comparison: Precision / Recall")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "folds_precision_recall.png", dpi=150)
    plt.close()


def summarize_run(run_dir: Path, run_type: str) -> None:
    fold_dirs = sorted(path for path in run_dir.glob("fold_*") if path.is_dir())
    fold_summaries = [json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8")) for fold_dir in fold_dirs]
    if not fold_summaries:
        raise ValueError(f"No fold summaries found in {run_dir}")

    fieldnames = [
        "fold",
        "best_epoch",
        "best_step",
        "train_loss_last",
        "val_loss_best_epoch",
        "map50",
        "map50_95",
        "map75",
        "mar",
        "precision",
        "recall",
        "f1",
        "epochs_logged",
    ]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in fold_summaries:
            writer.writerow({key: item.get(key) for key in fieldnames})

    write_json(run_dir / "summary.json", {"folds": fold_summaries})

    aggregate = {
        "map50": _aggregate_metric([item.get("map50") for item in fold_summaries]),
        "map50_95": _aggregate_metric([item.get("map50_95") for item in fold_summaries]),
        "precision": _aggregate_metric([item.get("precision") for item in fold_summaries]),
        "recall": _aggregate_metric([item.get("recall") for item in fold_summaries]),
        "f1": _aggregate_metric([item.get("f1") for item in fold_summaries]),
        "val_loss_best_epoch": _aggregate_metric([item.get("val_loss_best_epoch") for item in fold_summaries]),
    }
    write_json(run_dir / "aggregate_metrics.json", aggregate)

    with (run_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for metric, stats in aggregate.items():
            writer.writerow([metric, stats["mean"], stats["std"], stats["min"], stats["max"]])

    _plot_run_summary(run_dir, fold_summaries)

    summary_rows = [
        [
            item["fold"],
            item.get("best_epoch"),
            item.get("map50"),
            item.get("map50_95"),
            item.get("precision"),
            item.get("recall"),
            item.get("f1"),
        ]
        for item in fold_summaries
    ]
    aggregate_rows = [
        [metric, stats["mean"], stats["std"], stats["min"], stats["max"]]
        for metric, stats in aggregate.items()
    ]
    report_lines = [
        f"# RF-DETR {run_type.capitalize()} Summary",
        "",
        "## Fold Metrics",
        "",
        _write_markdown_table(
            ["Fold", "Best Epoch", "mAP50", "mAP50_95", "Precision", "Recall", "F1"],
            summary_rows,
        ),
        "",
        "## Aggregate Metrics",
        "",
        _write_markdown_table(["Metric", "Mean", "Std", "Min", "Max"], aggregate_rows),
        "",
        "## Artifacts",
        "",
        "- `summary.csv`: one-row-per-fold summary",
        "- `summary.json`: machine-readable fold summaries",
        "- `aggregate_metrics.json`: mean/std/min/max aggregation",
        "- `plots/`: fold comparison charts",
    ]
    (run_dir / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
