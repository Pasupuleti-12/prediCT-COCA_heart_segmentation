from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency at authoring time
    plt = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at authoring time
    torch = None

try:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
except ImportError:  # pragma: no cover - optional dependency at authoring time
    UNet = None
    sliding_window_inference = None

from coca_pipeline import LoaderConfig, WindowConfig, create_dataloaders_from_split


def _require_module(module: Any, package_name: str, feature: str) -> None:
    if module is None:
        raise ImportError(f"{feature} requires '{package_name}'. Install it first and rerun.")


def build_model(model_config: Dict[str, Any]) -> "UNet":
    _require_module(UNet, "monai", "Model loading")
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=tuple(model_config["channels"]),
        strides=tuple(model_config["strides"]),
        num_res_units=int(model_config["num_res_units"]),
        dropout=float(model_config.get("dropout", 0.0)),
    )


def binary_dice(predictions: "torch.Tensor", targets: "torch.Tensor", epsilon: float = 1e-6) -> float:
    predictions = predictions.float()
    targets = targets.float()
    dims = tuple(range(1, predictions.ndim))
    intersection = (predictions * targets).sum(dim=dims)
    denominator = predictions.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return float(dice.mean().item())


def mask_bbox(mask_array: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    coordinates = np.argwhere(mask_array > 0)
    if coordinates.size == 0:
        return None
    return coordinates.min(axis=0), coordinates.max(axis=0)


def bbox_iou(prediction_array: np.ndarray, target_array: np.ndarray) -> float:
    pred_box = mask_bbox(prediction_array)
    target_box = mask_bbox(target_array)
    if pred_box is None or target_box is None:
        return 0.0

    pred_min, pred_max = pred_box
    target_min, target_max = target_box
    intersection_min = np.maximum(pred_min, target_min)
    intersection_max = np.minimum(pred_max, target_max)
    intersection_dims = np.maximum(0, intersection_max - intersection_min + 1)
    pred_dims = pred_max - pred_min + 1
    target_dims = target_max - target_min + 1

    intersection_volume = int(np.prod(intersection_dims))
    pred_volume = int(np.prod(pred_dims))
    target_volume = int(np.prod(target_dims))
    union = pred_volume + target_volume - intersection_volume
    return float(intersection_volume / union) if union else 0.0


def maybe_save_preview(
    image_array: np.ndarray,
    label_array: np.ndarray,
    prediction_array: np.ndarray,
    output_path: str | Path,
) -> None:
    if plt is None:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slice_index = image_array.shape[0] // 2
    image_slice = image_array[slice_index]
    label_slice = label_array[slice_index]
    prediction_slice = prediction_array[slice_index]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("CT")
    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.45)
    axes[1].set_title("TotalSegmentator")
    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(prediction_slice, cmap="Blues", alpha=0.45)
    axes[2].set_title("Prediction")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    _require_module(torch, "torch", "Evaluation")
    _require_module(sliding_window_inference, "monai", "Sliding-window inference")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])

    loader_config_dict = checkpoint.get("loader_config", {})
    window_dict = checkpoint.get("window", {})
    cache_mode = args.cache_mode or ("persistent" if args.cache_dir else "none")
    loader_config = LoaderConfig(
        target_spacing=tuple(float(value) for value in args.target_spacing or loader_config_dict.get("target_spacing", (1.0, 1.0, 1.0))),
        patch_size=tuple(int(value) for value in args.patch_size or loader_config_dict.get("patch_size", (160, 160, 128))),
        batch_size=1,
        num_workers=args.num_workers,
        cache_mode=cache_mode,
        cache_rate=float(loader_config_dict.get("cache_rate", 1.0)),
        cache_dir=args.cache_dir or str(Path(args.output_dir) / "cache"),
        pin_memory=torch.cuda.is_available(),
        train_positive_samples=1,
    )
    window = WindowConfig(
        hu_min=int(window_dict.get("hu_min", -250)),
        hu_max=int(window_dict.get("hu_max", 1000)),
    )

    dataloaders = create_dataloaders_from_split(
        args.split_file,
        loader_config=loader_config,
        sampling_key=None,
        window=window,
    )
    test_loader = dataloaders["test_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    case_rows = []
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()
            case_id_value = batch["case_id"]
            if isinstance(case_id_value, (list, tuple)):
                case_id = str(case_id_value[0])
            else:
                case_id = str(case_id_value)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            logits = sliding_window_inference(
                images,
                roi_size=loader_config.patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - start_time

            predictions = (torch.sigmoid(logits) > 0.5).float()
            dice_score = binary_dice(predictions, labels)

            image_array = images[0, 0].detach().cpu().numpy()
            label_array = labels[0, 0].detach().cpu().numpy()
            prediction_array = predictions[0, 0].detach().cpu().numpy()
            bbox_score = bbox_iou(prediction_array, label_array)

            totalseg_seconds = batch.get("totalseg_seconds")
            if isinstance(totalseg_seconds, torch.Tensor):
                totalseg_seconds = float(totalseg_seconds.item())
            elif isinstance(totalseg_seconds, list) and totalseg_seconds:
                totalseg_seconds = float(totalseg_seconds[0])

            if args.num_visualizations > 0 and index < args.num_visualizations:
                maybe_save_preview(
                    image_array=image_array,
                    label_array=label_array,
                    prediction_array=prediction_array,
                    output_path=figures_dir / f"{case_id}.png",
                )

            row = {
                "case_id": case_id,
                "dice": round(dice_score, 6),
                "bbox_iou": round(bbox_score, 6),
                "inference_seconds": round(inference_seconds, 6),
            }
            if totalseg_seconds is not None:
                row["totalseg_seconds"] = round(float(totalseg_seconds), 6)
                row["speedup_vs_totalseg"] = round(float(totalseg_seconds) / inference_seconds, 4) if inference_seconds else None
            case_rows.append(row)

            print(json.dumps(row))

    metrics_path = output_dir / "case_metrics.csv"
    fieldnames = sorted({key for row in case_rows for key in row.keys()})
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow(row)

    summary = {
        "num_cases": len(case_rows),
        "mean_dice": round(float(np.mean([row["dice"] for row in case_rows])) if case_rows else 0.0, 6),
        "mean_bbox_iou": round(float(np.mean([row["bbox_iou"] for row in case_rows])) if case_rows else 0.0, 6),
        "mean_inference_seconds": round(
            float(np.mean([row["inference_seconds"] for row in case_rows])) if case_rows else 0.0,
            6,
        ),
    }
    totalseg_rows = [row for row in case_rows if "totalseg_seconds" in row]
    if totalseg_rows:
        summary["mean_totalseg_seconds"] = round(float(np.mean([row["totalseg_seconds"] for row in totalseg_rows])), 6)
        summary["mean_speedup_vs_totalseg"] = round(
            float(np.mean([row["speedup_vs_totalseg"] for row in totalseg_rows if row.get("speedup_vs_totalseg") is not None])),
            4,
        )

    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the heart-segmentation model on the COCA test split.")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-mode", choices=("persistent", "memory", "none"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-spacing", nargs=3)
    parser.add_argument("--patch-size", nargs=3)
    parser.add_argument("--num-visualizations", type=int, default=5)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
