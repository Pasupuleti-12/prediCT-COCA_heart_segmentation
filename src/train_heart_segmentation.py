from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

try:
    import torch
    from torch.cuda.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - optional dependency at authoring time
    torch = None
    GradScaler = None
    autocast = None

try:
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet
except ImportError:  # pragma: no cover - optional dependency at authoring time
    DiceCELoss = None
    UNet = None
    sliding_window_inference = None

from coca_pipeline import LoaderConfig, WindowConfig, create_dataloaders_from_split


MODEL_CONFIG = {
    "channels": (16, 32, 64, 128, 256),
    "strides": (2, 2, 2, 2),
    "num_res_units": 2,
    "dropout": 0.1,
}


def _require_module(module: Any, package_name: str, feature: str) -> None:
    if module is None:
        raise ImportError(f"{feature} requires '{package_name}'. Install it first and rerun.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> "UNet":
    _require_module(UNet, "monai", "Model construction")
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=MODEL_CONFIG["channels"],
        strides=MODEL_CONFIG["strides"],
        num_res_units=MODEL_CONFIG["num_res_units"],
        dropout=MODEL_CONFIG["dropout"],
    )


def binary_dice(predictions: "torch.Tensor", targets: "torch.Tensor", epsilon: float = 1e-6) -> float:
    predictions = predictions.float()
    targets = targets.float()
    dims = tuple(range(1, predictions.ndim))
    intersection = (predictions * targets).sum(dim=dims)
    denominator = predictions.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return float(dice.mean().item())


def validate(
    model: "torch.nn.Module",
    loader,
    device: "torch.device",
    roi_size: Sequence[int],
) -> float:
    _require_module(sliding_window_inference, "monai", "Sliding-window validation")

    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()
            logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=1, predictor=model, overlap=0.25)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            scores.append(binary_dice(predictions, labels))
    return float(np.mean(scores)) if scores else 0.0


def save_checkpoint(
    model: "torch.nn.Module",
    output_path: str | Path,
    epoch: int,
    best_val_dice: float,
    loader_config: LoaderConfig,
    window: WindowConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "model_state": model.state_dict(),
        "model_config": MODEL_CONFIG,
        "loader_config": asdict(loader_config),
        "window": asdict(window),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def train(args: argparse.Namespace) -> None:
    _require_module(torch, "torch", "Training")
    _require_module(DiceCELoss, "monai", "Training")

    set_seed(args.seed)
    device = get_device()

    loader_config = LoaderConfig(
        target_spacing=tuple(float(value) for value in args.target_spacing),
        patch_size=tuple(int(value) for value in args.patch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_mode=args.cache_mode,
        cache_rate=args.cache_rate,
        cache_dir=args.cache_dir or str(Path(args.output_dir) / "cache"),
        pin_memory=torch.cuda.is_available(),
        train_positive_samples=args.train_positive_samples,
    )
    window = WindowConfig(hu_min=args.hu_min, hu_max=args.hu_max)

    dataloaders = create_dataloaders_from_split(
        split_path=args.split_file,
        loader_config=loader_config,
        sampling_key=args.sampling_key,
        window=window,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=0.7, lambda_ce=0.3)
    scaler = GradScaler(enabled=args.amp and torch.cuda.is_available())

    best_val_dice = 0.0
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for batch in dataloaders["train_loader"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_summary: Dict[str, Any] = {"epoch": epoch, "train_loss": round(epoch_loss, 6)}

        if epoch % args.val_interval == 0:
            val_dice = validate(model, dataloaders["val_loader"], device, loader_config.patch_size)
            epoch_summary["val_dice"] = round(val_dice, 6)

            if val_dice >= best_val_dice:
                best_val_dice = val_dice
                save_checkpoint(
                    model=model,
                    output_path=output_dir / "best_model.pt",
                    epoch=epoch,
                    best_val_dice=best_val_dice,
                    loader_config=loader_config,
                    window=window,
                )

        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

    save_checkpoint(
        model=model,
        output_path=output_dir / "last_model.pt",
        epoch=args.epochs,
        best_val_dice=best_val_dice,
        loader_config=loader_config,
        window=window,
    )
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "best_val_dice": round(best_val_dice, 6),
        "epochs": args.epochs,
        "device": str(device),
        "train_cases": len(dataloaders["train_records"]),
        "val_cases": len(dataloaders["val_records"]),
        "test_cases": len(dataloaders["test_records"]),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight 3D U-Net for coarse heart segmentation.")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--sampling-key", help="Optional metadata field for weighted sampling.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-mode", choices=("persistent", "memory", "none"), default="memory")
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--target-spacing", nargs=3, default=("1.0", "1.0", "1.0"))
    parser.add_argument("--patch-size", nargs=3, default=("160", "160", "128"))
    parser.add_argument("--train-positive-samples", type=int, default=1)
    parser.add_argument("--hu-min", type=int, default=-250)
    parser.add_argument("--hu-max", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
