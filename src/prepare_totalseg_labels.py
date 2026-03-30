from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency at authoring time
    sitk = None

try:
    from totalsegmentator.python_api import totalsegmentator
except ImportError:  # pragma: no cover - optional dependency at authoring time
    totalsegmentator = None

from coca_pipeline import build_manifest, write_manifest


HEART_STRUCTURE_NAME_GROUPS = (
    ("heart",),
    ("heart_myocardium", "myocardium"),
    ("heart_atrium_left", "atrium_left"),
    ("heart_atrium_right", "atrium_right"),
    ("heart_ventricle_left", "ventricle_left"),
    ("heart_ventricle_right", "ventricle_right"),
)


def _require_module(module: Any, package_name: str, feature: str) -> None:
    if module is None:
        raise ImportError(f"{feature} requires '{package_name}'. Install/configure it and rerun.")


def _mask_bbox(mask_array: np.ndarray) -> Optional[Dict[str, List[int]]]:
    coordinates = np.argwhere(mask_array > 0)
    if coordinates.size == 0:
        return None

    z_min, y_min, x_min = coordinates.min(axis=0).tolist()
    z_max, y_max, x_max = coordinates.max(axis=0).tolist()
    return {
        "zyx_min": [int(z_min), int(y_min), int(x_min)],
        "zyx_max": [int(z_max), int(y_max), int(x_max)],
    }


def combine_heart_structures(
    structure_dir: str | Path,
    output_path: str | Path,
    structure_names: Sequence[Sequence[str]] = HEART_STRUCTURE_NAME_GROUPS,
) -> Dict[str, Any]:
    _require_module(sitk, "SimpleITK", "Heart-mask combination")

    structure_dir = Path(structure_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_mask = None
    used_structures: List[str] = []

    for structure_group in structure_names:
        structure_path = None
        chosen_name = None
        for structure_name in structure_group:
            candidate = structure_dir / f"{structure_name}.nii.gz"
            if candidate.exists():
                structure_path = candidate
                chosen_name = structure_name
                break
        if structure_path is None or chosen_name is None:
            continue
        structure_image = sitk.ReadImage(str(structure_path))
        binary_image = sitk.Cast(structure_image > 0, sitk.sitkUInt8)
        combined_mask = binary_image if combined_mask is None else sitk.Or(combined_mask, binary_image)
        used_structures.append(chosen_name)

    if combined_mask is None:
        raise FileNotFoundError(
            f"No expected heart structures were found in {structure_dir}. "
            f"Expected one of: {', '.join(name for group in structure_names for name in group)}"
        )

    sitk.WriteImage(combined_mask, str(output_path))
    mask_array = sitk.GetArrayFromImage(combined_mask)

    return {
        "label": str(output_path.resolve()),
        "structures_used": used_structures,
        "foreground_voxels": int(mask_array.sum()),
        "heart_bbox": _mask_bbox(mask_array),
    }


def summarise_label(label_path: str | Path) -> Dict[str, Any]:
    _require_module(sitk, "SimpleITK", "Label summarisation")
    label_image = sitk.ReadImage(str(label_path))
    mask_array = sitk.GetArrayFromImage(label_image)
    return {
        "label": str(Path(label_path).resolve()),
        "structures_used": ["precomputed_heart_mask"],
        "foreground_voxels": int(mask_array.sum()),
        "heart_bbox": _mask_bbox(mask_array),
    }


def run_totalsegmentator(
    image_path: str | Path,
    structure_output_dir: str | Path,
    task: str = "total",
    fast: bool = False,
) -> float:
    _require_module(totalsegmentator, "totalsegmentator", "TotalSegmentator execution")

    start_time = time.perf_counter()
    totalsegmentator(
        input=str(image_path),
        output=str(structure_output_dir),
        task=task,
        fast=fast,
    )
    return time.perf_counter() - start_time


def process_cases(
    images_dir: str | Path,
    output_dir: str | Path,
    metadata_csv: str | Path | None = None,
    max_cases: Optional[int] = None,
    task: str = "total",
    fast: bool = False,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    _require_module(sitk, "SimpleITK", "Heart-mask preparation")

    records = build_manifest(images_dir=images_dir, metadata_csv=metadata_csv)
    selected_records = records[:max_cases] if max_cases else records

    output_dir = Path(output_dir)
    structure_root = output_dir / "structures"
    label_root = output_dir / "masks"
    structure_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    prepared_records: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []

    for record in selected_records:
        case_id = str(record["case_id"])
        structure_output_dir = structure_root / case_id
        label_output_path = label_root / f"{case_id}_heart_mask.nii.gz"

        if not overwrite and label_output_path.exists():
            if structure_output_dir.exists():
                combined = combine_heart_structures(structure_output_dir, label_output_path)
            else:
                combined = summarise_label(label_output_path)
            updated = dict(record)
            updated.update(combined)
            prepared_records.append(updated)
            continue

        if (
            not overwrite
            and structure_output_dir.exists()
            and any(structure_output_dir.glob("*.nii.gz"))
            and not label_output_path.exists()
        ):
            combined = combine_heart_structures(structure_output_dir, label_output_path)
            updated = dict(record)
            updated.update(combined)
            prepared_records.append(updated)
            continue

        structure_output_dir.mkdir(parents=True, exist_ok=True)
        runtime_seconds = run_totalsegmentator(record["image"], structure_output_dir, task=task, fast=fast)
        combined = combine_heart_structures(structure_output_dir, label_output_path)

        updated = dict(record)
        updated.update(combined)
        updated["totalseg_seconds"] = round(runtime_seconds, 4)
        updated["totalseg_task"] = task
        prepared_records.append(updated)
        runtime_rows.append(
            {
                "case_id": case_id,
                "image": record["image"],
                "label": combined["label"],
                "totalseg_seconds": round(runtime_seconds, 4),
                "totalseg_task": task,
            }
        )

    runtime_path = output_dir / "totalseg_runtime.csv"
    with runtime_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "image", "label", "totalseg_seconds", "totalseg_task"],
        )
        writer.writeheader()
        for row in runtime_rows:
            writer.writerow(row)

    summary = {
        "num_cases": len(prepared_records),
        "mean_totalseg_seconds": round(
            float(np.mean([row["totalseg_seconds"] for row in runtime_rows])) if runtime_rows else 0.0,
            4,
        ),
        "task": task,
        "fast_mode": fast,
        "images_dir": str(Path(images_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    with (output_dir / "totalseg_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return prepared_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator on COCA volumes and extract a coarse whole-heart mask."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing resampled CT volumes.")
    parser.add_argument("--output-dir", required=True, help="Directory where heart masks and runtimes are written.")
    parser.add_argument("--metadata-csv", help="Optional metadata CSV to merge into the output manifest.")
    parser.add_argument("--manifest-output", help="Optional CSV/JSON manifest path for the labeled dataset.")
    parser.add_argument("--license-number", help="Optional TotalSegmentator license number.")
    parser.add_argument("--max-cases", type=int, help="Limit processing to the first N cases for the assignment.")
    parser.add_argument(
        "--task",
        default="total",
        help="TotalSegmentator task name. Use 'total' for the coarse whole-heart mask or a licensed task such as "
        "'heartchambers_highres' if desired.",
    )
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode if available.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run cases even if masks already exist.")
    args = parser.parse_args()

    if args.license_number:
        os.environ["TOTALSEG_LICENSE_NUMBER"] = args.license_number

    prepared_records = process_cases(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        max_cases=args.max_cases,
        task=args.task,
        fast=args.fast,
        overwrite=args.overwrite,
    )

    if args.manifest_output:
        manifest_path = write_manifest(prepared_records, args.manifest_output)
        print(f"Wrote labeled manifest for {len(prepared_records)} cases to {manifest_path}")
    else:
        print(f"Prepared heart masks for {len(prepared_records)} cases under {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
