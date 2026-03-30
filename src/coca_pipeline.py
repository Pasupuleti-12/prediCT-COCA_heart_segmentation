from __future__ import annotations

import argparse
import csv
import json
import plistlib
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from matplotlib.path import Path as MplPath
except ImportError:  # pragma: no cover - optional dependency at authoring time
    MplPath = None

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency at authoring time
    sitk = None

try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except ImportError:  # pragma: no cover - optional dependency at authoring time
    torch = None
    WeightedRandomSampler = None

try:
    from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        RandAffined,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandScaleIntensityd,
        RandShiftIntensityd,
        ScaleIntensityRanged,
        SpatialPadd,
        Spacingd,
    )
except ImportError:  # pragma: no cover - optional dependency at authoring time
    CacheDataset = None
    Compose = None
    CropForegroundd = None
    DataLoader = None
    Dataset = None
    EnsureChannelFirstd = None
    EnsureTyped = None
    LoadImaged = None
    Orientationd = None
    PersistentDataset = None
    RandAffined = None
    RandCropByPosNegLabeld = None
    RandFlipd = None
    RandGaussianNoised = None
    RandScaleIntensityd = None
    RandShiftIntensityd = None
    ScaleIntensityRanged = None
    SpatialPadd = None
    Spacingd = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - optional dependency at authoring time
    train_test_split = None


DEFAULT_IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")


@dataclass(frozen=True)
class WindowConfig:
    hu_min: int = -250
    hu_max: int = 1000


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    val_size: float = 0.1
    test_size: float = 0.2
    stratify_key: str = "split_label"


@dataclass(frozen=True)
class LoaderConfig:
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    patch_size: Tuple[int, int, int] = (160, 160, 128)
    batch_size: int = 2
    num_workers: int = 4
    cache_mode: str = "persistent"
    cache_rate: float = 1.0
    cache_dir: Optional[str] = None
    pin_memory: bool = True
    train_positive_samples: int = 1
    image_key: str = "image"
    label_key: str = "label"


def _require_module(module: Any, package_name: str, feature: str) -> None:
    if module is None:
        raise ImportError(
            f"{feature} requires '{package_name}'. Install it first and rerun the command."
        )


def _normalise_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if trimmed == "":
        return ""
    lowered = trimmed.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in trimmed:
            return float(trimmed)
        return int(trimmed)
    except ValueError:
        return trimmed


def _is_medical_image(path: Path) -> bool:
    return any(str(path).lower().endswith(ext) for ext in DEFAULT_IMAGE_EXTENSIONS)


def _strip_known_suffixes(case_id: str, suffixes: Sequence[str]) -> str:
    normalized = case_id
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def infer_case_id(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def hu_windowing(
    volume: np.ndarray,
    hu_min: int = WindowConfig.hu_min,
    hu_max: int = WindowConfig.hu_max,
) -> np.ndarray:
    clipped = np.clip(volume, hu_min, hu_max)
    scaled = (clipped - hu_min) / float(hu_max - hu_min)
    return scaled.astype(np.float32)


def load_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [{key: _coerce_scalar(value) for key, value in row.items()} for row in reader]
    if manifest_path.suffix.lower() in {".json", ".jsonl"}:
        with manifest_path.open("r", encoding="utf-8") as handle:
            if manifest_path.suffix.lower() == ".jsonl":
                return [json.loads(line) for line in handle if line.strip()]
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "records" in payload:
            return payload["records"]
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def write_manifest(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        fieldnames: List[str] = []
        for record in records:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
        return output_path

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2)
    return output_path


def build_manifest(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    image_suffixes: Sequence[str] = ("_img", "_image"),
    label_suffixes: Sequence[str] = ("_heart_mask", "_mask", "_seg", "_label"),
) -> List[Dict[str, Any]]:
    images_dir = Path(images_dir)
    image_files = sorted(path for path in images_dir.rglob("*") if path.is_file() and _is_medical_image(path))
    if not image_files:
        raise FileNotFoundError(f"No medical image files were found under {images_dir}")

    metadata_by_case: Dict[str, Dict[str, Any]] = {}
    if metadata_csv is not None:
        with Path(metadata_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_id = str(row.get("case_id") or row.get("patient_id") or row.get("study_id") or "").strip()
                if not case_id:
                    continue
                metadata_by_case[case_id] = {key: _coerce_scalar(value) for key, value in row.items()}

    labels_by_case: Dict[str, Path] = {}
    if labels_dir is not None:
        labels_dir = Path(labels_dir)
        for path in labels_dir.rglob("*"):
            if not path.is_file() or not _is_medical_image(path):
                continue
            label_case = _strip_known_suffixes(infer_case_id(path), label_suffixes)
            labels_by_case[label_case] = path

    records: List[Dict[str, Any]] = []
    for image_path in image_files:
        case_id = _strip_known_suffixes(infer_case_id(image_path), image_suffixes)
        record: Dict[str, Any] = {
            "case_id": case_id,
            "image": _normalise_path(image_path),
        }
        label_path = labels_by_case.get(case_id)
        if label_path is not None:
            record["label"] = _normalise_path(label_path)
        metadata = metadata_by_case.get(case_id)
        if metadata:
            record.update(metadata)
        records.append(record)

    return records


def _directory_sort_key(path: Path) -> Tuple[int, str]:
    return (0, f"{int(path.name):08d}") if path.name.isdigit() else (1, path.name)


def _count_dicom_files(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.dcm"))


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


def resolve_coca_gated_roots(
    dataset_root: str | Path | None = None,
    patient_root: str | Path | None = None,
    xml_root: str | Path | None = None,
) -> Tuple[Path, Path]:
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        patient_root = dataset_root / "patient"
        xml_root = dataset_root / "calcium_xml"

    if patient_root is None or xml_root is None:
        raise ValueError("Provide either --dataset-root or both --patient-root and --xml-root.")

    patient_root = Path(patient_root)
    xml_root = Path(xml_root)
    if not patient_root.exists():
        raise FileNotFoundError(f"COCA patient root does not exist: {patient_root}")
    if not xml_root.exists():
        raise FileNotFoundError(f"COCA XML root does not exist: {xml_root}")
    return patient_root, xml_root


def discover_coca_gated_series(patient_root: str | Path) -> List[Dict[str, Any]]:
    patient_root = Path(patient_root)
    if not patient_root.exists():
        raise FileNotFoundError(f"COCA patient root does not exist: {patient_root}")

    discovered: List[Dict[str, Any]] = []
    patient_dirs = sorted((path for path in patient_root.iterdir() if path.is_dir()), key=_directory_sort_key)

    for patient_dir in patient_dirs:
        candidate_dirs: Dict[Path, int] = {}

        direct_count = _count_dicom_files(patient_dir)
        if direct_count >= 5:
            candidate_dirs[patient_dir] = direct_count

        for child_dir in patient_dir.iterdir():
            if not child_dir.is_dir():
                continue
            child_count = _count_dicom_files(child_dir)
            if child_count >= 5:
                candidate_dirs[child_dir] = child_count

        if not candidate_dirs:
            for nested_dir in patient_dir.rglob("*"):
                if not nested_dir.is_dir():
                    continue
                nested_count = _count_dicom_files(nested_dir)
                if nested_count >= 5:
                    candidate_dirs[nested_dir] = nested_count

        if not candidate_dirs:
            warnings.warn(
                f"No valid DICOM series directory was found under {patient_dir}. Skipping this patient.",
                stacklevel=2,
            )
            continue

        series_dir, dicom_count = max(
            candidate_dirs.items(),
            key=lambda item: (item[1], -len(item[0].parts)),
        )
        discovered.append(
            {
                "patient_id": patient_dir.name,
                "series_dir": series_dir,
                "series_name": series_dir.name,
                "nested_series": series_dir != patient_dir,
                "num_candidate_series": len(candidate_dirs),
                "num_dicom_files": dicom_count,
            }
        )

    return discovered


def _select_gdcm_series(series_dir: Path) -> Tuple[List[str], Optional[str]]:
    _require_module(sitk, "SimpleITK", "DICOM loading")

    series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_dir)) or [])
    if series_ids:
        series_candidates = [
            (
                series_id,
                list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir), series_id)),
            )
            for series_id in series_ids
        ]
        selected_series_id, selected_files = max(series_candidates, key=lambda item: len(item[1]))
        return selected_files, selected_series_id

    direct_files = sorted(str(path) for path in series_dir.glob("*.dcm"))
    return direct_files, None


def load_coca_dicom_series(series_dir: str | Path) -> Tuple["sitk.Image", Dict[str, Any]]:
    _require_module(sitk, "SimpleITK", "DICOM loading")

    series_dir = Path(series_dir)
    dicom_files, series_id = _select_gdcm_series(series_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files were found in {series_dir}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    details = {
        "gdcm_series_id": series_id or "",
        "num_dicom_files": len(dicom_files),
    }
    return image, details


def parse_coca_calcium_xml(
    xml_path: str | Path,
    image_shape: Sequence[int],
) -> Tuple[np.ndarray, List[int]]:
    _require_module(MplPath, "matplotlib", "COCA XML parsing")

    xml_path = Path(xml_path)
    mask = np.zeros(tuple(int(value) for value in image_shape), dtype=np.uint8)
    segmented_slices: set[int] = set()

    if not xml_path.exists():
        return mask, []

    total_z, total_y, total_x = mask.shape
    with xml_path.open("rb") as handle:
        payload = plistlib.load(handle)

    for image_entry in payload.get("Images", []):
        try:
            z_index = int(image_entry.get("ImageIndex", -1))
        except (TypeError, ValueError):
            continue

        if z_index < 0 or z_index >= total_z:
            continue

        for roi in image_entry.get("ROIs", []):
            polygon_points: List[List[int]] = []
            for point_text in roi.get("Point_px", []):
                cleaned = str(point_text).replace("(", "").replace(")", "")
                parts = [part.strip() for part in cleaned.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    x_coord = int(round(float(parts[0])))
                    y_coord = int(round(float(parts[1])))
                except ValueError:
                    continue
                polygon_points.append([x_coord, y_coord])

            if not polygon_points:
                continue

            slice_mask = np.zeros((total_y, total_x), dtype=np.uint8)
            points_array = np.asarray(polygon_points, dtype=np.float32)
            if len(points_array) >= 3:
                min_x = max(int(np.floor(points_array[:, 0].min())), 0)
                max_x = min(int(np.ceil(points_array[:, 0].max())), total_x - 1)
                min_y = max(int(np.floor(points_array[:, 1].min())), 0)
                max_y = min(int(np.ceil(points_array[:, 1].max())), total_y - 1)

                if min_x <= max_x and min_y <= max_y:
                    grid_x, grid_y = np.meshgrid(
                        np.arange(min_x, max_x + 1),
                        np.arange(min_y, max_y + 1),
                    )
                    sample_points = np.column_stack((grid_x.ravel() + 0.5, grid_y.ravel() + 0.5))
                    polygon = MplPath(points_array)
                    inside = polygon.contains_points(sample_points, radius=1e-6).reshape(grid_x.shape)
                    slice_mask[min_y : max_y + 1, min_x : max_x + 1] = inside.astype(np.uint8)
            else:
                for point_x, point_y in points_array:
                    if 0 <= point_x < total_x and 0 <= point_y < total_y:
                        slice_mask[int(point_y), int(point_x)] = 1

            if np.any(slice_mask):
                mask[z_index] = np.maximum(mask[z_index], slice_mask)
                segmented_slices.add(z_index)

    return mask, sorted(segmented_slices)


def import_coca_gated_dataset(
    output_root: str | Path,
    dataset_root: str | Path | None = None,
    patient_root: str | Path | None = None,
    xml_root: str | Path | None = None,
    max_cases: Optional[int] = None,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    _require_module(sitk, "SimpleITK", "COCA gated import")
    _require_module(MplPath, "matplotlib", "COCA gated import")

    patient_root, xml_root = resolve_coca_gated_roots(
        dataset_root=dataset_root,
        patient_root=patient_root,
        xml_root=xml_root,
    )

    output_root = Path(output_root)
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    metadata_dir = output_root / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    discovered_series = discover_coca_gated_series(patient_root)
    selected_series = discovered_series[:max_cases] if max_cases else discovered_series
    imported_records: List[Dict[str, Any]] = []

    for series_info in selected_series:
        case_id = str(series_info["patient_id"])
        image_output = images_dir / f"{case_id}_img.nii.gz"
        label_output = labels_dir / f"{case_id}_seg.nii.gz"
        metadata_output = metadata_dir / f"{case_id}_meta.json"

        if image_output.exists() and label_output.exists() and metadata_output.exists() and not overwrite:
            with metadata_output.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            imported_records.append(payload["manifest_record"])
            continue

        image, dicom_details = load_coca_dicom_series(series_info["series_dir"])
        image_array = sitk.GetArrayFromImage(image)

        xml_path = xml_root / f"{case_id}.xml"
        mask_array, calcium_slices = parse_coca_calcium_xml(xml_path, image_array.shape)
        mask_image = sitk.GetImageFromArray(mask_array.astype(np.uint8))
        mask_image.CopyInformation(image)

        sitk.WriteImage(image, str(image_output), useCompression=True)
        sitk.WriteImage(mask_image, str(label_output), useCompression=True)

        calcium_voxels = int(mask_array.sum())
        bbox = _mask_bbox(mask_array)
        manifest_record: Dict[str, Any] = {
            "case_id": case_id,
            "patient_id": case_id,
            "image": _normalise_path(image_output),
            "label": _normalise_path(label_output),
            "source_dataset": "COCA",
            "subset": "gated",
            "series_dir": _normalise_path(series_info["series_dir"]),
            "series_name": str(series_info["series_name"]),
            "nested_series": bool(series_info["nested_series"]),
            "num_candidate_series": int(series_info["num_candidate_series"]),
            "num_dicom_files": int(dicom_details["num_dicom_files"]),
            "xml_present": xml_path.exists(),
            "xml_path": _normalise_path(xml_path) if xml_path.exists() else "",
            "spacing_x_mm": round(float(image.GetSpacing()[0]), 5),
            "spacing_y_mm": round(float(image.GetSpacing()[1]), 5),
            "spacing_z_mm": round(float(image.GetSpacing()[2]), 5),
            "size_x": int(image.GetSize()[0]),
            "size_y": int(image.GetSize()[1]),
            "size_z": int(image.GetSize()[2]),
            "calcium_voxels": calcium_voxels,
            "num_calcium_slices": len(calcium_slices),
            "has_calcium": calcium_voxels > 0,
            "split_label": "calcium_present" if calcium_voxels > 0 else "calcium_absent",
        }
        imported_records.append(manifest_record)

        rich_metadata = dict(manifest_record)
        rich_metadata["gdcm_series_id"] = str(dicom_details["gdcm_series_id"])
        rich_metadata["array_shape_zyx"] = [int(value) for value in image_array.shape]
        rich_metadata["calcium_slices"] = calcium_slices
        rich_metadata["calcium_bbox"] = bbox
        with metadata_output.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "manifest_record": manifest_record,
                    "details": rich_metadata,
                },
                handle,
                indent=2,
            )

    summary = {
        "num_discovered_cases": len(discovered_series),
        "num_imported_cases": len(imported_records),
        "num_positive_cases": int(sum(1 for record in imported_records if record["has_calcium"])),
        "num_missing_xml": int(sum(1 for record in imported_records if not record["xml_present"])),
        "patient_root": _normalise_path(patient_root),
        "xml_root": _normalise_path(xml_root),
        "output_root": _normalise_path(output_root),
    }
    with (output_root / "import_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return imported_records


def _resample_size(
    current_size: Sequence[int],
    current_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> List[int]:
    return [
        max(1, int(round(size * spacing / target)))
        for size, spacing, target in zip(current_size, current_spacing, target_spacing)
    ]


def resample_image(
    image: "sitk.Image",
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    is_label: bool = False,
) -> "sitk.Image":
    _require_module(sitk, "SimpleITK", "Image resampling")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(target_spacing))
    resampler.SetSize(_resample_size(image.GetSize(), image.GetSpacing(), target_spacing))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0 if is_label else -1024)
    return resampler.Execute(image)


def resample_case(
    image_path: str | Path,
    image_output_path: str | Path,
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    label_path: str | Path | None = None,
    label_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    _require_module(sitk, "SimpleITK", "Image resampling")

    image = sitk.ReadImage(str(image_path))
    resampled_image = resample_image(image, target_spacing=target_spacing, is_label=False)
    image_output_path = Path(image_output_path)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled_image, str(image_output_path))

    result = {
        "image": _normalise_path(image_output_path),
        "original_spacing": tuple(float(value) for value in image.GetSpacing()),
        "resampled_spacing": tuple(float(value) for value in resampled_image.GetSpacing()),
        "original_size": tuple(int(value) for value in image.GetSize()),
        "resampled_size": tuple(int(value) for value in resampled_image.GetSize()),
    }

    if label_path and label_output_path:
        label = sitk.ReadImage(str(label_path))
        resampled_label = resample_image(label, target_spacing=target_spacing, is_label=True)
        label_output_path = Path(label_output_path)
        label_output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled_label, str(label_output_path))
        result["label"] = _normalise_path(label_output_path)

    return result


def resample_manifest_records(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    output_root = Path(output_root)
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    resampled_records: List[Dict[str, Any]] = []

    for record in records:
        case_id = str(record["case_id"])
        image_name = Path(str(record["image"])).name
        image_output = output_images / image_name
        label_output = output_labels / Path(str(record["label"])).name if record.get("label") else None

        if image_output.exists() and (label_output is None or label_output.exists()) and not overwrite:
            updated = dict(record)
            updated["image"] = _normalise_path(image_output)
            if label_output is not None:
                updated["label"] = _normalise_path(label_output)
            resampled_records.append(updated)
            continue

        result = resample_case(
            image_path=record["image"],
            image_output_path=image_output,
            label_path=record.get("label"),
            label_output_path=label_output,
            target_spacing=target_spacing,
        )
        updated = dict(record)
        updated.update(result)
        resampled_records.append(updated)

    return resampled_records


def random_split(
    records: Sequence[Mapping[str, Any]],
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    shuffled = [dict(record) for record in records]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = int(round(total * test_size))
    val_count = int(round(total * val_size))
    test = shuffled[:test_count]
    val = shuffled[test_count : test_count + val_count]
    train = shuffled[test_count + val_count :]
    if not train:
        raise ValueError("Random split produced an empty train set; reduce val/test size.")
    return train, val, test


def stratified_split(
    records: Sequence[Mapping[str, Any]],
    label_key: str = "split_label",
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        raise ValueError("Cannot split an empty record list.")
    if label_key not in records[0]:
        warnings.warn(
            f"'{label_key}' was not found in the manifest. Falling back to a random split.",
            stacklevel=2,
        )
        return random_split(records, val_size=val_size, test_size=test_size, seed=seed)

    labels = [record.get(label_key) for record in records]
    label_counts = Counter(labels)
    if any(count < 2 for count in label_counts.values()):
        warnings.warn(
            f"At least one '{label_key}' class has fewer than 2 samples. Falling back to a random split.",
            stacklevel=2,
        )
        return random_split(records, val_size=val_size, test_size=test_size, seed=seed)

    if train_test_split is None:
        warnings.warn(
            "scikit-learn is not installed, so the pipeline is falling back to a random split.",
            stacklevel=2,
        )
        return random_split(records, val_size=val_size, test_size=test_size, seed=seed)

    train_val, test = train_test_split(
        [dict(record) for record in records],
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = [record.get(label_key) for record in train_val]
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )
    return train, val, test


def save_split_file(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    split_config: SplitConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(split_config),
        "train": list(train_records),
        "val": list(val_records),
        "test": list(test_records),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def load_split_file(split_path: str | Path) -> Dict[str, Any]:
    with Path(split_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not all(key in payload for key in ("train", "val", "test")):
        raise ValueError("Split file must contain 'train', 'val', and 'test' keys.")
    return payload


def build_weighted_sampler(
    records: Sequence[Mapping[str, Any]],
    sampling_key: str,
) -> Optional["WeightedRandomSampler"]:
    _require_module(torch, "torch", "Weighted sampling")
    _require_module(WeightedRandomSampler, "torch", "Weighted sampling")

    if not records:
        return None
    if sampling_key not in records[0]:
        return None

    labels = [str(record.get(sampling_key)) for record in records]
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(weight_tensor, num_samples=len(sample_weights), replacement=True)


def get_segmentation_transforms(
    mode: str,
    loader_config: LoaderConfig,
    window: WindowConfig = WindowConfig(),
    include_label: bool = True,
):
    _require_module(Compose, "monai", "MONAI transforms")

    keys = [loader_config.image_key]
    if include_label:
        keys.append(loader_config.label_key)

    if include_label:
        spacing_mode: Sequence[str] = ("bilinear", "nearest")
    else:
        spacing_mode = ("bilinear",)

    base = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=loader_config.target_spacing, mode=spacing_mode),
        ScaleIntensityRanged(
            keys=[loader_config.image_key],
            a_min=float(window.hu_min),
            a_max=float(window.hu_max),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=keys, source_key=loader_config.image_key),
    ]

    if mode == "train":
        base.append(SpatialPadd(keys=keys, spatial_size=loader_config.patch_size))
        if include_label:
            base.append(
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key=loader_config.label_key,
                    spatial_size=loader_config.patch_size,
                    pos=2,
                    neg=1,
                    num_samples=loader_config.train_positive_samples,
                    image_key=loader_config.image_key,
                    image_threshold=0.0,
                    allow_smaller=True,
                )
            )
        base.extend(
            [
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
                RandAffined(
                    keys=keys,
                    prob=0.2,
                    rotate_range=(0.08, 0.08, 0.08),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=spacing_mode,
                    padding_mode="zeros",
                ),
                RandGaussianNoised(keys=[loader_config.image_key], prob=0.15, mean=0.0, std=0.01),
                RandScaleIntensityd(keys=[loader_config.image_key], prob=0.15, factors=0.1),
                RandShiftIntensityd(keys=[loader_config.image_key], prob=0.15, offsets=0.1),
            ]
        )

    base.append(EnsureTyped(keys=keys))
    return Compose(base)


def build_dataset(
    records: Sequence[Mapping[str, Any]],
    transform,
    loader_config: LoaderConfig,
):
    _require_module(Dataset, "monai", "Dataset creation")

    data = [dict(record) for record in records]
    cache_mode = loader_config.cache_mode.lower()
    if cache_mode == "persistent":
        if not loader_config.cache_dir:
            raise ValueError("cache_dir must be provided when cache_mode='persistent'.")
        cache_dir = Path(loader_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=data, transform=transform, cache_dir=str(cache_dir))
    if cache_mode == "memory":
        return CacheDataset(
            data=data,
            transform=transform,
            cache_rate=loader_config.cache_rate,
            num_workers=loader_config.num_workers,
        )
    return Dataset(data=data, transform=transform)


def build_dataloader(
    dataset,
    loader_config: LoaderConfig,
    shuffle: bool,
    sampler: Optional["WeightedRandomSampler"] = None,
):
    _require_module(DataLoader, "monai", "DataLoader creation")

    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
    )


def create_dataloaders_from_split(
    split_path: str | Path,
    loader_config: LoaderConfig,
    sampling_key: Optional[str] = None,
    window: WindowConfig = WindowConfig(),
) -> Dict[str, Any]:
    split_payload = load_split_file(split_path)
    train_records = split_payload["train"]
    val_records = split_payload["val"]
    test_records = split_payload["test"]

    cache_dir = Path(loader_config.cache_dir) if loader_config.cache_dir else None

    train_loader_config = loader_config
    val_loader_config = loader_config
    test_loader_config = loader_config

    if cache_dir is not None:
        train_loader_config = LoaderConfig(**{**asdict(loader_config), "cache_dir": str(cache_dir / "train")})
        val_loader_config = LoaderConfig(**{**asdict(loader_config), "cache_dir": str(cache_dir / "val")})
        test_loader_config = LoaderConfig(**{**asdict(loader_config), "cache_dir": str(cache_dir / "test")})

    train_transform = get_segmentation_transforms("train", train_loader_config, window=window, include_label=True)
    val_transform = get_segmentation_transforms("val", val_loader_config, window=window, include_label=True)
    test_transform = get_segmentation_transforms("test", test_loader_config, window=window, include_label=True)

    train_dataset = build_dataset(train_records, train_transform, train_loader_config)
    val_dataset = build_dataset(val_records, val_transform, val_loader_config)
    test_dataset = build_dataset(test_records, test_transform, test_loader_config)

    sampler = build_weighted_sampler(train_records, sampling_key) if sampling_key else None
    train_loader = build_dataloader(train_dataset, train_loader_config, shuffle=True, sampler=sampler)
    val_loader = build_dataloader(val_dataset, val_loader_config, shuffle=False)
    test_loader = build_dataloader(test_dataset, test_loader_config, shuffle=False)

    return {
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def compute_dataset_statistics(
    records: Sequence[Mapping[str, Any]],
    stratify_key: Optional[str] = None,
) -> Dict[str, Any]:
    _require_module(sitk, "SimpleITK", "Dataset statistics")

    spacings: List[Tuple[float, float, float]] = []
    sizes: List[Tuple[int, int, int]] = []
    physical_sizes: List[Tuple[float, float, float]] = []
    foreground_fractions: List[float] = []
    foreground_volumes_ml: List[float] = []
    missing_labels = 0

    for record in records:
        image = sitk.ReadImage(str(record["image"]))
        spacing = tuple(float(value) for value in image.GetSpacing())
        size = tuple(int(value) for value in image.GetSize())
        physical_size = tuple(round(size_axis * spacing_axis, 3) for size_axis, spacing_axis in zip(size, spacing))

        spacings.append(spacing)
        sizes.append(size)
        physical_sizes.append(physical_size)

        label_path = record.get("label")
        if not label_path:
            missing_labels += 1
            continue

        label_image = sitk.ReadImage(str(label_path))
        label_array = sitk.GetArrayFromImage(label_image) > 0
        foreground_voxels = int(label_array.sum())
        total_voxels = int(label_array.size)
        foreground_fractions.append(foreground_voxels / total_voxels if total_voxels else 0.0)
        voxel_volume_ml = np.prod(label_image.GetSpacing()) / 1000.0
        foreground_volumes_ml.append(round(foreground_voxels * voxel_volume_ml, 4))

    spacing_array = np.asarray(spacings, dtype=np.float32)
    size_array = np.asarray(sizes, dtype=np.float32)
    physical_array = np.asarray(physical_sizes, dtype=np.float32)

    statistics: Dict[str, Any] = {
        "num_scans": len(records),
        "num_labeled_scans": len(records) - missing_labels,
        "missing_labels": missing_labels,
        "spacing_mm": {
            "mean": spacing_array.mean(axis=0).round(4).tolist(),
            "min": spacing_array.min(axis=0).round(4).tolist(),
            "max": spacing_array.max(axis=0).round(4).tolist(),
        },
        "image_size_voxels": {
            "mean": size_array.mean(axis=0).round(2).tolist(),
            "min": size_array.min(axis=0).astype(int).tolist(),
            "max": size_array.max(axis=0).astype(int).tolist(),
        },
        "physical_size_mm": {
            "mean": physical_array.mean(axis=0).round(2).tolist(),
            "min": physical_array.min(axis=0).round(2).tolist(),
            "max": physical_array.max(axis=0).round(2).tolist(),
        },
    }

    if foreground_fractions:
        statistics["label_foreground_fraction"] = {
            "mean": round(float(np.mean(foreground_fractions)), 6),
            "min": round(float(np.min(foreground_fractions)), 6),
            "max": round(float(np.max(foreground_fractions)), 6),
        }
        statistics["label_volume_ml"] = {
            "mean": round(float(np.mean(foreground_volumes_ml)), 3),
            "min": round(float(np.min(foreground_volumes_ml)), 3),
            "max": round(float(np.max(foreground_volumes_ml)), 3),
        }

    if stratify_key:
        statistics["class_distribution"] = dict(Counter(str(record.get(stratify_key, "missing")) for record in records))

    return statistics


def save_dataset_statistics(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    stratify_key: Optional[str] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    statistics = compute_dataset_statistics(records, stratify_key=stratify_key)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2)
    return output_path


def _parse_spacing(values: Sequence[str]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("target spacing must contain exactly 3 values.")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _manifest_cli(args: argparse.Namespace) -> None:
    records = build_manifest(args.images_dir, labels_dir=args.labels_dir, metadata_csv=args.metadata_csv)
    output_path = write_manifest(records, args.output)
    print(f"Wrote {len(records)} manifest records to {output_path}")


def _import_coca_gated_cli(args: argparse.Namespace) -> None:
    records = import_coca_gated_dataset(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        patient_root=args.patient_root,
        xml_root=args.xml_root,
        max_cases=args.max_cases,
        overwrite=args.overwrite,
    )
    manifest_path = write_manifest(records, args.output_manifest)
    print(f"Imported {len(records)} COCA gated cases and wrote the manifest to {manifest_path}")


def _resample_cli(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest)
    target_spacing = _parse_spacing(args.target_spacing)
    resampled = resample_manifest_records(
        records=records,
        output_root=args.output_root,
        target_spacing=target_spacing,
        overwrite=args.overwrite,
    )
    output_path = write_manifest(resampled, args.output_manifest)
    print(f"Resampled {len(resampled)} scans and wrote the updated manifest to {output_path}")


def _split_cli(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest)
    split_config = SplitConfig(
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        stratify_key=args.stratify_key,
    )
    train_records, val_records, test_records = stratified_split(
        records,
        label_key=split_config.stratify_key,
        seed=split_config.seed,
        val_size=split_config.val_size,
        test_size=split_config.test_size,
    )
    output_path = save_split_file(train_records, val_records, test_records, args.output, split_config)
    print(
        "Split complete: "
        f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)} -> {output_path}"
    )


def _stats_cli(args: argparse.Namespace) -> None:
    records = load_manifest(args.manifest)
    output_path = save_dataset_statistics(records, args.output, stratify_key=args.stratify_key)
    print(f"Saved dataset statistics for {len(records)} scans to {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COCA preprocessing and data-loading utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-coca-gated",
        help="Convert the raw Stanford COCA gated DICOM/XML release into NIfTI images and calcium masks.",
    )
    import_parser.add_argument("--output-root", required=True)
    import_parser.add_argument(
        "--output-manifest",
        help="Optional CSV/JSON manifest output path. Defaults to <output-root>/manifest.csv",
    )
    import_parser.add_argument(
        "--dataset-root",
        help="Path to the raw COCA Gated_release_final folder containing 'patient' and 'calcium_xml'.",
    )
    import_parser.add_argument("--patient-root", help="Optional explicit path to the COCA gated patient folder.")
    import_parser.add_argument("--xml-root", help="Optional explicit path to the COCA calcium_xml folder.")
    import_parser.add_argument("--max-cases", type=int)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.set_defaults(func=_import_coca_gated_cli)

    manifest_parser = subparsers.add_parser("build-manifest", help="Scan image/label folders and write a manifest.")
    manifest_parser.add_argument("--images-dir", required=True)
    manifest_parser.add_argument("--labels-dir")
    manifest_parser.add_argument("--metadata-csv")
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.set_defaults(func=_manifest_cli)

    resample_parser = subparsers.add_parser("resample", help="Resample a manifest of scans to a common spacing.")
    resample_parser.add_argument("--manifest", required=True)
    resample_parser.add_argument("--output-root", required=True)
    resample_parser.add_argument("--output-manifest", required=True)
    resample_parser.add_argument("--target-spacing", nargs=3, default=("1.0", "1.0", "1.0"))
    resample_parser.add_argument("--overwrite", action="store_true")
    resample_parser.set_defaults(func=_resample_cli)

    split_parser = subparsers.add_parser("split", help="Create train/val/test splits.")
    split_parser.add_argument("--manifest", required=True)
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--stratify-key", default="split_label")
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--val-size", type=float, default=0.1)
    split_parser.add_argument("--test-size", type=float, default=0.2)
    split_parser.set_defaults(func=_split_cli)

    stats_parser = subparsers.add_parser("stats", help="Compute dataset statistics from a manifest.")
    stats_parser.add_argument("--manifest", required=True)
    stats_parser.add_argument("--output", required=True)
    stats_parser.add_argument("--stratify-key")
    stats_parser.set_defaults(func=_stats_cli)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if getattr(args, "command", None) == "import-coca-gated" and not args.output_manifest:
        args.output_manifest = str(Path(args.output_root) / "manifest.csv")
    args.func(args)


if __name__ == "__main__":
    main()
