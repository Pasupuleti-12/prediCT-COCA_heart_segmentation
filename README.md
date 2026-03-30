# PrediCT GSoC 2026: COCA Preprocessing + Project 1 Heart Segmentation

This repository contains a submission-ready implementation for the PrediCT applicant tasks built around the Stanford COCA cardiac CT dataset and Project 1 (calcium segmentation preparation via coarse whole-heart segmentation).

The code covers:

- Raw Stanford COCA gated DICOM/XML import directly from `Gated_release_final`.
- COCA manifest creation, spacing normalization, HU windowing, stratified splitting, dataset statistics, and efficient MONAI dataloaders.
- TotalSegmentator-based pseudo-ground-truth preparation for whole-heart masks on 30-50 scans.
- A lightweight 3D U-Net training pipeline for coarse heart segmentation.
- Test-set evaluation with Dice, bounding-box IoU, inference timing, and visualization export.

## Recommended Environment

Use Python 3.10 or 3.11. MONAI and TotalSegmentator are typically smoother there than on newer interpreters.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repository Layout

- `src/coca_pipeline.py`: raw COCA gated import, manifest building, resampling, HU windowing, dataset statistics, splits, samplers, MONAI dataloaders
- `src/prepare_totalseg_labels.py`: TotalSegmentator runner + heart-mask extraction
- `src/train_heart_segmentation.py`: lightweight 3D U-Net training
- `src/evaluate_heart_segmentation.py`: Dice/timing/bbox evaluation + PNG overlays
- `docs/submission_justification.md`: written justification text for the application
- `docs/coca_pipeline_instructions.md`: step-by-step workflow
- `notebooks/heart_segmentation_evaluation.ipynb`: notebook template for plots and qualitative review

## Expected Data Layout

If you start from the raw Stanford COCA download, keep the original `Gated_release_final` anywhere on disk and let this repo build a working tree like this:

```text
data/
  raw_gated/
    images/
    labels/
    metadata/
  resampled/
    images/
    labels/
  labels/
    masks/
  manifests/
  splits/
reports/
runs/
```

If you already have NIfTI volumes, the simpler layout below still works:

```text
data/
  raw/
    images/
  resampled/
    images/
  labels/
    masks/
```

If you have additional metadata for stratification, keep it as a CSV with a `case_id` column plus one or more label columns such as `split_label`, `site`, or `has_calcium`.

## End-to-End Workflow

### 1. Import the raw Stanford COCA gated subset

```powershell
python src/coca_pipeline.py import-coca-gated `
  --dataset-root "D:\path\to\COCA\cocacoronarycalciumandchestcts-2\Gated_release_final" `
  --output-root data/raw_gated `
  --output-manifest data/manifests/raw_gated_manifest.csv
```

This reads the raw gated DICOM series plus `calcium_xml`, writes NIfTI CT volumes and calcium masks, and produces the initial manifest.

Optional smoke test on a single case first:

```powershell
python src/coca_pipeline.py import-coca-gated `
  --dataset-root "D:\path\to\COCA\cocacoronarycalciumandchestcts-2\Gated_release_final" `
  --output-root data/raw_gated `
  --output-manifest data/manifests/raw_gated_manifest.csv `
  --max-cases 1
```

### 2. Resample the dataset to a common spacing

```powershell
python src/coca_pipeline.py resample `
  --manifest data/manifests/raw_gated_manifest.csv `
  --output-root data/resampled `
  --output-manifest data/manifests/resampled_manifest.csv `
  --target-spacing 1.0 1.0 1.0
```

### 3. Generate TotalSegmentator heart masks for 30-50 scans

```powershell
python src/prepare_totalseg_labels.py `
  --images-dir data/resampled/images `
  --metadata-csv data/manifests/resampled_manifest.csv `
  --output-dir data/labels `
  --manifest-output data/manifests/project1_manifest.csv `
  --max-cases 40
```

If your TotalSegmentator setup requires a license number, add `--license-number YOUR_KEY`.

### 4. Export dataset statistics

```powershell
python src/coca_pipeline.py stats `
  --manifest data/manifests/project1_manifest.csv `
  --stratify-key split_label `
  --output reports/dataset_statistics.json
```

### 5. Create stratified train/val/test splits

```powershell
python src/coca_pipeline.py split `
  --manifest data/manifests/project1_manifest.csv `
  --stratify-key split_label `
  --seed 42 `
  --val-size 0.1 `
  --test-size 0.2 `
  --output data/splits/project1_split.json
```

### 6. Train the coarse heart model

```powershell
python src/train_heart_segmentation.py `
  --split-file data/splits/project1_split.json `
  --output-dir runs/heart_unet `
  --epochs 100 `
  --batch-size 2 `
  --cache-mode memory `
  --amp
```

### 7. Evaluate Dice and timing

```powershell
python src/evaluate_heart_segmentation.py `
  --split-file data/splits/project1_split.json `
  --checkpoint runs/heart_unet/best_model.pt `
  --output-dir runs/heart_unet/eval `
  --num-visualizations 5
```

### 8. Alternative entry point for pre-converted NIfTI data

If you already have NIfTI images and labels, you can skip the raw COCA DICOM/XML import and build a manifest directly:

```powershell
python src/coca_pipeline.py build-manifest `
  --images-dir data/raw/images `
  --metadata-csv data/metadata.csv `
  --output data/manifests/raw_manifest.csv
```

### 9. Open the evaluation notebook

Point the notebook at `runs/heart_unet/eval` and use the generated CSV/JSON/PNG files in your submission.

## Submission Checklist

- Pipeline code: included in `src/`
- Data loader: included in `src/coca_pipeline.py`
- Written justification: included in `docs/submission_justification.md`
- Dataset statistics: generate `reports/dataset_statistics.json`
- Model weights: `runs/heart_unet/best_model.pt`
- Evaluation notebook/artifacts: `notebooks/heart_segmentation_evaluation.ipynb` plus `runs/heart_unet/eval/*`
