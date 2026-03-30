# COCA Pipeline Instructions

This repo is organized to satisfy both the common PrediCT preprocessing task and the Project 1 heart-segmentation task. The intended workflow is:

1. Download the COCA scans using the PrediCT instructions.
2. Import the raw gated DICOM + XML release into NIfTI images and calcium masks.
3. Resample all scans to a common voxel spacing.
4. Generate TotalSegmentator heart masks on 30-50 cases.
5. Compute dataset statistics and create a stratified split.
6. Train and evaluate the lightweight 3D U-Net.

## Command Sequence

### Import the raw Stanford COCA gated release

```powershell
python src/coca_pipeline.py import-coca-gated `
  --dataset-root "D:\path\to\COCA\cocacoronarycalciumandchestcts-2\Gated_release_final" `
  --output-root data/raw_gated `
  --output-manifest data/manifests/raw_gated_manifest.csv
```

### Resample

```powershell
python src/coca_pipeline.py resample `
  --manifest data/manifests/raw_gated_manifest.csv `
  --output-root data/resampled `
  --output-manifest data/manifests/resampled_manifest.csv `
  --target-spacing 1.0 1.0 1.0
```

### Generate heart masks

```powershell
python src/prepare_totalseg_labels.py `
  --images-dir data/resampled/images `
  --metadata-csv data/manifests/resampled_manifest.csv `
  --output-dir data/labels `
  --manifest-output data/manifests/project1_manifest.csv `
  --max-cases 40
```

### Dataset statistics

```powershell
python src/coca_pipeline.py stats `
  --manifest data/manifests/project1_manifest.csv `
  --stratify-key split_label `
  --output reports/dataset_statistics.json
```

### Stratified split

```powershell
python src/coca_pipeline.py split `
  --manifest data/manifests/project1_manifest.csv `
  --stratify-key split_label `
  --output data/splits/project1_split.json
```

### Train

```powershell
python src/train_heart_segmentation.py `
  --split-file data/splits/project1_split.json `
  --output-dir runs/heart_unet `
  --epochs 100 `
  --batch-size 2 `
  --cache-mode memory `
  --amp
```

### Evaluate

```powershell
python src/evaluate_heart_segmentation.py `
  --split-file data/splits/project1_split.json `
  --checkpoint runs/heart_unet/best_model.pt `
  --output-dir runs/heart_unet/eval `
  --num-visualizations 5
```

## Outputs

- `data/raw_gated/images/*.nii.gz`: imported gated CT volumes
- `data/raw_gated/labels/*_seg.nii.gz`: calcium masks created from the Stanford XML files
- `data/raw_gated/metadata/*.json`: per-case import metadata
- `data/manifests/*.csv`: dataset manifests
- `data/splits/project1_split.json`: stratified split file
- `reports/dataset_statistics.json`: dataset statistics for the write-up
- `runs/heart_unet/best_model.pt`: best checkpoint
- `runs/heart_unet/eval/case_metrics.csv`: case-level Dice and timing
- `runs/heart_unet/eval/evaluation_summary.json`: aggregate evaluation summary
- `runs/heart_unet/eval/figures/*.png`: qualitative overlays
