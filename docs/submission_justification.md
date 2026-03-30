# Submission Justification

## Common Task Justification

The preprocessing pipeline starts from the raw Stanford COCA gated DICOM/XML release, converts each series into a consistent NIfTI image plus calcium mask pair, and then standardizes those volumes in a way that is directly aligned with segmentation training. All scans are resampled to a common isotropic spacing so the network sees anatomically comparable voxel geometry, and cardiac-window HU normalization clips irrelevant extremes while preserving both soft tissue and hyperdense calcium. The training transform stack uses MONAI with foreground cropping, label-aware positive/negative patch sampling, spatial flips, affine perturbations, and mild intensity noise/shift augmentation. This gives the model enough variation to learn robust heart localization without destroying the spatial structure that matters in volumetric CT.

To support reproducible experimentation, the repo also exports manifest-driven dataset statistics, deterministic train/val/test splits, and optional weighted sampling for imbalanced metadata labels. Persistent or in-memory dataset caching is included so repeated runs avoid unnecessary I/O bottlenecks. For Project 1 specifically, the pipeline is optimized around coarse whole-heart segmentation, which is the fastest path to deriving a reliable cardiac bounding region before any more specific calcium analysis stage.

## Model Choice Justification

I selected a lightweight 3D U-Net because it provides a strong speed/accuracy tradeoff for coarse volumetric organ segmentation on a modest dataset size. The architecture preserves enough 3D context to model heart shape and location while remaining substantially cheaper than running TotalSegmentator at inference time. In this project, the model is intended to recover a coarse whole-heart mask or bounding region rather than a fine-grained anatomical parse, so a compact 3D U-Net is a practical and well-justified baseline for reaching the target Dice while keeping inference latency low.
