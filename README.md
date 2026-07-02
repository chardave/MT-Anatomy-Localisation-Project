# MT Anatomy Localisation Project

Live anatomy localisation on fluoroscopy videos during mechanical thrombectomy procedures.

This repository contains scripts for building a multi-dataset CT/CTA scan index, preparing anatomical labelmaps, and running interactive fluoroscopy / DRR simulation using nanoDRR.

---

## Main tools

### `nanodrr-test/fluoroscopy_sim.py`

Interactive fluoroscopy simulation app.

Features:

- Runs a PySide6 fluoroscopy simulation interface.
- Uses GPU acceleration through PyTorch/CUDA when available.
- Simulates C-arm pose control.
- Allows the user to change the input CT image and segmentation by browsing local files.
- Allows selected segmentations to be imposed on the simulated fluoroscopy view.
- Supports toggling and selecting visible anatomical labels.

---

### `build_scan_index.py`

Builds a unified scan index across multiple datasets.

The output CSV stores:

- `case_id`
- `dataset`
- `image_path`
- `label_path`
- `anatomical_region`
- `labels_present`
- `annotation_type`

Each dataset has its own adapter script in the `adapters/` folder. The adapter scripts standardise different dataset folder structures into a common `ScanRecord` format.

---

### `adapters/`

Dataset-specific adapters.

Current adapters include:

- TotalSegmentator
- TopAneu
- SegA
- ASOCA

Each adapter finds valid image/segmentation pairs and assigns useful metadata such as anatomical region, labels present, and annotation type.

---

### `check_ct.py`

Repairs NIfTI header geometry so that ITK-compatible tools can open the file.

Some scans contain affine matrices that are accepted by nanoDRR but not by ITK-based tools. ITK requires orthonormal direction cosines, so this script is mainly useful for viewing scans in tools such as 3D Slicer or ITK-SNAP.

For nanoDRR experiments, the original scan should usually be kept unless the corresponding segmentation is corrected in the same way.

---

### `find_composite_candidate.py`

Searches through available segmentations to identify which anatomical region each scan mainly represents.

This is used to help select candidate scans for building composite body-region examples.

Outputs are saved in:

```text
metadata/composite_candidate_output/