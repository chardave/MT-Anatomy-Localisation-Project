# MT Anatomy Localisation Project

Live anatomy localisation on fluoroscopy videos during mechanical thrombectomy procedures.

This repository contains scripts for building a multi-dataset CT/CTA scan index, preparing anatomical labelmaps, and running interactive fluoroscopy / DRR simulation using nanoDRR.

---

## Repository layout

```text
adapters/                      dataset adapters used by build_scan_index.py
metadata/                      label mappings and scan index outputs
subtraction_method/            CT/CTA registration and vessel-residual subtraction
synthetic_fluoro_generation/   numbered CT/CTA preprocessing pipeline + fluoroscopy simulators
xvr_project/scripts/           xvr / DiffDRR registration experiments
gen_fluoro_alike_radiographs/  early scan alignment script
requirements/                  pip requirements for the two environments
```

Imaging data and outputs (`subtraction_results/`, `synth_fluoro_gen_test/`, `xvr_project/input/`, `xvr_project/outputs/`, any `*.nii.gz`, `*.pth`, `*.mp4`, ...) are ignored by git and must never be committed. See `.gitignore`.

Earlier versions of each script are kept in the git history rather than as copies, e.g.

```bash
git log --follow synthetic_fluoro_generation/8_1_fluoroscopy_sim_nanodrr.py
```

## Environments

Two virtual environments, both Python 3.11:

| Environment | Used for | Requirements |
|---|---|---|
| main | everything except xvr: scan index, preprocessing, Elastix registration, subtraction, nanoDRR simulator | `requirements/fluoro.txt` |
| xvr | `xvr_project/` and `generate_kch_synthetic.py` (DiffDRR); separate because xvr needs an older PyTorch | `requirements/xvr.txt` |

Keep environments outside OneDrive (e.g. `C:\Users\<you>\venvs\`), then:

```bash
py -3.11 -m venv C:\Users\<you>\venvs\fluoro
C:\Users\<you>\venvs\fluoro\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/fluoro.txt
```

---

## Main tools

### `synthetic_fluoro_generation/8_1_fluoroscopy_sim_nanodrr.py`

Interactive fluoroscopy simulation app.

Features:

- Runs a PySide6 fluoroscopy simulation interface.
- Uses GPU acceleration through PyTorch/CUDA when available.
- Simulates C-arm pose control, with AP / lateral / oblique view presets.
- Allows the user to change the input CT image, segmentation and label mapping by browsing local files.
- Allows selected segmentations to be imposed on the simulated fluoroscopy view.
- Percentile windowing, gamma, quantum noise and MUSICA-like multiscale enhancement.
- Optional HU-based 3D vessel extraction with iodine-style attenuation boost; the extracted mask is shown side by side, projected with the same geometry as the DRR.

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

### `find_composite_candidates.py`

Searches through available segmentations to identify which anatomical region each scan mainly represents.

This is used to help select candidate scans for building composite body-region examples.

Outputs are saved in:

```text
metadata/composite_candidate_outputs/