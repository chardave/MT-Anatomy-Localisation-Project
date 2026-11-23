# Arterial vessel segmentation on CREATE

How to install [arterial](https://github.com/FLOWCAT-CV/arterial) on the CREATE cluster and run its vessel segmentation on our CTAs, alongside TopBrain (Step 6 of [README_SYNTHETIC_FLUOROSCOPY_PIPELINE.md](README_SYNTHETIC_FLUOROSCOPY_PIPELINE.md)).

## What arterial gives you

- Input: one CTA as NIfTI.
- Segmentation output: a **binary** vessel mask (0 = background, 1 = vessel) on the input CTA grid, written to `<case_dir>/<mode>/segmentation.nii.gz`.
- Two modes:
  - `extracranial_vessels` (default): aortic arch → neck → Circle of Willis. Needs a CTA that includes the neck. TotalSegmentator's skull model splits head/neck; head uses nnU-Net 3d_fullres, neck 3d_lowres.
  - `intracranial_vessels`: head only, 3d_fullres. Use this on our cropped `10_Ready_CTA`.
- Optional later stages (extracranial only): centerlines with radii, vessel labelling into 14 classes (AA, BT, R/L CCA, R/L SA, R/L VA, R/L ICA, R/L ECA, BA, other), landmarks, catheter-access prediction.

Licence: code PolyForm Noncommercial 1.0.0, weights CC BY-NC 4.0. Fine for research; not for commercial use.

Supported OS is Linux/macOS only, so run it on CREATE, not on Windows.

## One-time setup (CREATE login node)

Do every step in this section on a **login node**: conda, pip and the weight download all need internet, and compute nodes may not have it. Keep everything on scratch, as with TopBrain.

### 1. Folder layout

```bash
export ARTERIAL_ROOT=/scratch/users/$USER/arterial
mkdir -p "$ARTERIAL_ROOT"/{cases,models,logs}
cd "$ARTERIAL_ROOT"
```

### 2. Conda environment

Install Miniforge on scratch rather than using the CREATE `anaconda3` module, which is missing on some login nodes and breaks inside activated environments. Then create the environment by path, so it lives on scratch rather than in the home quota:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /scratch/users/$USER/miniforge3
source /scratch/users/$USER/miniforge3/etc/profile.d/conda.sh
conda create -p "$ARTERIAL_ROOT/env" python=3.11 vmtk -y
conda activate "$ARTERIAL_ROOT/env"
```

VMTK is only needed for the centerline stages, but install it anyway so the full pipeline works later.

### 3. Python packages

The arterial README installs `torch==2.6.0` without choosing a CUDA build, but its PyG wheels are CUDA 12.4. Pin torch to the cu124 build so the two match. This pin is our addition, not from the arterial docs.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.6.1 monai torchio nnunetv2
pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 4. Install arterial

```bash
cd "$ARTERIAL_ROOT"
git clone https://github.com/FLOWCAT-CV/arterial.git
cd arterial
pip install -e .
```

### 5. Download the model weights (~1.1 GB)

```bash
export arterial_dir="$ARTERIAL_ROOT/arterial/arterial"
export ARTERIAL_MODELS_DIR="$ARTERIAL_ROOT/models"
bash download_models.sh --no-persist
```

The script downloads from Zenodo record 23488527, checks the SHA-256 checksum, extracts the weights into `$ARTERIAL_MODELS_DIR`, and checks that all 24 checkpoints are there. `--no-persist` stops it editing `~/.bashrc`. SLURM batch jobs don't read `~/.bashrc` anyway, so the job script exports these variables itself.

If the login node can't reach Zenodo, download on another machine and copy the archive across:

```bash
curl -L -o arterial-models-v2.tar.gz \
  "https://zenodo.org/records/23488527/files/arterial-models-v2.tar.gz?download=1"
# copy to CREATE, then:
tar xzf arterial-models-v2.tar.gz -C "$ARTERIAL_ROOT/models"
```

(This route skips the checksum and checkpoint checks.)

Expected layout:

```text
models/
├── access_prediction/
├── landmark_detection/
├── segmentation/        extracranial_vessels/, intracranial_vessels/, totalsegmentator_mandible/
└── vessel_labelling/
```

### 6. Check the install

```bash
python -c "import torch, nnunetv2, arterial; print(torch.__version__, torch.version.cuda)"
arterial --help
```

The CUDA check has to run on a GPU node: see the first lines of the job log below.

## Per-case inputs

Two runs per case are useful:

| Run | Input CTA | Mode | Purpose |
|---|---|---|---|
| A | `10_Ready_CTA.nii.gz` (the same file TopBrain gets) | `intracranial_vessels` | Direct voxel-wise comparison and fusion with TopBrain on the same grid |
| B | Full-field CTA including the neck: `7_Aligned_CTA.nii.gz` on the template grid (use `4_CTA_LPS.nii.gz` only on the native grid) | `extracranial_vessels` | Neck and arch vessels that TopBrain does not cover |

Upload into a per-case folder (from Windows, e.g. with `scp` or WinSCP):

```text
$ARTERIAL_ROOT/cases/<CASE_ID>/
├── ready/cta.nii.gz      ← 10_Ready_CTA.nii.gz   (run A)
└── full/cta.nii.gz       ← 7_Aligned_CTA.nii.gz  (run B)
```

arterial reads `<case_dir>/cta.nii.gz` by default. It documents no orientation or spacing requirements; nnU-Net resamples internally.

## Running (SLURM)

Segmentation only: the arterial docs don't give a single "segment only" flag, so skip every later stage explicitly with `-sce -sb -svl -sfe -sap -sld`. To run the full extracranial pipeline (centerlines, labels, landmarks) for run B, drop those flags.

The job script is [run_arterial_cta.slurm](run_arterial_cta.slurm). Copy it to `$ARTERIAL_ROOT` on CREATE. It checks the inputs, weights and CUDA, then runs arterial. Afterwards it confirms the mask is binary and on the input CTA grid. It won't overwrite an existing output unless you pass `OVERWRITE=1`.

The script activates the environment with a Miniforge installed at `/scratch/users/$USER/miniforge3` (the CREATE `anaconda3` module is unreliable; see Step 3.2 of [ARTERIAL_CASE_43_WALKTHROUGH.md](ARTERIAL_CASE_43_WALKTHROUGH.md)). Pass `CONDA_BASE=<path>` in `--export` if yours is elsewhere.

Submit from `$ARTERIAL_ROOT`:

```bash
sbatch --export=ALL,CASE_ID=<case>,RUN=ready run_arterial_cta.slurm
sbatch --export=ALL,CASE_ID=<case>,RUN=full  run_arterial_cta.slurm
sbatch --export=ALL,CASE_ID=<case>,RUN=full,SEG_ONLY=0 run_arterial_cta.slurm     # full pipeline
sbatch --export=ALL,CASE_ID=<case>,RUN=ready,EXTRA_ARGS="-clnn" run_arterial_cta.slurm   # clDice model
```

Logs go to `$ARTERIAL_ROOT/logs/arterial-<jobid>.out`.

Expect about 1 minute per case for segmentation (the authors report a median of 49 s on an RTX A5000). It needs about 8.5 GB of GPU memory and 16 GB of RAM at peak.

Useful extra flags:
- `-clnn`: use the clDice-trained nnU-Net, which may keep thin vessels better connected.
- `-s99`: binarise at probability 0.99 instead of the default threshold (fewer false positives).
- `-fast`: 3d_lowres for the whole volume. The docs say intracranial arteries come out less reliably; avoid it.

## Outputs and download

```text
cases/<CASE_ID>/ready/intracranial_vessels/segmentation.nii.gz   (run A)
cases/<CASE_ID>/full/extracranial_vessels/segmentation.nii.gz    (run B)
```

Download them and save them into the patient folder as:

```text
13b_Arterial_vessel_mask_ready.nii.gz   ← run A
13c_Arterial_vessel_mask_full.nii.gz    ← run B (still on the uncropped CTA grid)
```

## Checks after download

1. Run A must be on exactly the same grid as `10_Ready_CTA` and the TopBrain labels:

```python
import SimpleITK as sitk
a = sitk.ReadImage("13b_Arterial_vessel_mask_ready.nii.gz")
r = sitk.ReadImage("10_Ready_CTA.nii.gz")
for f in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
    print(f, getattr(a, f)() == getattr(r, f)())
print("labels:", sorted(set(sitk.GetArrayViewFromImage(a).ravel().tolist())))
```

If the grids differ, resample the mask onto `10_Ready_CTA` with nearest-neighbour interpolation before comparing.

2. Run B on `7_Aligned_CTA` is on the template grid. The ready grid is a crop of that grid, so cropping with `ready_grid.json` brings the mask into Ready space. [compare_arterial_topbrain.py](compare_arterial_topbrain.py) does the crop and the comparison with TopBrain; see [ARTERIAL_CASE_43_WALKTHROUGH.md](ARTERIAL_CASE_43_WALKTHROUGH.md). If run B was made on `4_CTA_LPS` instead, apply the CTA-to-template transform (nearest-neighbour) before cropping. The Ready crop may cut off most of the neck; if you need the neck for DRRs, the crop will have to be enlarged.

3. Look at both masks over the CTA in 3D Slicer. Things to look for: veins included, skull-base bone leaking into the mask, and whether the masks reach the M2/M3 and A2/A3 branches.

## Known gaps in the arterial docs

- `-sc/--skip_clipping` is accepted but does nothing in the current code.
- `-ns/--no_slicing` matters for run A. Without it, intracranial mode first cuts the CTA to the bounding box of a TotalSegmentator skull mask and segments only inside that box (the mask is then saved on the cut box, not the input grid). On case 43 the box started at slice 128 of the Ready CTA, above the Circle of Willis (slices ~50–90), so the run found 0.6 ml of vessel against TopBrain's 4.7 ml of artery and missed every proximal artery. The Ready CTA is already head-only, so the job script passes `-ns` for `RUN=ready` by default (`NO_SLICING=0` turns it off).
- The docs don't say whether the mask excludes veins, or how far into the distal branches intracranial mode reaches.
- Vessel labelling (the 14 classes) only exists in extracranial mode, and the labels are attached to centerline segments, not mask voxels.
