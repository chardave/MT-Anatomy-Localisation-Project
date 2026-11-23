# Arterial vessel segmentation: Case 43 walkthrough

**Goal:** run arterial on case 43 on CREATE, bring the masks back, and compare them with the TopBrain labels for the same case.

**Case:** `MTAL_KCH_43` (template-grid run, 4–5 Nov 2026)
**Time needed:** about 45 min the first time (most of it the one-off install), about 15 min per case after that
**Background:** [README_CREATE_ARTERIAL.md](README_CREATE_ARTERIAL.md)

---

## 0. Overview

| # | Where | What | Time |
|---|---|---|---|
| 1 | Windows | Check the case 43 input files | 2 min |
| 2 | CREATE login node | Log in | 1 min |
| 3 | CREATE login node | One-time install: conda env, packages, arterial, weights | 30 min |
| 4 | Windows → CREATE | Upload the job script and case 43 CTAs | 5 min |
| 5 | CREATE | Submit run A (ready CTA, intracranial) | ~5 min incl. queue |
| 6 | CREATE | Submit run B (aligned full-field CTA, extracranial) | ~5 min incl. queue |
| 7 | CREATE → Windows | Download the two masks | 2 min |
| 8 | Windows | Crop run B onto the ready grid and compare with TopBrain | 2 min |
| 9 | 3D Slicer | Visual check | 10 min |

The two runs:

| Run | Input | arterial mode | Output grid | Why |
|---|---|---|---|---|
| **A** (`RUN=ready`) | `10_ready_CTA.nii.gz` | `intracranial_vessels` | ready grid (512×512×256) | Same input and grid as TopBrain, for a direct comparison |
| **B** (`RUN=full`) | `7_Aligned_CTA.nii.gz` | `extracranial_vessels` | template grid (512×512×518) | Includes the neck. Same grid as the ready CTA before cropping, so a simple crop brings it into ready space |

> **Note:** for run B, use `7_Aligned_CTA`, not `4_CTA_LPS`. The ready grid is a crop of the aligned CTA (`ready_grid.json`: slices 257–512 of 518). A mask made on the aligned CTA therefore only needs cropping. A mask made on `4_CTA_LPS` would also need the CTA-to-template transform applied.

---

## 1. Check the case 43 files (Windows)

All the case 43 files you need are in:

```text
MT-Anatomy-Localisation-Project\synth_fluoro_gen_test\MTAL_KCH_43\template\
```

- [ ] `10_ready_CTA.nii.gz` (~224 MB): input for run A
- [ ] `7_Aligned_CTA.nii.gz` (~457 MB): input for run B
- [ ] `13_CTA_vessel_labels.nii.gz`: TopBrain labels, for the comparison
- [ ] `ready_grid.json`: crop used to make the ready grid (`crop_index_xyz [0, 0, 257]`, `crop_size_xyz [512, 512, 256]`)

---

## 2. Log in to CREATE

In PowerShell:

```powershell
ssh k21013869@hpc.create.kcl.ac.uk
```

Everything in Steps 3–6 runs in this SSH session unless it says otherwise.

---

## 3. One-time install (login node)

> **Skip this step** if `/scratch/users/$USER/arterial/env` and `/scratch/users/$USER/arterial/models` already exist.

> **Use the login node.** pip, conda and the weight download all need internet access, which compute nodes may not have.

### 3.1 Folders, and keep caches off the home quota

```bash
export ARTERIAL_ROOT=/scratch/users/$USER/arterial
mkdir -p "$ARTERIAL_ROOT"/{cases,models,logs,pip_cache,conda_pkgs}
export PIP_CACHE_DIR="$ARTERIAL_ROOT/pip_cache"
export CONDA_PKGS_DIRS="$ARTERIAL_ROOT/conda_pkgs"
cd "$ARTERIAL_ROOT"
```

### 3.2 Install your own conda (Miniforge on scratch)

Don't use the CREATE `anaconda3` module. It has disappeared from some login nodes, and inside an activated environment its `conda` breaks with `No module named 'conda'`. Miniforge installed on scratch avoids both problems, and as it's conda-forge only, there's no Terms of Service prompt.

```bash
cd /scratch/users/$USER
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /scratch/users/$USER/miniforge3
source /scratch/users/$USER/miniforge3/etc/profile.d/conda.sh
conda --version
```

Each new SSH session only needs the `source` line (plus `conda activate`, below). No `module load` is needed; don't load `nvhpc` either, because the pip PyTorch wheels bring their own CUDA libraries.

### 3.3 Create the environment

```bash
conda create -p "$ARTERIAL_ROOT/env" python=3.11 vmtk -y
conda activate "$ARTERIAL_ROOT/env"
python -c "import vmtk, sys; print('vmtk ok', sys.version.split()[0])"
which pip        # must be /scratch/users/<you>/arterial/env/bin/pip
```

If the environment already exists (made earlier with the module), skip `conda create`: the Miniforge conda can activate it directly. To leave the environment, run `conda deactivate`.

### 3.4 Install the Python packages

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.6.1 monai torchio nnunetv2
pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 3.5 Install arterial

```bash
cd "$ARTERIAL_ROOT"
git clone https://github.com/FLOWCAT-CV/arterial.git
cd arterial
pip install -e .
```

### 3.6 Download the weights (~1.1 GB)

```bash
export arterial_dir="$ARTERIAL_ROOT/arterial/arterial"
export ARTERIAL_MODELS_DIR="$ARTERIAL_ROOT/models"
bash download_models.sh --no-persist
```

When it finishes, the script reports that verification passed. Check:

```bash
ls "$ARTERIAL_MODELS_DIR/segmentation"
# extracranial_vessels  intracranial_vessels  totalsegmentator_mandible
```

### 3.7 Check the install

```bash
python -c "import torch, nnunetv2, SimpleITK, arterial; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
arterial --help | head -20
```

- [ ] torch prints `2.6.0+cu124` and `cuda build 12.4`
- [ ] `arterial --help` prints the usage

> **Optional GPU test** before submitting batch jobs:
> ```bash
> srun -p interruptible_gpu --gres=gpu:1 --constraint=a40 --mem=32G --cpus-per-task=8 -t 00:20:00 --pty bash -l
> source /scratch/users/$USER/miniforge3/etc/profile.d/conda.sh; conda activate /scratch/users/$USER/arterial/env
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> exit
> ```

---

## 4. Upload the job script and the case 43 CTAs

### 4.1 Make the case folders (CREATE)

```bash
mkdir -p /scratch/users/$USER/arterial/cases/MTAL_KCH_43/{ready,full}
```

### 4.2 Upload (a new local PowerShell window, not the SSH session)

```powershell
cd "C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\MT-Anatomy-Localisation-Project"

# job script
scp synthetic_fluoro_generation\run_arterial_cta.slurm k21013869@hpc.create.kcl.ac.uk:/scratch/users/k21013869/arterial/

# run A input -> ready/cta.nii.gz
scp synth_fluoro_gen_test\MTAL_KCH_43\template\10_ready_CTA.nii.gz k21013869@hpc.create.kcl.ac.uk:/scratch/users/k21013869/arterial/cases/MTAL_KCH_43/ready/cta.nii.gz

# run B input -> full/cta.nii.gz
scp synth_fluoro_gen_test\MTAL_KCH_43\template\7_Aligned_CTA.nii.gz k21013869@hpc.create.kcl.ac.uk:/scratch/users/k21013869/arterial/cases/MTAL_KCH_43/full/cta.nii.gz
```

> **Note:** arterial always looks for `cta.nii.gz` in the case folder, so the files are renamed on upload.

### 4.3 Confirm the job script loads conda (CREATE)

```bash
grep -n "CONDA_BASE=" /scratch/users/$USER/arterial/run_arterial_cta.slurm
# ...: CONDA_BASE="${CONDA_BASE:-/scratch/users/$USER/miniforge3}"
```

The script uses the Miniforge from Step 3.2. If you installed it somewhere else, submit with `CONDA_BASE=<path>` in `--export`.

### 4.4 Check (CREATE)

```bash
ls -lh /scratch/users/$USER/arterial/cases/MTAL_KCH_43/*/cta.nii.gz
```

- [ ] `ready/cta.nii.gz` is ~214 MB and `full/cta.nii.gz` is ~435 MB. `ls -h` shows MiB, so these match the Windows sizes.

---

## 5. Run A: ready CTA, intracranial

```bash
cd /scratch/users/$USER/arterial
sbatch --export=ALL,CASE_ID=MTAL_KCH_43,RUN=ready run_arterial_cta.slurm
# the script adds -ns (no skull-box slicing) for RUN=ready; the log shows "No slicing (-ns): 1"
```

The job ID is printed: `Submitted batch job 1234567`.

Watch it:

```bash
squeue -u $USER
tail -f logs/arterial-<jobid>.out        # Ctrl+C to stop watching
```

A good log looks like this, in order:

1. The header: `Case: MTAL_KCH_43`, `Run: ready (intracranial_vessels)`
2. The `nvidia-smi` table showing an A40
3. `CUDA available: True` and `GPU: NVIDIA A40`
4. arterial's own progress messages
5. `Mask values: [0, 1]` and a non-zero `Vessel voxels:` count
6. `Size OK`, `Spacing OK`, `Origin OK`, `Direction OK`
7. `arterial segmentation completed successfully.`

- [ ] Output exists: `cases/MTAL_KCH_43/ready/intracranial_vessels/segmentation.nii.gz`

---

## 6. Run B: aligned full-field CTA, extracranial

```bash
sbatch --export=ALL,CASE_ID=MTAL_KCH_43,RUN=full run_arterial_cta.slurm
```

Check the log the same way. The header should now say `Run: full (extracranial_vessels)`. This mode first runs the skull split, so expect a little more output.

- [ ] Output exists: `cases/MTAL_KCH_43/full/extracranial_vessels/segmentation.nii.gz`

> **Optional:** to get arterial's centerlines, the 14 vessel labels and the landmarks as well, run run B again with the full pipeline:
> ```bash
> sbatch --export=ALL,CASE_ID=MTAL_KCH_43,RUN=full,SEG_ONLY=0,OVERWRITE=1 run_arterial_cta.slurm
> ```

You can submit runs A and B together; they don't depend on each other.

---

## 7. Download the masks (local PowerShell)

```powershell
cd "C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\MT-Anatomy-Localisation-Project\synth_fluoro_gen_test\MTAL_KCH_43\template"

scp k21013869@hpc.create.kcl.ac.uk:/scratch/users/k21013869/arterial/cases/MTAL_KCH_43/ready/intracranial_vessels/segmentation.nii.gz 13b_Arterial_vessel_mask_ready.nii.gz

scp k21013869@hpc.create.kcl.ac.uk:/scratch/users/k21013869/arterial/cases/MTAL_KCH_43/full/extracranial_vessels/segmentation.nii.gz 13c_Arterial_vessel_mask_full_template.nii.gz
```

- [ ] `13b_Arterial_vessel_mask_ready.nii.gz`: run A, on the ready grid
- [ ] `13c_Arterial_vessel_mask_full_template.nii.gz`: run B, on the template grid (518 slices)

---

## 8. Crop run B and compare with TopBrain (Windows)

[compare_arterial_topbrain.py](compare_arterial_topbrain.py) does this in one go:

1. Checks that the run B mask is on the `7_Aligned_CTA` grid.
2. Crops it with `ready_grid.json` to give `13c_Arterial_vessel_mask_full_cropped_ready.nii.gz`.
3. Checks that both masks are on the `10_ready_CTA` grid.
4. Compares each mask with `13_CTA_vessel_labels.nii.gz` (TopBrain).

Run it with the `totalseg_env` Python, which already has SimpleITK:

```powershell
cd "C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\MT-Anatomy-Localisation-Project\synthetic_fluoro_generation"

& "..\..\totalseg_env\Scripts\python.exe" compare_arterial_topbrain.py `
    --case-dir "..\synth_fluoro_gen_test\MTAL_KCH_43\template" `
    --ready-mask 13b_Arterial_vessel_mask_ready.nii.gz `
    --full-mask 13c_Arterial_vessel_mask_full_template.nii.gz
```

### How to read the output

For each mask the script prints:

| Line | Meaning |
|---|---|
| `arterial vessel volume` | Total vessel volume arterial segmented (ml) |
| `TopBrain volume (all labels)` / `(arteries)` | TopBrain's volume with and without the veins (labels 35–40) |
| `Dice vs TopBrain arteries` | Overall agreement with TopBrain's arterial labels. **This is the main number**, because arterial is not expected to segment veins |
| `arterial voxels not labelled by TopBrain` | Volume arterial finds that TopBrain doesn't: extra small vessels, veins or false positives |
| Per-label table | For each TopBrain vessel, the % of its voxels arterial also segments, with a bar |

What to look for in the per-label table:
- **ICA, BA, VA, M1, A1A2, P1P2** should be close to 100%. If not, one of the models is missing a main vessel.
- **M2/M3, A3, P3P4, SCA, PICA, AChA, OA** show how far distally arterial reaches.
- **Veins (35–40)**: near 0% means arterial leaves the veins out. A high value means it includes them.
- **Run A vs run B:** compare the two blocks. Run B uses the extracranial model's head path, so it can differ from run A.

The numbers are also saved to `MTAL_KCH_43\template\arterial_vs_topbrain.csv`.

> **For reference:** TopBrain's case 43 label map is 10.6 ml in total, of which only 4.7 ml is arterial. The superior sagittal sinus alone is ~5 ml. If you don't exclude the veins, Dice against TopBrain's full map will look poor even if the arteries match perfectly.

---

## 9. Visual check in 3D Slicer

1. **File → Add Data** and load, from `MTAL_KCH_43\template`:
   - `10_ready_CTA.nii.gz` (volume)
   - `13_CTA_vessel_labels.nii.gz`: tick **Show Options → LabelMap**, or load it as a segmentation
   - `13b_Arterial_vessel_mask_ready.nii.gz`: as a segmentation
   - `13c_Arterial_vessel_mask_full_cropped_ready.nii.gz`: as a segmentation
2. In **Segmentations**, give each arterial mask one plain colour (e.g. yellow for run A, cyan for run B), and leave TopBrain in its own colours.
3. Click **Show 3D** for each segmentation and turn them on and off one at a time.
4. Also open `7_Aligned_CTA.nii.gz` with `13c_Arterial_vessel_mask_full_template.nii.gz` to see the neck and arch vessels that TopBrain doesn't cover.

Things to note for case 43:
- [ ] Does arterial include the venous sinuses (SSS, straight sinus)?
- [ ] How far distally do the MCA/ACA/PCA branches go compared with TopBrain?
- [ ] Any bone or calcification leakage at the skull base or the carotid siphon?
- [ ] Are the ICA and VA continuous through the skull base in run B?
- [ ] Does run B reach the aortic arch, or does the field of view stop before it?

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: Miniforge not found at ...` | Miniforge isn't at `/scratch/users/<you>/miniforge3`. Install it (Step 3.2) or pass `CONDA_BASE=<path>` (Step 4.3) |
| `ERROR: Model weights not found` | Step 3.6 wasn't finished, or the weights are somewhere other than `/scratch/users/$USER/arterial/models` |
| `CUDA available: False` | torch is a CPU or wrong-CUDA build: reinstall it with the `--index-url .../cu124` line. Also check that the job landed on a GPU node (`nvidia-smi` in the log) |
| `ERROR: Output already exists` | Resubmit with `OVERWRITE=1` if you really want to redo it |
| Job sits in `PD` for a long time | The A40s are busy. Remove `--constraint=a40` (any GPU with ≥ 10 GB will do) |
| Job killed partway (interruptible partition) | Resubmit. If it keeps happening, use a non-interruptible GPU partition |
| `ModuleNotFoundError` for e.g. `cc3d` or `totalsegmentator` | On the login node, activate the env and `pip install connected-components-3d` (or the missing package), then resubmit |
| Comparison script: `not on the 7_Aligned_CTA grid` | Run B was made on a different file. Re-upload `7_Aligned_CTA.nii.gz` as `full/cta.nii.gz` |
| Comparison script: `not on the ready grid` | For run A, the upload wasn't `10_ready_CTA.nii.gz`, or arterial changed the grid. Resample the mask with nearest-neighbour onto `10_ready_CTA` |

---

## Done checklist

- [ ] Install finished and checked (Step 3)
- [ ] Run A finished: `13b_Arterial_vessel_mask_ready.nii.gz`
- [ ] Run B finished: `13c_Arterial_vessel_mask_full_template.nii.gz`
- [ ] Cropped run B: `13c_Arterial_vessel_mask_full_cropped_ready.nii.gz`
- [ ] `arterial_vs_topbrain.csv` saved
- [ ] Slicer notes written up (veins, distal reach, leakage, neck continuity)
