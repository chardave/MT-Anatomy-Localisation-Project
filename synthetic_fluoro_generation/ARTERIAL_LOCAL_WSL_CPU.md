# Arterial on the laptop CPU (WSL): case 43 test

A stand-in for CREATE while it's unavailable. It runs arterial on the laptop **CPU** inside WSL2 Ubuntu. It's slow, but fine for testing. For the CREATE version, see [ARTERIAL_CASE_43_WALKTHROUGH.md](ARTERIAL_CASE_43_WALKTHROUGH.md).

**Why CPU:** the laptop GPU (RTX PRO 500 Blackwell, 6 GB) has less memory than arterial needs (~8.5 GB), and Blackwell isn't supported by arterial's pinned PyTorch build.

**Laptop as checked:** WSL Ubuntu 26.04, 14 CPU threads, Miniforge already installed in WSL (`~/miniforge3`, conda-forge only, so no Terms of Service prompt), WSL memory limit 15 GB.

---

## 1. Give WSL more memory (Windows, once)

arterial peaks at ~16 GB of RAM, and WSL is currently capped at 15 GB. The laptop has 31 GB.

1. In Windows, open (or create) `C:\Users\cd26\.wslconfig` in Notepad.
2. Put in:

   ```ini
   [wsl2]
   memory=24GB
   swap=16GB
   ```

3. In PowerShell, run `wsl --shutdown`, wait 10 seconds, then start Ubuntu again.
4. In Ubuntu, `free -g` should now show ~23 GB total.

---

## 2. Open Ubuntu and set up folders

From the Start menu open **Ubuntu**, or run `wsl -d Ubuntu` in PowerShell. Everything from here runs in Ubuntu unless it says otherwise.

```bash
export ARTERIAL_ROOT=$HOME/arterial
mkdir -p "$ARTERIAL_ROOT"/{cases,models,logs}
cd "$ARTERIAL_ROOT"
```

---

## 3. Create the environment (once)

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -p "$ARTERIAL_ROOT/env" python=3.11 vmtk -y
conda activate "$ARTERIAL_ROOT/env"
which python pip        # both should be in /home/cd26/arterial/env/bin/
```

### CPU-only PyTorch and the other packages

Use the **CPU** builds. They're smaller, and they never try to use the unsupported GPU.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.6.1 monai torchio nnunetv2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

### arterial and the weights

```bash
cd "$ARTERIAL_ROOT"
git clone https://github.com/FLOWCAT-CV/arterial.git
cd arterial
pip install -e .

export arterial_dir="$ARTERIAL_ROOT/arterial/arterial"
export ARTERIAL_MODELS_DIR="$ARTERIAL_ROOT/models"
bash download_models.sh --no-persist
```

### Check the install

```bash
python -c "import torch, nnunetv2, SimpleITK, arterial; print(torch.__version__, 'cuda?', torch.cuda.is_available())"
# expect: 2.6.0+cpu cuda? False
ls "$ARTERIAL_MODELS_DIR/segmentation"
# extracranial_vessels  intracranial_vessels  totalsegmentator_mandible
```

### Save the settings for next time

```bash
cat >> ~/.bashrc <<'EOF'
# arterial (CPU test)
export ARTERIAL_ROOT=$HOME/arterial
export arterial_dir=$ARTERIAL_ROOT/arterial/arterial
export ARTERIAL_MODELS_DIR=$ARTERIAL_ROOT/models
alias arterial_env='source ~/miniforge3/etc/profile.d/conda.sh && conda activate $ARTERIAL_ROOT/env'
EOF
```

In later sessions, typing `arterial_env` is enough to get started.

---

## 4. Copy the case 43 CTAs into WSL

WSL can read the Windows drive under `/mnt/c`. Copying the files into WSL first is much faster than reading from OneDrive while arterial runs. If OneDrive shows the files as cloud-only, right-click the case folder in Windows and choose **Always keep on this device** first.

```bash
WIN_CASE="/mnt/c/Users/cd26/OneDrive - King's College London/MT Anatomy Localisation Project/MT-Anatomy-Localisation-Project/synth_fluoro_gen_test/MTAL_KCH_43/template"
ls -lh "$WIN_CASE"/10_ready_CTA.nii.gz "$WIN_CASE"/7_Aligned_CTA.nii.gz

mkdir -p "$ARTERIAL_ROOT"/cases/MTAL_KCH_43/{ready,full}
cp "$WIN_CASE/10_ready_CTA.nii.gz"  "$ARTERIAL_ROOT/cases/MTAL_KCH_43/ready/cta.nii.gz"
cp "$WIN_CASE/7_Aligned_CTA.nii.gz" "$ARTERIAL_ROOT/cases/MTAL_KCH_43/full/cta.nii.gz"
```

---

## 5. Run A: ready CTA, intracranial (do this first)

```bash
cd "$ARTERIAL_ROOT"
time arterial -cd cases/MTAL_KCH_43/ready -m intracranial_vessels \
    -sce -sb -svl -sfe -sap -sld -rs scipy -nt 12 \
    2>&1 | tee logs/MTAL_KCH_43_ready.log
```

- `-sce -sb -svl -sfe -sap -sld`: segmentation only, skipping all later stages
- `-rs scipy`: resample on the CPU (the default `torch` resampling is meant for the GPU)
- `-nt 12`: use 12 of the 14 threads, leaving some for Windows

Expect roughly **10–30 minutes**. The laptop will be busy, so keep it plugged in. When it's done:

```bash
ls -lh cases/MTAL_KCH_43/ready/intracranial_vessels/segmentation.nii.gz
```

---

## 6. Run B: aligned full-field CTA, extracranial (optional, slower)

This CTA has about twice as many slices, and the skull split runs first, so expect roughly twice as long.

```bash
time arterial -cd cases/MTAL_KCH_43/full -m extracranial_vessels \
    -sce -sb -svl -sfe -sap -sld -rs scipy -nt 12 \
    2>&1 | tee logs/MTAL_KCH_43_full.log
```

---

## 7. Copy the results back to Windows

```bash
cp "$ARTERIAL_ROOT/cases/MTAL_KCH_43/ready/intracranial_vessels/segmentation.nii.gz" "$WIN_CASE/13b_Arterial_vessel_mask_ready.nii.gz"
cp "$ARTERIAL_ROOT/cases/MTAL_KCH_43/full/extracranial_vessels/segmentation.nii.gz"  "$WIN_CASE/13c_Arterial_vessel_mask_full_template.nii.gz"   # only if you ran run B
```

(If you opened a new terminal, set `WIN_CASE` again as in Step 4.)

---

## 8. Compare with TopBrain (Windows PowerShell)

This is the same as Step 8 of the CREATE walkthrough. Leave out `--full-mask` if you only ran run A.

```powershell
cd "C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\MT-Anatomy-Localisation-Project\synthetic_fluoro_generation"

& "..\..\totalseg_env\Scripts\python.exe" compare_arterial_topbrain.py `
    --case-dir "..\synth_fluoro_gen_test\MTAL_KCH_43\template" `
    --ready-mask 13b_Arterial_vessel_mask_ready.nii.gz
```

Then do the 3D Slicer check (Step 9 of the walkthrough).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Process `Killed` with no Python error | WSL ran out of memory. Do Step 1 (raise `memory`/`swap`), close other apps, or lower `-nt` to 6 |
| `CUDA` / `no kernel image` errors | A GPU build of torch slipped in. Run `pip uninstall -y torch`, then reinstall it with the `--index-url .../whl/cpu` line |
| `ModuleNotFoundError: cc3d` (or similar) | `pip install connected-components-3d` (or the package named), then rerun |
| `Neither ARTERIAL_MODELS_DIR nor arterial_dir is set` | New terminal: run `arterial_env` and check the `export` lines in `~/.bashrc` |
| `cp: cannot stat ... /mnt/c/...` | Set `WIN_CASE` again. Check the files are downloaded locally in OneDrive, not cloud-only |
| Very slow (> 1 h) | Normal on a slow CPU. Run only run A for testing |

The Linux environment, models and cases take about 5 GB in WSL. Delete `~/arterial` to remove them.
