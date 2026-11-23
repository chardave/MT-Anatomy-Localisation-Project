#!/bin/bash -l

#SBATCH --job-name=arterial
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/users/%u/arterial/logs/arterial-%j.out

# arterial (FLOWCAT-CV) vessel segmentation for one case.
# Setup and folder layout: README_CREATE_ARTERIAL.md
#
# sbatch --export=ALL,CASE_ID=<case>,RUN=ready run_arterial_cta.slurm   # 10_Ready_CTA, intracranial mode
# sbatch --export=ALL,CASE_ID=<case>,RUN=full  run_arterial_cta.slurm   # uncropped CTA, extracranial mode
#
# Optional: SEG_ONLY=0 (full pipeline), OVERWRITE=1, EXTRA_ARGS="-clnn -s99", ARTERIAL_ROOT=<path>,
#           NO_SLICING=0|1 (default 1 for ready, 0 for full), CONDA_BASE=<miniforge path>

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

CASE_ID="${CASE_ID:?Set CASE_ID: sbatch --export=ALL,CASE_ID=<case>,RUN=ready run_arterial_cta.slurm}"
RUN="${RUN:-ready}"
SEG_ONLY="${SEG_ONLY:-1}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ARTERIAL_ROOT="${ARTERIAL_ROOT:-/scratch/users/$USER/arterial}"

case "$RUN" in
    ready) MODE=intracranial_vessels; NO_SLICING="${NO_SLICING:-1}" ;;
    full)  MODE=extracranial_vessels; NO_SLICING="${NO_SLICING:-0}" ;;
    *) echo "ERROR: RUN must be 'ready' or 'full', got '$RUN'"; exit 1 ;;
esac
# NO_SLICING=1 passes -ns: segment the whole CTA instead of first cutting it to arterial's
# skull bounding box. The Ready CTA is already head-only, and on case 43 the skull box
# started above the Circle of Willis, so the default (slicing) missed almost every artery.

CASE_DIR="$ARTERIAL_ROOT/cases/$CASE_ID/$RUN"
CTA="$CASE_DIR/cta.nii.gz"
OUT="$CASE_DIR/$MODE/segmentation.nii.gz"

if [[ ! -s "$CTA" ]]; then
    echo "ERROR: Input CTA does not exist: $CTA"
    exit 1
fi

if [[ -e "$OUT" && "$OVERWRITE" != "1" ]]; then
    echo "ERROR: Output already exists: $OUT"
    echo "Resubmit with OVERWRITE=1 only if you intentionally want to replace it."
    exit 1
fi

# Own Miniforge on scratch (the CREATE anaconda3 module is not reliably available).
CONDA_BASE="${CONDA_BASE:-/scratch/users/$USER/miniforge3}"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Miniforge not found at $CONDA_BASE (see ARTERIAL_CASE_43_WALKTHROUGH.md, step 3)."
    exit 1
fi
# conda's activate scripts reference unset variables, so relax 'set -u' around it.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ARTERIAL_ROOT/env"
set -u

export arterial_dir="$ARTERIAL_ROOT/arterial/arterial"
export ARTERIAL_MODELS_DIR="$ARTERIAL_ROOT/models"
export ARTERIAL_NUM_THREADS=8

if [[ ! -d "$ARTERIAL_MODELS_DIR/segmentation/$MODE" ]]; then
    echo "ERROR: Model weights not found: $ARTERIAL_MODELS_DIR/segmentation/$MODE"
    echo "Run download_models.sh on a login node first (README_CREATE_ARTERIAL.md, step 5)."
    exit 1
fi

echo "arterial vessel segmentation"
echo "Case:      $CASE_ID"
echo "Run:       $RUN ($MODE)"
echo "Seg only:  $SEG_ONLY"
echo "No slicing (-ns): $NO_SLICING"
echo "Node:      $HOSTNAME"
echo "Job ID:    $SLURM_JOB_ID"
echo "Input CTA: $CTA"

nvidia-smi

python - <<'PY'
import sys
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available in the arterial environment.")
print("GPU:", torch.cuda.get_device_name(0))
PY

SKIP_ARGS=()
if [[ "$SEG_ONLY" == "1" ]]; then
    # arterial has no single 'segmentation only' flag; skip every later stage.
    SKIP_ARGS=(-sce -sb -svl -sfe -sap -sld)
fi
if [[ "$NO_SLICING" == "1" ]]; then
    SKIP_ARGS+=(-ns)
fi

if [[ "$OVERWRITE" == "1" && -e "$OUT" ]]; then
    rm -f "$OUT"
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
arterial -cd "$CASE_DIR" -cnp "$CTA" -m "$MODE" "${SKIP_ARGS[@]}" $EXTRA_ARGS

if [[ ! -s "$OUT" ]]; then
    echo "ERROR: arterial finished but no segmentation was written: $OUT"
    exit 1
fi

# Confirm the mask is binary and on the input CTA grid.
python - "$OUT" "$CTA" <<'PY'
import sys
import numpy as np
import SimpleITK as sitk

mask = sitk.ReadImage(sys.argv[1])
cta = sitk.ReadImage(sys.argv[2])

values = np.unique(sitk.GetArrayViewFromImage(mask))
print("Mask values:", values.tolist())
print("Vessel voxels:", int((sitk.GetArrayViewFromImage(mask) > 0).sum()))

same = True
for name in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
    a, b = getattr(mask, name)(), getattr(cta, name)()
    ok = np.allclose(a, b, atol=1e-4)
    same &= ok
    print(f"{name[3:]:9s} {'OK' if ok else 'DIFFERS'}  mask={a}  cta={b}")

if not same:
    print("NOTE: mask is not on the input CTA grid. If only Size/Origin differ, it is arterial's head crop;")
    print("      compare_arterial_topbrain.py pastes it back onto the CTA grid automatically.")
PY

echo
echo "arterial segmentation completed successfully."
echo "Output: $OUT"
md5sum "$OUT"
