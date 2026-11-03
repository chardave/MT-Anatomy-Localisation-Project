#!/bin/bash -l

#SBATCH --job-name=topbrain-kdh
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/users/%u/topbrain/topbrain-kdh-%j.out

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Required: the case folder name under $TOPBRAIN_ROOT/cases, e.g.
# sbatch --export=ALL,CASE_ID=<case-folder-name> run_topbrain_kdh_ready_cta.slurm
CASE_ID="${CASE_ID:?Set CASE_ID: sbatch --export=ALL,CASE_ID=<case-folder-name> run_topbrain_kdh_ready_cta.slurm}"
OVERWRITE="${OVERWRITE:-0}"

# Override with --export=ALL,CASE_ID=...,TOPBRAIN_ROOT=<path> if your setup lives elsewhere.
TOPBRAIN_ROOT="${TOPBRAIN_ROOT:-/scratch/users/$USER/topbrain}"

# KDH compatibility setup previously installed on CREATE.
SIF="$TOPBRAIN_ROOT/containers/topbrain_kdh_ct.sif"
PACKAGES="$TOPBRAIN_ROOT/kdh_python_packages"
TORCH_PACKAGES="$TOPBRAIN_ROOT/kdh_torch_280_cu128"
CONVERTER_PYTHON="$TOPBRAIN_ROOT/convert_env/bin/python"

# Per-case input and final output.
CASE_ROOT="$TOPBRAIN_ROOT/cases/$CASE_ID"
CTA="$CASE_ROOT/input/10_ready_CTA.nii.gz"
FINAL_OUTPUT_DIR="$CASE_ROOT/output"
FINAL_PREDICTION="$FINAL_OUTPUT_DIR/13_CTA_vessel_labels.nii.gz"

# Writable working directory required by the KDH container.
JOB_WORK="$TOPBRAIN_ROOT/work_kdh_ready/$CASE_ID/$SLURM_JOB_ID"
CONTAINER_INPUT="$JOB_WORK/test/input/images/head-ct-angio"
CONTAINER_OUTPUT="$JOB_WORK/test/output"
PREDICTION_MHA="$CONTAINER_OUTPUT/images/head-ct-angio-segmentation/output.mha"
MODEL_LOG="$JOB_WORK/kdh_inference.log"

for required_file in "$SIF" "$CTA" "$CONVERTER_PYTHON"; do
    if [[ ! -e "$required_file" ]]; then
        echo "ERROR: Required file does not exist: $required_file"
        exit 1
    fi
done

for required_dir in "$PACKAGES" "$TORCH_PACKAGES"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "ERROR: Required KDH package directory does not exist: $required_dir"
        exit 1
    fi
done

if [[ -e "$FINAL_PREDICTION" && "$OVERWRITE" != "1" ]]; then
    echo "ERROR: Final output already exists: $FINAL_PREDICTION"
    echo "Resubmit with OVERWRITE=1 only if you intentionally want to replace it."
    exit 1
fi

mkdir -p "$CONTAINER_INPUT" "$CONTAINER_OUTPUT" "$FINAL_OUTPUT_DIR"
cp "$CTA" "$CONTAINER_INPUT/10_ready_CTA.nii.gz"

echo "TopBrain KDH vessel segmentation"
echo "Case:       $CASE_ID"
echo "Node:       $HOSTNAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Input CTA:  $CTA"
echo "Container:  $SIF"
echo "Final file: $FINAL_PREDICTION"

nvidia-smi

SINGULARITY_COMMAND=(
    singularity exec
    --nv
    --cleanenv
    --pwd /work
    --bind "$JOB_WORK:/work"
    --bind "$PACKAGES:/opt/kdh_packages:ro"
    --bind "$TORCH_PACKAGES:/opt/kdh_torch_cu128:ro"
    --env PYTHONPATH=/opt/kdh_torch_cu128:/opt/kdh_packages
    --env PATH=/opt/kdh_torch_cu128/bin:/opt/kdh_packages/bin:/usr/local/bin:/usr/bin:/bin
    "$SIF"
)

check_cuda() {
    "${SINGULARITY_COMMAND[@]}" python -c '
import sys
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0))
'
}

echo "Checking patched KDH PyTorch environment..."
if ! check_cuda; then
    # On some nodes Singularity cannot list the host NVIDIA driver libraries
    # ("could not retrieve ld cache: /usr/sbin/ldconfig.real"), so --nv binds
    # nothing. Copy the driver libraries into the job folder and put them on the
    # container's library path instead.
    echo "CUDA not visible through --nv; retrying with the host NVIDIA libraries bound explicitly..."
    HOST_NV="$JOB_WORK/host_nvidia_libs"
    mkdir -p "$HOST_NV"
    for dir in /usr/lib64 /usr/lib/x86_64-linux-gnu /lib64 /lib/x86_64-linux-gnu /usr/lib64/nvidia /usr/lib/nvidia; do
        [[ -d "$dir" ]] || continue
        find "$dir" -maxdepth 1 \( -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' \
            -o -name 'libnvidia-ptxjitcompiler.so*' -o -name 'libnvidia-nvvm.so*' \
            -o -name 'libnvidia-gpucomp.so*' \) -exec cp -L -n {} "$HOST_NV/" \; 2>/dev/null || true
    done
    if ! ls "$HOST_NV"/libcuda.so* >/dev/null 2>&1; then
        echo "ERROR: no libcuda.so found on $HOSTNAME; report the node to CREATE support."
        exit 1
    fi
    echo "Host NVIDIA libraries: $(ls "$HOST_NV" | tr '\n' ' ')"

    # same command, with the copied libraries bound in before the image path
    last=$((${#SINGULARITY_COMMAND[@]} - 1))
    SINGULARITY_COMMAND=(
        "${SINGULARITY_COMMAND[@]:0:$last}"
        --bind "$HOST_NV:/opt/host_nvidia:ro"
        --env LD_LIBRARY_PATH=/opt/host_nvidia
        "$SIF"
    )
    if ! check_cuda; then
        echo "ERROR: CUDA is not available in the patched KDH environment, even with the host libraries."
        exit 1
    fi
fi

echo "Starting KDH inference..."
setsid "${SINGULARITY_COMMAND[@]}" python /opt/app/inference.py \
    >"$MODEL_LOG" 2>&1 &
MODEL_PID=$!

echo "Inference PID: $MODEL_PID"
echo "Inference log: $MODEL_LOG"

# KDH may remain alive after writing a complete prediction. Monitor the output
# and stop the process group only after the file has remained stable.
previous_size=-1
stable_checks=0
watchdog_triggered=0

while true; do
    sleep 30

    if [[ -s "$PREDICTION_MHA" ]]; then
        current_size=$(stat -c %s "$PREDICTION_MHA")
        if [[ "$current_size" -eq "$previous_size" ]]; then
            stable_checks=$((stable_checks + 1))
        else
            previous_size="$current_size"
            stable_checks=0
        fi

        if grep -q "Done!" "$MODEL_LOG" 2>/dev/null && [[ "$stable_checks" -ge 2 ]]; then
            echo "Prediction is stable and the model reported Done."
            watchdog_triggered=1
            break
        fi

        if [[ "$stable_checks" -ge 6 ]]; then
            echo "Prediction has remained stable for at least three minutes."
            watchdog_triggered=1
            break
        fi
    fi

    if ! kill -0 "$MODEL_PID" 2>/dev/null; then
        set +e
        wait "$MODEL_PID"
        model_status=$?
        set -e

        if [[ ! -s "$PREDICTION_MHA" ]]; then
            echo "ERROR: KDH exited with status $model_status without a prediction."
            tail -n 100 "$MODEL_LOG" || true
            exit 1
        fi

        echo "KDH exited normally after creating the prediction."
        break
    fi

    echo "Waiting: output_size=${previous_size}, stable_checks=${stable_checks}"
    tail -n 5 "$MODEL_LOG" || true
done

if [[ "$watchdog_triggered" -eq 1 ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
    echo "Stopping the completed KDH process group..."
    kill -TERM -- "-$MODEL_PID" 2>/dev/null || true

    for _ in $(seq 1 24); do
        if ! kill -0 "$MODEL_PID" 2>/dev/null; then
            break
        fi
        sleep 5
    done

    if kill -0 "$MODEL_PID" 2>/dev/null; then
        echo "KDH did not stop after 120 seconds; sending SIGKILL."
        kill -KILL -- "-$MODEL_PID" 2>/dev/null || true
    fi

    set +e
    wait "$MODEL_PID"
    set -e
fi

if [[ ! -s "$PREDICTION_MHA" ]]; then
    echo "ERROR: Prediction is absent or empty: $PREDICTION_MHA"
    exit 1
fi

# Confirm that the output is no longer being written before conversion.
size_before=$(stat -c %s "$PREDICTION_MHA")
sleep 10
size_after=$(stat -c %s "$PREDICTION_MHA")
if [[ "$size_before" -ne "$size_after" ]]; then
    echo "ERROR: Prediction size changed during final validation."
    exit 1
fi

echo "Converting KDH output.mha to an integer NIfTI label map..."
# convert_env stops working on nodes whose system Python differs from the one it was
# built with (its packages are then not found). Fall back to the container's Python,
# which only sees the job folder (mounted at /work), so it reads the CTA copy made for
# the container and writes into /work; the result is copied out afterwards.
CONVERT_IN_CONTAINER=0
if "$CONVERTER_PYTHON" -c "import numpy, SimpleITK" >/dev/null 2>&1; then
    CONVERT=("$CONVERTER_PYTHON" - "$PREDICTION_MHA" "$CTA" "$FINAL_PREDICTION")
elif "${SINGULARITY_COMMAND[@]}" python -c "import numpy, SimpleITK" >/dev/null 2>&1; then
    echo "convert_env cannot import numpy/SimpleITK on $HOSTNAME; converting with the container's Python."
    CONVERT_IN_CONTAINER=1
    CONVERT=("${SINGULARITY_COMMAND[@]}" python -
             "/work/${PREDICTION_MHA#"$JOB_WORK"/}"
             "/work/${CONTAINER_INPUT#"$JOB_WORK"/}/10_ready_CTA.nii.gz"
             "/work/$(basename "$FINAL_PREDICTION")")
else
    echo "ERROR: neither convert_env nor the container can import numpy and SimpleITK."
    echo "The prediction is kept at: $PREDICTION_MHA"
    echo "Recreate convert_env on a compute node, or convert that file elsewhere."
    exit 1
fi
"${CONVERT[@]}" <<'PY'
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

prediction_path = Path(sys.argv[1])
reference_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

prediction = sitk.ReadImage(str(prediction_path))
reference = sitk.ReadImage(str(reference_path))

if prediction.GetDimension() != 3:
    raise RuntimeError(
        f"Expected a 3D prediction, got {prediction.GetDimension()} dimensions."
    )

if prediction.GetSize() != reference.GetSize():
    raise RuntimeError(
        "Prediction and input CTA sizes differ: "
        f"{prediction.GetSize()} versus {reference.GetSize()}."
    )

values = sitk.GetArrayFromImage(prediction)
if not np.all(np.isfinite(values)):
    raise RuntimeError("Prediction contains non-finite values.")

rounded = np.rint(values)
if not np.allclose(values, rounded, atol=1e-4, rtol=0.0):
    raise RuntimeError("Prediction contains non-integer label values.")

minimum = int(rounded.min())
maximum = int(rounded.max())
if minimum < 0:
    raise RuntimeError(f"Prediction contains negative label IDs: {minimum}.")

if maximum <= np.iinfo(np.uint8).max:
    prediction = sitk.Cast(prediction, sitk.sitkUInt8)
elif maximum <= np.iinfo(np.uint16).max:
    prediction = sitk.Cast(prediction, sitk.sitkUInt16)
else:
    prediction = sitk.Cast(prediction, sitk.sitkUInt32)

# The KDH prediction is returned on the input index grid. Copy the exact input
# geometry without resampling or interpolating the labels.
prediction.CopyInformation(reference)

output_path.parent.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(prediction, str(output_path), True)

saved = sitk.ReadImage(str(output_path))
if saved.GetSize() != reference.GetSize():
    raise RuntimeError("Saved NIfTI failed the final size check.")

print("Saved:", output_path)
print("Size:", saved.GetSize())
print("Spacing:", saved.GetSpacing())
print("Label range:", minimum, "to", maximum)
PY

if [[ "$CONVERT_IN_CONTAINER" == "1" ]]; then
    cp "$JOB_WORK/$(basename "$FINAL_PREDICTION")" "$FINAL_PREDICTION"
fi

echo
echo "KDH segmentation completed successfully."
echo "Final NIfTI: $FINAL_PREDICTION"
echo "Working files: $JOB_WORK"
md5sum "$FINAL_PREDICTION"
