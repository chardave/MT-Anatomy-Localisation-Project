"""Clean the thresholded brain vessel signal using multiscale objectness.

For every participant in ``bone_subtraction_results`` this script reads:

    8_vessel_signal_brain_threshold20.nii.gz

The input already contains the positive vessel residual (>20 HU) inside the
brain mask and zero elsewhere. The script calculates a multiscale 3D
objectness response configured for bright, tube-like structures, then uses
that response to select the most vessel-like voxels from the original signal.

Outputs per participant:

    13_vessel_objectness_multiscale_normalised.nii.gz
        Continuous objectness response, robustly normalised to [0, 1].

    14_vessel_objectness_mask.nii.gz
        Binary mask after objectness thresholding and optional small-component
        removal.

    15_vessel_signal_objectness_cleaned.nii.gz
        Original vessel-signal HU values where the binary mask is one, and
        zero elsewhere. This is the most useful output for later addition to
        the registered CT.

The script does not overwrite any existing residual or enhanced CT volumes.
"""

from __future__ import annotations

import argparse
import os
import gc
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# =============================================================================
# SETTINGS
# =============================================================================

# Folder holding one sub-folder per participant: pass --results-root or set
# this environment variable. Kept out of the code so no local paths are committed.
RESULTS_ROOT_ENV = "SUBTRACTION_RESULTS_ROOT"

PARTICIPANT_PATTERN = "MTCT_KCH_*"

INPUT_FILENAME = "8_vessel_signal_brain_threshold20.nii.gz"
OBJECTNESS_OUTPUT_FILENAME = (
    "13_vessel_objectness_multiscale_normalised.nii.gz"
)
MASK_OUTPUT_FILENAME = "14_vessel_objectness_mask.nii.gz"
CLEANED_SIGNAL_OUTPUT_FILENAME = (
    "15_vessel_signal_objectness_cleaned.nii.gz"
)

# Gaussian scales in physical units (millimetres). Each scale detects a
# different approximate vessel width. Add a larger value if the largest ICA
# segments are poorly retained; remove the smallest scales if noise dominates.
SIGMAS_MM = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

# Objectness parameters. ObjectDimension=1 below means lines/tubes in 3D.
# These defaults are a sensible initial experiment; change one parameter at a
# time when comparing results.
ALPHA = 0.5
BETA = 0.5
GAMMA = 5.0
BRIGHT_OBJECT = True
SCALE_OBJECTNESS_MEASURE = True

# Objectness is robustly normalised per participant. A value at the stated
# percentile becomes 1.0, and larger responses are clipped to 1.0. This avoids
# one extremely large response controlling the entire normalisation.
NORMALISATION_PERCENTILE = 99.5

# Keep input voxels whose normalised multiscale objectness is at least this
# value. Lower it to retain more weak/small vessels; raise it to remove more
# residual tissue. Useful comparisons: 0.05, 0.10, 0.20, and 0.30.
OBJECTNESS_THRESHOLD = 0.10

# Remove disconnected components smaller than this physical volume. Set to
# 0.0 to disable component removal. Keep this conservative because distal
# vessels can become disconnected after thresholding.
MIN_COMPONENT_VOLUME_MM3 = 1.0

# When False, participants whose three outputs already exist are skipped.
OVERWRITE = True


def validate_settings() -> None:
    """Raise a clear error for invalid user-editable settings."""
    if not SIGMAS_MM or any(sigma <= 0 for sigma in SIGMAS_MM):
        raise ValueError("Every value in SIGMAS_MM must be greater than zero.")
    if not 0 < NORMALISATION_PERCENTILE <= 100:
        raise ValueError(
            "NORMALISATION_PERCENTILE must be greater than 0 and at most 100."
        )
    if not 0 <= OBJECTNESS_THRESHOLD <= 1:
        raise ValueError("OBJECTNESS_THRESHOLD must be between 0 and 1.")
    if MIN_COMPONENT_VOLUME_MM3 < 0:
        raise ValueError("MIN_COMPONENT_VOLUME_MM3 cannot be negative.")


def create_objectness_filter() -> sitk.ObjectnessMeasureImageFilter:
    """Configure SimpleITK objectness for bright tubular structures in 3D."""
    objectness = sitk.ObjectnessMeasureImageFilter()
    objectness.SetObjectDimension(1)
    objectness.SetBrightObject(BRIGHT_OBJECT)
    objectness.SetScaleObjectnessMeasure(SCALE_OBJECTNESS_MEASURE)
    objectness.SetAlpha(ALPHA)
    objectness.SetBeta(BETA)
    objectness.SetGamma(GAMMA)
    return objectness


def calculate_multiscale_objectness(signal: sitk.Image) -> sitk.Image:
    """Return the maximum objectness response across all Gaussian scales."""
    maximum_response: sitk.Image | None = None

    for index, sigma in enumerate(SIGMAS_MM, start=1):
        print(
            f"    Scale {index}/{len(SIGMAS_MM)}: sigma={sigma:g} mm",
            flush=True,
        )

        # SmoothingRecursiveGaussian uses the image's physical spacing, so
        # sigma is interpreted in millimetres rather than voxels.
        smoothed = sitk.SmoothingRecursiveGaussian(signal, sigma)
        response = create_objectness_filter().Execute(smoothed)
        response = sitk.Cast(response, sitk.sitkFloat32)

        if maximum_response is None:
            maximum_response = response
        else:
            maximum_response = sitk.Maximum(maximum_response, response)

        del smoothed, response
        gc.collect()

    if maximum_response is None:
        raise RuntimeError("No objectness response was calculated.")
    maximum_response.CopyInformation(signal)
    return maximum_response


def remove_small_components(
    binary_mask: sitk.Image,
    minimum_volume_mm3: float,
) -> tuple[sitk.Image, int, int]:
    """Remove 3D connected components below a physical-volume threshold."""
    connected_filter = sitk.ConnectedComponentImageFilter()
    connected_filter.SetFullyConnected(True)
    components = connected_filter.Execute(binary_mask)
    components_before = connected_filter.GetObjectCount()

    if minimum_volume_mm3 <= 0:
        return sitk.Cast(binary_mask > 0, sitk.sitkUInt8), components_before, 0

    voxel_volume_mm3 = float(np.prod(binary_mask.GetSpacing()))
    minimum_voxels = max(
        1,
        int(math.ceil(minimum_volume_mm3 / voxel_volume_mm3)),
    )

    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(minimum_voxels)
    relabelled = relabel_filter.Execute(components)
    cleaned_mask = sitk.Cast(relabelled > 0, sitk.sitkUInt8)

    removed_components = int(
        components_before - relabel_filter.GetNumberOfObjects()
    )
    return cleaned_mask, int(components_before), removed_components


def save_array_like(
    array: np.ndarray,
    reference: sitk.Image,
    output_path: Path,
    pixel_type: int,
) -> None:
    """Save a NumPy array using the input vessel signal's physical geometry."""
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    image = sitk.Cast(image, pixel_type)
    sitk.WriteImage(image, str(output_path), useCompression=True)


def process_participant(participant_dir: Path) -> bool:
    """Apply multiscale objectness to one participant's vessel signal."""
    input_path = participant_dir / INPUT_FILENAME
    objectness_path = participant_dir / OBJECTNESS_OUTPUT_FILENAME
    mask_path = participant_dir / MASK_OUTPUT_FILENAME
    cleaned_signal_path = participant_dir / CLEANED_SIGNAL_OUTPUT_FILENAME
    outputs = (objectness_path, mask_path, cleaned_signal_path)

    if not input_path.is_file():
        print(f"[SKIP] {participant_dir.name}: missing {INPUT_FILENAME}")
        return False

    if not OVERWRITE and all(path.is_file() for path in outputs):
        print(f"[SKIP] {participant_dir.name}: outputs already exist")
        return True

    print(f"\n[PROCESSING] {participant_dir.name}")
    signal_image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    if signal_image.GetDimension() != 3:
        raise ValueError(
            f"Objectness requires a 3D image; got "
            f"{signal_image.GetDimension()}D."
        )

    signal_array = sitk.GetArrayFromImage(signal_image).astype(
        np.float32,
        copy=False,
    )
    candidate_mask = np.isfinite(signal_array) & (signal_array > 0)
    candidate_count = int(np.count_nonzero(candidate_mask))
    if candidate_count == 0:
        raise ValueError("The input vessel-signal volume has no positive voxels.")

    # Replace any non-finite values before filtering. Negative values are also
    # zeroed because the input is intended to contain positive enhancement.
    safe_signal_array = np.where(candidate_mask, signal_array, 0).astype(
        np.float32,
        copy=False,
    )
    safe_signal_image = sitk.GetImageFromArray(safe_signal_array)
    safe_signal_image.CopyInformation(signal_image)

    raw_objectness = calculate_multiscale_objectness(safe_signal_image)
    raw_objectness_array = sitk.GetArrayFromImage(raw_objectness).astype(
        np.float32,
        copy=False,
    )

    # Normalise using only positive objectness values at locations that were
    # already part of the thresholded vessel signal.
    usable_responses = raw_objectness_array[
        candidate_mask
        & np.isfinite(raw_objectness_array)
        & (raw_objectness_array > 0)
    ]
    if usable_responses.size == 0:
        raise ValueError(
            "Objectness produced no positive response inside the input signal."
        )

    normalisation_value = float(
        np.percentile(usable_responses, NORMALISATION_PERCENTILE)
    )
    if not np.isfinite(normalisation_value) or normalisation_value <= 0:
        raise ValueError(
            f"Invalid objectness normalisation value: {normalisation_value}"
        )

    normalised_objectness = np.clip(
        raw_objectness_array / normalisation_value,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

    initial_mask_array = (
        candidate_mask
        & (normalised_objectness >= OBJECTNESS_THRESHOLD)
    )
    initial_mask_image = sitk.GetImageFromArray(
        initial_mask_array.astype(np.uint8)
    )
    initial_mask_image.CopyInformation(signal_image)

    cleaned_mask_image, component_count, removed_component_count = (
        remove_small_components(
            initial_mask_image,
            MIN_COMPONENT_VOLUME_MM3,
        )
    )
    cleaned_mask_array = sitk.GetArrayFromImage(cleaned_mask_image) > 0

    # Preserve the original vessel-signal HU values. Objectness is used only
    # to decide which voxels are kept.
    cleaned_signal_array = np.zeros_like(signal_array, dtype=np.float32)
    cleaned_signal_array[cleaned_mask_array] = signal_array[cleaned_mask_array]

    save_array_like(
        normalised_objectness,
        signal_image,
        objectness_path,
        sitk.sitkFloat32,
    )
    save_array_like(
        cleaned_mask_array.astype(np.uint8),
        signal_image,
        mask_path,
        sitk.sitkUInt8,
    )
    save_array_like(
        cleaned_signal_array,
        signal_image,
        cleaned_signal_path,
        sitk.sitkFloat32,
    )

    initial_count = int(np.count_nonzero(initial_mask_array))
    final_count = int(np.count_nonzero(cleaned_mask_array))
    retained_percentage = 100.0 * final_count / candidate_count

    print(f"  Input positive voxels:       {candidate_count:,}")
    print(f"  After objectness threshold:  {initial_count:,}")
    print(f"  After component removal:     {final_count:,}")
    print(f"  Input signal retained:       {retained_percentage:.2f}%")
    print(f"  Connected components:        {component_count:,}")
    print(f"  Small components removed:    {removed_component_count:,}")
    print(
        f"  Normalisation ({NORMALISATION_PERCENTILE:g}th percentile): "
        f"{normalisation_value:.6g}"
    )
    print(f"  Saved: {objectness_path.name}")
    print(f"  Saved: {mask_path.name}")
    print(f"  Saved: {cleaned_signal_path.name}")

    del (
        signal_array,
        safe_signal_array,
        raw_objectness_array,
        normalised_objectness,
        cleaned_signal_array,
    )
    gc.collect()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_root = os.environ.get(RESULTS_ROOT_ENV)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help=f"folder with one sub-folder per participant (or set {RESULTS_ROOT_ENV})",
    )
    parser.add_argument(
        "--pattern",
        default=PARTICIPANT_PATTERN,
        help="glob for participant folder names (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    pattern = args.pattern
    validate_settings()

    if not results_root.is_dir():
        raise NotADirectoryError(
            "Results root does not exist:\n"
            f"  {results_root}\n"
            "Pass the correct folder with --results-root."
        )

    participant_dirs = sorted(
        path
        for path in results_root.glob(pattern)
        if path.is_dir()
    )
    if not participant_dirs:
        raise FileNotFoundError(
            f"No participant folders matching {pattern!r} in:\n"
            f"  {results_root}"
        )

    print(f"Found {len(participant_dirs)} participant folder(s).")
    print(f"Input: {INPUT_FILENAME}")
    print(f"Scales (mm): {SIGMAS_MM}")
    print(f"Normalised objectness threshold: {OBJECTNESS_THRESHOLD:g}")
    print(
        "Minimum connected-component volume: "
        f"{MIN_COMPONENT_VOLUME_MM3:g} mm^3"
    )

    successful = 0
    for participant_dir in participant_dirs:
        try:
            successful += int(process_participant(participant_dir))
        except Exception as error:
            print(f"[ERROR] {participant_dir.name}: {error}")

    print("\n[DONE]")
    print(
        f"Successfully processed {successful}/{len(participant_dirs)} "
        "participants."
    )


if __name__ == "__main__":
    main()
