"""Create vessel-enhanced registered CT volumes for every KCH participant.

For each participant, the script calculates:

    vessel_signal = vessel_residual where brain_mask > 0 and residual > 20 HU
                    0 elsewhere

    enhanced_CT = registered_CT + strength * vessel_signal

The registered CT is preserved everywhere outside the selected vessel signal.
If the registered CT or brain mask is not already on the residual image grid,
it is resampled using physical image coordinates before the calculation.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# -----------------------------------------------------------------------------
# Settings you may want to change
# -----------------------------------------------------------------------------

# Folder holding one sub-folder per participant: pass --results-root or set
# this environment variable. Kept out of the code so no local paths are committed.
RESULTS_ROOT_ENV = "SUBTRACTION_RESULTS_ROOT"

PARTICIPANT_PATTERN = "MTCT_KCH_*"

REGISTERED_CT_NAME = "3_registered_CT.nii.gz"
VESSEL_RESIDUAL_NAME = "6_vessel_residual.nii.gz"
BRAIN_MASK_NAME = "7_registered_brain_mask.nii.gz"

RESIDUAL_THRESHOLD_HU = 20.0
ENHANCEMENT_STRENGTHS = (1.0, 2.0, 100.0)

# Set to a value such as 99.5 if very bright isolated artefacts dominate the
# result. Leave as None to keep every retained residual at its original value.
CAP_VESSEL_SIGNAL_PERCENTILE = None

VESSEL_SIGNAL_OUTPUT_NAME = "8_vessel_signal_brain_threshold20.nii.gz"


def same_geometry(image_a: sitk.Image, image_b: sitk.Image) -> bool:
    """Return True when two images have the same voxel and physical geometry."""
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=1e-6)
        and np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=1e-4)
        and np.allclose(image_a.GetDirection(), image_b.GetDirection(), atol=1e-6)
    )


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float,
) -> sitk.Image:
    """Resample an image onto a reference grid using an identity transform."""
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        sitk.sitkFloat32,
    )


def strength_label(strength: float) -> str:
    """Create a filename-friendly label such as '1' or '1p5'."""
    if float(strength).is_integer():
        return str(int(strength))
    return str(strength).replace(".", "p")


def save_array_like_reference(
    array: np.ndarray,
    reference: sitk.Image,
    output_path: Path,
) -> None:
    """Save a float32 NumPy array with the reference image geometry."""
    output_image = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    output_image.CopyInformation(reference)
    sitk.WriteImage(output_image, str(output_path), useCompression=True)


def process_participant(participant_dir: Path) -> bool:
    """Create vessel signal and enhanced CT files for one participant."""
    ct_path = participant_dir / REGISTERED_CT_NAME
    residual_path = participant_dir / VESSEL_RESIDUAL_NAME
    mask_path = participant_dir / BRAIN_MASK_NAME

    required_paths = (ct_path, residual_path, mask_path)
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        print(f"[SKIP] {participant_dir.name}")
        for path in missing_paths:
            print(f"       Missing: {path.name}")
        return False

    print(f"\n[PROCESSING] {participant_dir.name}")

    registered_ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    vessel_residual = sitk.ReadImage(str(residual_path), sitk.sitkFloat32)
    brain_mask = sitk.ReadImage(str(mask_path))

    # The residual is used as the output/reference grid because it is the
    # volume being thresholded and is normally already on the cropped CTA grid.
    reference = vessel_residual

    if not same_geometry(registered_ct, reference):
        print("  Registered CT grid differs from residual grid: resampling CT.")
        registered_ct = resample_to_reference(
            registered_ct,
            reference,
            interpolator=sitk.sitkLinear,
            default_value=-1000.0,
        )

    if not same_geometry(brain_mask, reference):
        print("  Brain-mask grid differs from residual grid: resampling mask.")
        brain_mask = resample_to_reference(
            brain_mask,
            reference,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0.0,
        )

    ct_array = sitk.GetArrayFromImage(registered_ct).astype(np.float32)
    residual_array = sitk.GetArrayFromImage(vessel_residual).astype(np.float32)
    mask_array = sitk.GetArrayFromImage(brain_mask)

    if not (
        ct_array.shape == residual_array.shape == mask_array.shape
    ):
        raise RuntimeError(
            f"Array shape mismatch after resampling for {participant_dir.name}: "
            f"CT={ct_array.shape}, residual={residual_array.shape}, "
            f"mask={mask_array.shape}"
        )

    brain = mask_array > 0.5
    selected_vessels = (
        brain
        & np.isfinite(residual_array)
        & (residual_array > RESIDUAL_THRESHOLD_HU)
    )

    # Keep the original residual intensity after thresholding. In particular,
    # this does not subtract 20 HU from every selected voxel.
    vessel_signal = np.zeros_like(residual_array, dtype=np.float32)
    vessel_signal[selected_vessels] = residual_array[selected_vessels]

    if CAP_VESSEL_SIGNAL_PERCENTILE is not None and np.any(selected_vessels):
        cap = float(
            np.percentile(
                vessel_signal[selected_vessels],
                CAP_VESSEL_SIGNAL_PERCENTILE,
            )
        )
        vessel_signal = np.minimum(vessel_signal, cap)
        print(
            f"  Vessel signal capped at percentile "
            f"{CAP_VESSEL_SIGNAL_PERCENTILE}: {cap:.2f} HU"
        )

    vessel_signal_path = participant_dir / VESSEL_SIGNAL_OUTPUT_NAME
    save_array_like_reference(vessel_signal, reference, vessel_signal_path)

    selected_count = int(np.count_nonzero(selected_vessels))
    brain_count = int(np.count_nonzero(brain))
    percentage = 100.0 * selected_count / brain_count if brain_count else 0.0

    print(f"  Brain voxels:          {brain_count:,}")
    print(f"  Selected vessel voxels:{selected_count:>12,} ({percentage:.3f}%)")

    if selected_count:
        retained_values = vessel_signal[selected_vessels]
        print(
            "  Retained residual HU:  "
            f"min={retained_values.min():.2f}, "
            f"mean={retained_values.mean():.2f}, "
            f"max={retained_values.max():.2f}"
        )
    else:
        print("  [WARNING] No residual voxels passed the brain-mask threshold.")

    print(f"  Saved: {vessel_signal_path.name}")

    for output_number, strength in enumerate(ENHANCEMENT_STRENGTHS, start=9):
        enhanced_ct = ct_array + float(strength) * vessel_signal
        label = strength_label(strength)
        output_path = participant_dir / (
            f"{output_number}_registered_CT_vessel_enhanced_x{label}.nii.gz"
        )
        save_array_like_reference(enhanced_ct, reference, output_path)
        print(f"  Saved: {output_path.name}")

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
    if not results_root.exists():
        raise FileNotFoundError(
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
            f"No participant folders matching '{pattern}' were "
            f"found in:\n  {results_root}"
        )

    print(f"Found {len(participant_dirs)} participant folder(s).")
    print(f"Residual threshold: > {RESIDUAL_THRESHOLD_HU:g} HU")
    print(f"Enhancement strengths: {ENHANCEMENT_STRENGTHS}")

    successful = 0
    for participant_dir in participant_dirs:
        try:
            successful += int(process_participant(participant_dir))
        except Exception as error:
            print(f"[ERROR] {participant_dir.name}: {error}")

    print("\n[DONE]")
    print(f"Successfully processed {successful}/{len(participant_dirs)} participants.")


if __name__ == "__main__":
    main()
