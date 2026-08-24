"""Create a direct CT-subtraction comparison for every participant.

For voxels inside the registered brain mask:

    output = cropped CTA - registered CT

Outside the brain mask, the cropped CTA is retained unchanged.  No best-fit
line, slope, intercept, or intensity enhancement is applied to the CT.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# Folder holding one sub-folder per participant: pass --results-root or set
# this environment variable. Kept out of the code so no local paths are committed.
RESULTS_ROOT_ENV = "SUBTRACTION_RESULTS_ROOT"

PARTICIPANT_PATTERN = "MTCT_KCH_*"
CTA_FILENAME = "4_cropped_CTA.nii.gz"
CT_FILENAME = "3_registered_CT.nii.gz"
MASK_FILENAME = "7_registered_brain_mask.nii.gz"
OUTPUT_FILENAME = "comparison_CTA_minus_registered_CT_brain_only.nii.gz"


def same_geometry(image: sitk.Image, reference: sitk.Image) -> bool:
    """Return True when two images occupy the same voxel grid."""
    return (
        image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing())
        and np.allclose(image.GetOrigin(), reference.GetOrigin())
        and np.allclose(image.GetDirection(), reference.GetDirection())
    )


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float,
) -> sitk.Image:
    """Resample an image onto the cropped CTA grid if required."""
    if same_geometry(image, reference):
        return image

    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        image.GetPixelID(),
    )


def process_participant(participant_dir: Path) -> None:
    cta_path = participant_dir / CTA_FILENAME
    ct_path = participant_dir / CT_FILENAME
    mask_path = participant_dir / MASK_FILENAME
    output_path = participant_dir / OUTPUT_FILENAME

    missing = [path.name for path in (cta_path, ct_path, mask_path) if not path.exists()]
    if missing:
        print(f"SKIPPED {participant_dir.name}: missing {', '.join(missing)}")
        return

    # The cropped CTA is the reference grid and supplies the output metadata.
    cta_image = sitk.ReadImage(str(cta_path), sitk.sitkFloat32)
    ct_image = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    mask_image = sitk.ReadImage(str(mask_path))

    ct_image = resample_to_reference(
        ct_image,
        cta_image,
        interpolator=sitk.sitkLinear,
        default_value=-1000.0,
    )
    mask_image = resample_to_reference(
        mask_image,
        cta_image,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
    )

    cta = sitk.GetArrayFromImage(cta_image).astype(np.float32, copy=False)
    registered_ct = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
    brain = sitk.GetArrayFromImage(mask_image) > 0.5

    if not np.any(brain):
        print(f"SKIPPED {participant_dir.name}: registered brain mask is empty")
        return

    # This is CTA - brainmask(registered CT):
    #   inside brain  -> CTA - CT
    #   outside brain -> CTA - 0 = CTA
    residual = cta.copy()
    residual[brain] = cta[brain] - registered_ct[brain]

    output_image = sitk.GetImageFromArray(residual.astype(np.float32))
    output_image.CopyInformation(cta_image)
    sitk.WriteImage(output_image, str(output_path), useCompression=True)

    inside = residual[brain]
    print(
        f"DONE {participant_dir.name}\n"
        f"  brain voxels: {int(brain.sum()):,}\n"
        f"  residual inside brain: min={inside.min():.2f}, "
        f"mean={inside.mean():.2f}, max={inside.max():.2f}\n"
        f"  saved: {output_path}"
    )


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
            f"Results folder not found:\n{results_root}\n"
            "Pass the correct folder with --results-root."
        )

    participant_dirs = sorted(
        path
        for path in results_root.glob(pattern)
        if path.is_dir()
    )
    if not participant_dirs:
        raise FileNotFoundError(
            f"No participant folders matching {pattern!r} in {results_root}"
        )

    print(f"Found {len(participant_dirs)} participant folder(s).")
    for participant_dir in participant_dirs:
        try:
            process_participant(participant_dir)
        except Exception as error:
            print(f"ERROR {participant_dir.name}: {error}")


if __name__ == "__main__":
    main()