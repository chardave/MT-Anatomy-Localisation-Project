#!/usr/bin/env python3
r"""Rigidly align a CTA to a fixed supine CTA template using brain masks.

The first patient may be declared as the template with ``--template-case``.
That mode keeps its CTA and mask unchanged while creating the standard Step 3
outputs and an identity transform. For later patients, the script performs a
six-degree-of-freedom rigid registration and writes every result on the fixed
template grid.

The template must already represent the desired head-first, supine orientation.
This script aligns later anatomy to that convention; it does not decide whether
the selected template itself is anatomically level.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk


ALIGNED_CTA_NAME = "7_Aligned_CTA.nii.gz"
ALIGNED_MASK_NAME = "8_Aligned_CTA_brain_mask.nii.gz"


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")
    return path


def geometry(image: sitk.Image) -> dict[str, object]:
    return {
        "size": tuple(int(value) for value in image.GetSize()),
        "spacing": tuple(float(value) for value in image.GetSpacing()),
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
    }


def print_geometry(name: str, image: sitk.Image) -> None:
    values = geometry(image)
    print(f"{name}")
    print(f"  size:      {values['size']}")
    print(f"  spacing:   {values['spacing']} mm")
    print(f"  origin:    {values['origin']}")
    print(f"  direction: {values['direction']}")


def assert_same_grid(
    image: sitk.Image,
    mask: sitk.Image,
    image_name: str,
    mask_name: str,
) -> None:
    """Require a mask to already occupy the exact grid of its source CTA."""
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError(f"{image_name} and {mask_name} must both be 3-D.")

    if image.GetSize() != mask.GetSize():
        raise ValueError(
            f"{image_name} and {mask_name} have different sizes: "
            f"{image.GetSize()} versus {mask.GetSize()}."
        )

    checks = (
        ("spacing", image.GetSpacing(), mask.GetSpacing(), 1e-5),
        ("origin", image.GetOrigin(), mask.GetOrigin(), 1e-4),
        ("direction", image.GetDirection(), mask.GetDirection(), 1e-6),
    )
    for field, image_values, mask_values, tolerance in checks:
        if not np.allclose(
            np.asarray(image_values, dtype=np.float64),
            np.asarray(mask_values, dtype=np.float64),
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                f"{image_name} and {mask_name} have different {field}. "
                "The brain mask must be on the CTA grid before registration."
            )


def make_binary_mask(mask: sitk.Image, name: str) -> sitk.Image:
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    count = int(np.count_nonzero(sitk.GetArrayViewFromImage(binary)))
    if count == 0:
        raise ValueError(f"{name} is empty.")
    print(f"{name} foreground voxels: {count:,}")
    return binary


def resample_mask(
    moving_mask: sitk.Image,
    fixed_reference: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    return sitk.Resample(
        moving_mask,
        fixed_reference,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def dice_score(first: sitk.Image, second: sitk.Image) -> float:
    a = sitk.GetArrayViewFromImage(first) > 0
    b = sitk.GetArrayViewFromImage(second) > 0
    denominator = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if denominator == 0:
        return 1.0
    intersection = int(np.count_nonzero(a & b))
    return 2.0 * intersection / denominator


def check_destinations(destinations: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "The following outputs already exist:\n"
            f"{listing}\nUse --overwrite to replace them."
        )


def write_template_case(
    *,
    moving_cta_path: Path,
    moving_mask_path: Path,
    output_cta_path: Path,
    output_mask_path: Path,
) -> None:
    """Create standard outputs for the patient used as the fixed template."""
    shutil.copy2(moving_cta_path, output_cta_path)
    shutil.copy2(moving_mask_path, output_mask_path)

    print("\nTemplate mode: no registration was performed.")
    print("The CTA and mask were copied unchanged to the Step 3 output names.")
    print("The template case uses identity alignment.")


def register_to_template(
    *,
    fixed_cta: sitk.Image,
    fixed_mask: sitk.Image,
    moving_cta: sitk.Image,
    moving_mask: sitk.Image,
    sampling_percentage: float,
    iterations: int,
) -> tuple[sitk.Image, sitk.Image, sitk.Transform, float, float]:
    """Perform brain-masked rigid CTA-to-CTA registration."""
    fixed_float = sitk.Cast(fixed_cta, sitk.sitkFloat32)
    moving_float = sitk.Cast(moving_cta, sitk.sitkFloat32)

    # Initialise translation from the centres of mass of the two brain masks.
    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32),
        sitk.Cast(moving_mask, sitk.sitkFloat32),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    initial_mask = resample_mask(moving_mask, fixed_cta, initial_transform)
    initial_dice = dice_score(fixed_mask, initial_mask)
    print(f"Initial brain-mask Dice: {initial_dice:.4f}")

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    print("\nRunning rigid registration...")
    final_transform = registration.Execute(fixed_float, moving_float)
    print("Registration finished.")
    print(f"Final metric value: {registration.GetMetricValue():.6g}")
    print(f"Optimizer stop: {registration.GetOptimizerStopConditionDescription()}")

    aligned_cta = sitk.Resample(
        moving_float,
        fixed_cta,
        final_transform,
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )
    aligned_mask = resample_mask(moving_mask, fixed_cta, final_transform)
    final_dice = dice_score(fixed_mask, aligned_mask)
    print(f"Final brain-mask Dice: {final_dice:.4f}")

    return aligned_cta, aligned_mask, final_transform, initial_dice, final_dice


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rigidly align a CTA to a fixed supine CTA template using matching "
            "brain masks."
        )
    )
    parser.add_argument(
        "--moving-cta",
        required=True,
        type=Path,
        help="Patient CTA to align (normally 4_CTA_LPS.nii.gz)",
    )
    parser.add_argument(
        "--moving-mask",
        required=True,
        type=Path,
        help="Brain mask on the moving CTA grid",
    )
    parser.add_argument(
        "--template-cta",
        type=Path,
        help="Fixed CTA from the first/template patient",
    )
    parser.add_argument(
        "--template-mask",
        type=Path,
        help="Brain mask on the fixed template CTA grid",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Patient folder for the standard Step 3 outputs",
    )
    parser.add_argument(
        "--template-case",
        action="store_true",
        help=(
            "Declare the moving CTA as the fixed template; copy it unchanged "
            "to the standard output names and save an identity transform"
        ),
    )
    parser.add_argument(
        "--sampling-percentage",
        type=float,
        default=0.20,
        help="Fraction of eligible voxels used for the metric (default: 0.20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Maximum optimizer iterations per resolution process (default: 500)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Step 3 outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.sampling_percentage <= 1.0:
        raise ValueError("--sampling-percentage must be greater than 0 and at most 1.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if not args.template_case and (args.template_cta is None or args.template_mask is None):
        raise ValueError(
            "Later patients require both --template-cta and --template-mask. "
            "Use --template-case only for the first patient selected as the template."
        )

    moving_cta_path = require_file(args.moving_cta, "Moving CTA")
    moving_mask_path = require_file(args.moving_mask, "Moving brain mask")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_cta_path = args.output_dir / ALIGNED_CTA_NAME
    output_mask_path = args.output_dir / ALIGNED_MASK_NAME
    destinations = (output_cta_path, output_mask_path)
    check_destinations(destinations, args.overwrite)

    print("CTA-to-template rigid alignment")
    print(f"Moving CTA:  {moving_cta_path}")
    print(f"Moving mask: {moving_mask_path}")

    moving_cta = sitk.ReadImage(str(moving_cta_path))
    moving_mask_original = sitk.ReadImage(str(moving_mask_path))
    assert_same_grid(moving_cta, moving_mask_original, "Moving CTA", "Moving mask")
    moving_mask = make_binary_mask(moving_mask_original, "Moving mask")
    print_geometry("Moving CTA geometry", moving_cta)

    if args.template_case:
        write_template_case(
            moving_cta_path=moving_cta_path,
            moving_mask_path=moving_mask_path,
            output_cta_path=output_cta_path,
            output_mask_path=output_mask_path,
        )
    else:
        template_cta_path = require_file(args.template_cta, "Template CTA")
        template_mask_path = require_file(args.template_mask, "Template brain mask")
        print(f"Template CTA:  {template_cta_path}")
        print(f"Template mask: {template_mask_path}")

        fixed_cta = sitk.ReadImage(str(template_cta_path))
        fixed_mask_original = sitk.ReadImage(str(template_mask_path))
        assert_same_grid(fixed_cta, fixed_mask_original, "Template CTA", "Template mask")
        fixed_mask = make_binary_mask(fixed_mask_original, "Template mask")
        print_geometry("Template CTA geometry", fixed_cta)

        aligned_cta, aligned_mask, transform, initial_dice, final_dice = (
            register_to_template(
                fixed_cta=fixed_cta,
                fixed_mask=fixed_mask,
                moving_cta=moving_cta,
                moving_mask=moving_mask,
                sampling_percentage=args.sampling_percentage,
                iterations=args.iterations,
            )
        )

        sitk.WriteImage(aligned_cta, str(output_cta_path), useCompression=True)
        sitk.WriteImage(aligned_mask, str(output_mask_path), useCompression=True)
        sitk.WriteTransform(
            transform,
            str(args.output_dir / "CTA_to_template_transform.tfm")
        )

        if final_dice < initial_dice:
            print(
                "WARNING: final brain-mask Dice is lower than the initial Dice. "
                "Inspect the registration carefully before continuing."
            )
        if final_dice < 0.80:
            print(
                "WARNING: final brain-mask Dice is below 0.80. This does not "
                "automatically mean failure, but visual review is required."
            )

    # Confirm the written outputs exist and share one grid.
    saved_cta = sitk.ReadImage(str(output_cta_path))
    saved_mask = sitk.ReadImage(str(output_mask_path))
    assert_same_grid(saved_cta, saved_mask, "Aligned CTA", "Aligned brain mask")

    print("\nCompleted successfully.")
    print(f"Aligned CTA:        {output_cta_path}")
    print(f"Aligned brain mask: {output_mask_path}")
    print("Inspect the aligned CTA and mask against the template before Step 4.")


if __name__ == "__main__":
    main()
