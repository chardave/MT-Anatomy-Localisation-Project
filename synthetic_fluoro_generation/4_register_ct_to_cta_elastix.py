#!/usr/bin/env python3
"""Register CT to aligned CTA, standardise the grid, and apply one fixed crop.

The fixed CTA must already be aligned to the common template by Step 3. Elastix
rigidly registers the CT to that CTA. The fitted rigid transform is also applied
to the CT brain mask with nearest-neighbour interpolation. All four volumes are
then placed on the fixed CTA grid. The permanent template brain mask determines
the crop centre, while the template patient's original LPS CT determines the
exact output voxel dimensions.

Only the four ready-to-use NIfTI files are retained. Elastix's internal result,
transform-parameter, and log files are created in a temporary directory.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Sequence

import itk
import numpy as np
import SimpleITK as sitk


OUTPUT_NAMES = {
    "ct": "9_ready_CT.nii.gz",
    "cta": "10_ready_CTA.nii.gz",
    "ct_mask": "11_ready_CT_brain_mask.nii.gz",
    "cta_mask": "12_ready_CTA_brain_mask.nii.gz",
}


def existing_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rigidly register CT to an aligned CTA, transform both brain "
            "masks, place everything on the template grid, and apply one "
            "template-derived crop."
        )
    )
    parser.add_argument(
        "--cta",
        required=True,
        type=Path,
        help="Fixed template-aligned CTA from Step 3",
    )
    parser.add_argument(
        "--ct",
        required=True,
        type=Path,
        help="Moving LPS CT",
    )
    parser.add_argument(
        "--cta-mask",
        required=True,
        type=Path,
        help="Brain mask on the fixed aligned-CTA grid",
    )
    parser.add_argument(
        "--ct-mask",
        required=True,
        type=Path,
        help="Brain mask on the moving CT grid",
    )
    parser.add_argument(
        "--crop-reference-mask",
        required=True,
        type=Path,
        help=(
            "Permanent brain mask from the first/template patient. Use this "
            "same file for every patient."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Patient directory for the four ready output files",
    )
    parser.add_argument(
        "--crop-size-reference-ct",
        required=True,
        type=Path,
        help=(
            "Permanent original LPS CT from the first/template patient. Its "
            "voxel dimensions define the output crop size for every patient."
        ),
    )
    parser.add_argument(
        "--initialization",
        choices=("header", "center-of-gravity", "geometrical-center"),
        default="header",
        help=(
            "Initial alignment. Use 'header' when paired CT/CTA physical "
            "headers already correspond (default)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing ready output files",
    )
    return parser.parse_args(argv)


def same_grid(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetDimension() == second.GetDimension() == 3
        and first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), atol=1e-5, rtol=0.0)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), atol=1e-4, rtol=0.0)
        and np.allclose(
            first.GetDirection(), second.GetDirection(), atol=1e-6, rtol=0.0
        )
    )


def assert_same_grid(
    first: sitk.Image,
    second: sitk.Image,
    first_name: str,
    second_name: str,
) -> None:
    if not same_grid(first, second):
        raise ValueError(
            f"{first_name} and {second_name} do not occupy the same physical "
            "grid. Check their size, spacing, origin, and direction."
        )


def binary_mask(mask: sitk.Image, name: str) -> sitk.Image:
    result = sitk.Cast(mask > 0, sitk.sitkUInt8)
    count = int(np.count_nonzero(sitk.GetArrayViewFromImage(result)))
    if count == 0:
        raise ValueError(f"{name} is empty.")
    return result


def crop_region_from_template_mask(
    template_mask: sitk.Image,
    target_size_xyz: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Centre a fixed-size crop on the template brain-mask bounding box."""
    mask_array = sitk.GetArrayViewFromImage(template_mask) > 0
    coordinates_zyx = np.argwhere(mask_array)
    if coordinates_zyx.size == 0:
        raise ValueError("The crop-reference brain mask is empty.")

    minimum_xyz = coordinates_zyx.min(axis=0)[::-1].astype(np.int64)
    maximum_xyz = (coordinates_zyx.max(axis=0)[::-1] + 1).astype(np.int64)
    image_size_xyz = np.asarray(template_mask.GetSize(), dtype=np.int64)
    target_size_xyz = np.asarray(target_size_xyz, dtype=np.int64)

    if target_size_xyz.shape != (3,) or np.any(target_size_xyz <= 0):
        raise ValueError(f"Invalid crop size: {tuple(target_size_xyz)}")
    if np.any(target_size_xyz > image_size_xyz):
        raise ValueError(
            "The requested original-CT crop size is larger than the template "
            f"grid. Crop size={tuple(target_size_xyz)}; template "
            f"size={tuple(image_size_xyz)}."
        )

    brain_centre_xyz = (minimum_xyz + maximum_xyz - 1) / 2.0
    start_xyz = np.floor(brain_centre_xyz - (target_size_xyz - 1) / 2.0).astype(
        np.int64
    )

    # Shift the crop back inside the image when the centred region would cross
    # a template boundary, while preserving the requested dimensions exactly.
    maximum_start_xyz = image_size_xyz - target_size_xyz
    start_xyz = np.minimum(np.maximum(start_xyz, 0), maximum_start_xyz)

    return start_xyz.astype(int).tolist(), target_size_xyz.astype(int).tolist()


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    is_mask: bool,
) -> sitk.Image:
    if is_mask:
        image = binary_mask(image, "Resampled mask input")
        interpolator = sitk.sitkNearestNeighbor
        default_value = 0.0
        output_type = sitk.sitkUInt8
    else:
        image = sitk.Cast(image, sitk.sitkFloat32)
        interpolator = sitk.sitkLinear
        default_value = -1024.0
        output_type = sitk.sitkFloat32

    identity = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(
        image,
        reference,
        identity,
        interpolator,
        default_value,
        output_type,
    )


def crop_image(image: sitk.Image, index: list[int], size: list[int]) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=size, index=index)


def mask_transform_parameters(
    registration_parameters: itk.ParameterObject,
) -> itk.ParameterObject:
    """Copy the fitted rigid transform and configure label-safe resampling."""
    parameter_map = registration_parameters.GetParameterMap(0)
    parameter_map["FinalBSplineInterpolationOrder"] = ["0"]
    parameter_map["DefaultPixelValue"] = ["0"]
    parameter_map["ResultImagePixelType"] = ["unsigned char"]

    result = itk.ParameterObject.New()
    result.SetParameterMap(parameter_map)
    return result


def check_outputs(output_paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "The following outputs already exist:\n"
            f"{listing}\nUse --overwrite to replace all ready outputs."
        )


def make_registration_parameters(initialization: str) -> itk.ParameterObject:
    parameter_map = itk.ParameterObject.GetDefaultParameterMap("rigid")

    if initialization == "header":
        parameter_map["AutomaticTransformInitialization"] = ["false"]
    else:
        method = {
            "center-of-gravity": "CenterOfGravity",
            "geometrical-center": "GeometricalCenter",
        }[initialization]
        parameter_map["AutomaticTransformInitialization"] = ["true"]
        parameter_map["AutomaticTransformInitializationMethod"] = [method]

    parameter_map["NumberOfResolutions"] = ["4"]
    parameter_map["MaximumNumberOfIterations"] = ["750"]
    parameter_map["NumberOfSpatialSamples"] = ["16000"]
    parameter_map["NumberOfHistogramBins"] = ["64"]
    parameter_map["ImageSampler"] = ["RandomSparseMask"]
    parameter_map["MaximumNumberOfSamplingAttempts"] = ["0"]
    parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
    parameter_map["DefaultPixelValue"] = ["-1024"]
    parameter_map["ResultImagePixelType"] = ["float"]

    parameter_object = itk.ParameterObject.New()
    parameter_object.SetParameterMap(parameter_map)
    return parameter_object


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    cta_path = existing_file(args.cta, "Fixed aligned CTA")
    ct_path = existing_file(args.ct, "Moving CT")
    cta_mask_path = existing_file(args.cta_mask, "Aligned CTA brain mask")
    ct_mask_path = existing_file(args.ct_mask, "CT brain mask")
    crop_reference_path = existing_file(
        args.crop_reference_mask,
        "Permanent template crop-reference mask",
    )
    crop_size_reference_path = existing_file(
        args.crop_size_reference_ct,
        "Permanent template original CT used for crop size",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        key: args.output_dir / filename for key, filename in OUTPUT_NAMES.items()
    }
    check_outputs(output_paths, args.overwrite)

    print("Combined CT registration and fixed-template preparation")
    print(f"Fixed aligned CTA:      {cta_path}")
    print(f"Moving CT:              {ct_path}")
    print(f"Aligned CTA mask:       {cta_mask_path}")
    print(f"Moving CT mask:         {ct_mask_path}")
    print(f"Template crop mask:     {crop_reference_path}")
    print(f"Crop-size reference CT: {crop_size_reference_path}")
    print(f"Registration initialisation: {args.initialization}")

    # Validate source image/mask pairs before starting the expensive step.
    fixed_cta_sitk = sitk.ReadImage(str(cta_path))
    fixed_cta_mask_sitk = sitk.ReadImage(str(cta_mask_path))
    moving_ct_sitk = sitk.ReadImage(str(ct_path))
    moving_ct_mask_sitk = sitk.ReadImage(str(ct_mask_path))
    crop_reference_mask = sitk.ReadImage(str(crop_reference_path))
    crop_size_reference_ct = sitk.ReadImage(str(crop_size_reference_path))
    if crop_size_reference_ct.GetDimension() != 3:
        raise ValueError("The crop-size reference CT must be 3-D.")

    assert_same_grid(
        fixed_cta_sitk,
        fixed_cta_mask_sitk,
        "Aligned CTA",
        "Aligned CTA mask",
    )
    assert_same_grid(moving_ct_sitk, moving_ct_mask_sitk, "CT", "CT mask")
    assert_same_grid(
        fixed_cta_sitk,
        crop_reference_mask,
        "Aligned CTA",
        "Template crop-reference mask",
    )
    fixed_cta_mask_sitk = binary_mask(fixed_cta_mask_sitk, "Aligned CTA mask")
    moving_ct_mask_sitk = binary_mask(moving_ct_mask_sitk, "CT mask")
    crop_reference_mask = binary_mask(crop_reference_mask, "Template crop mask")

    crop_index, crop_size = crop_region_from_template_mask(
        crop_reference_mask,
        crop_size_reference_ct.GetSize(),
    )
    print(f"Fixed crop index (XYZ): {tuple(crop_index)}")
    print(f"Fixed crop size (XYZ):  {tuple(crop_size)}")

    print("\nReading ITK registration inputs...")
    cta_itk = itk.imread(str(cta_path), itk.F)
    ct_itk = itk.imread(str(ct_path), itk.F)
    cta_mask_itk = itk.imread(str(cta_mask_path), itk.UC)
    ct_mask_itk = itk.imread(str(ct_mask_path), itk.UC)
    registration_parameters = make_registration_parameters(args.initialization)

    print("Running rigid CT-to-CTA registration...")
    with tempfile.TemporaryDirectory(prefix="elastix_register_prepare_") as temp_dir:
        registered_ct_itk, fitted_parameters = itk.elastix_registration_method(
            cta_itk,
            ct_itk,
            fixed_mask=cta_mask_itk,
            moving_mask=ct_mask_itk,
            parameter_object=registration_parameters,
            output_directory=temp_dir,
            log_to_console=True,
            log_to_file=False,
        )

        print("Applying the same rigid transform to the CT brain mask...")
        registered_ct_mask_itk = itk.transformix_filter(
            ct_mask_itk,
            mask_transform_parameters(fitted_parameters),
            output_directory=temp_dir,
        )

        # Temporary NIfTIs provide a reliable bridge between ITK-Elastix and
        # SimpleITK and disappear when this context exits.
        temporary_ct = Path(temp_dir) / "registered_ct_for_preparation.nii.gz"
        temporary_mask = Path(temp_dir) / "registered_ct_mask_for_preparation.nii.gz"
        itk.imwrite(registered_ct_itk, str(temporary_ct), compression=True)
        itk.imwrite(registered_ct_mask_itk, str(temporary_mask), compression=True)
        registered_ct_sitk = sitk.ReadImage(str(temporary_ct))
        registered_ct_mask_sitk = sitk.ReadImage(str(temporary_mask))

    print("Placing all volumes on the fixed template grid...")
    full_grid_images = {
        "ct": resample_to_reference(
            registered_ct_sitk,
            fixed_cta_sitk,
            is_mask=False,
        ),
        "cta": resample_to_reference(
            fixed_cta_sitk,
            fixed_cta_sitk,
            is_mask=False,
        ),
        "ct_mask": resample_to_reference(
            registered_ct_mask_sitk,
            fixed_cta_sitk,
            is_mask=True,
        ),
        "cta_mask": resample_to_reference(
            fixed_cta_mask_sitk,
            fixed_cta_sitk,
            is_mask=True,
        ),
    }

    print("Applying the identical template crop...")
    ready_images = {
        key: crop_image(image, crop_index, crop_size)
        for key, image in full_grid_images.items()
    }

    reference_ready = ready_images["cta"]
    for key, image in ready_images.items():
        assert_same_grid(reference_ready, image, "Ready CTA", f"Ready {key}")

    for key, image in ready_images.items():
        sitk.WriteImage(image, str(output_paths[key]), useCompression=True)

    # Reload the final files to validate what was actually written.
    saved = {key: sitk.ReadImage(str(path)) for key, path in output_paths.items()}
    for key, image in saved.items():
        assert_same_grid(saved["cta"], image, "Saved ready CTA", f"Saved ready {key}")

    print("\nCompleted successfully. Final grid:")
    print(f"  size:      {saved['cta'].GetSize()}")
    print(f"  spacing:   {saved['cta'].GetSpacing()} mm")
    print(f"  origin:    {saved['cta'].GetOrigin()}")
    print("Outputs:")
    for key in ("ct", "cta", "ct_mask", "cta_mask"):
        print(f"  {output_paths[key]}")
    print("Inspect all four outputs together before running TOPBrain.")


if __name__ == "__main__":
    main()
