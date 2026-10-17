#!/usr/bin/env python3
"""Step 4/5: CT -> original CTA (Elastix) -> template CTA (saved Step 3 transform).

The Step 3 SimpleITK transform MUST be the transform returned by the CTA-to-template
registration and used in sitk.Resample(moving_CTA, template_CTA, transform, ...).
It maps template/output physical points to original CTA/input physical points.

Inputs: original-space LPS CT/CTA and their brain masks; aligned CTA/mask from Step 3;
Step 3 .tfm; permanent template mask and desired crop-size reference CT.
Outputs: four cropped, grid-matched ready volumes plus Elastix parameters.
"""
from __future__ import annotations

import argparse
import faulthandler
faulthandler.enable()
import shutil
import tempfile
from pathlib import Path

import itk
import numpy as np
import SimpleITK as sitk

OUTPUT_NAMES = {
    'ct': '9_ready_CT.nii.gz',
    'cta': '10_ready_CTA.nii.gz',
    'ct_mask': '11_ready_CT_brain_mask.nii.gz',
    'cta_mask': '12_ready_CTA_brain_mask.nii.gz',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct', type=Path, required=True, help='3_CT_LPS.nii.gz')
    parser.add_argument('--ct-mask', type=Path, required=True, help='5_CT_LPS_brain_mask.nii.gz')
    parser.add_argument('--original-cta', type=Path, required=True, help='4_CTA_LPS.nii.gz (Elastix fixed)')
    parser.add_argument('--original-cta-mask', type=Path, required=True, help='6_CTA_LPS_brain_mask.nii.gz')
    parser.add_argument('--aligned-cta', type=Path, required=True, help='7_Aligned_CTA.nii.gz (template grid)')
    parser.add_argument('--aligned-cta-mask', type=Path, required=True, help='8_Aligned_CTA_brain_mask.nii.gz')
    parser.add_argument('--cta-to-template-transform', type=Path, required=True, help='Saved SimpleITK Step 3 transform .tfm')
    parser.add_argument('--crop-reference-mask', type=Path, required=True, help='Permanent template patient brain mask on aligned CTA grid')
    parser.add_argument('--crop-size-reference-ct', type=Path, required=True, help='Reference CT whose voxel dimensions define the crop size')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--initialization', choices=['header', 'center-of-gravity', 'geometrical-center'], default='header')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def require(path, label):
    if not path.is_file():
        raise FileNotFoundError(f'{label}: {path}')
    return path


def same_grid(a, b):
    return (a.GetDimension() == b.GetDimension() == 3 and
            a.GetSize() == b.GetSize() and
            np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5, rtol=0) and
            np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4, rtol=0) and
            np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6, rtol=0))


def require_grid(a, b, label_a, label_b):
    if not same_grid(a, b):
        raise ValueError(f'{label_a} and {label_b} must occupy the same grid (size/spacing/origin/direction).')


def binary_mask(mask):
    result = sitk.Cast(mask > 0, sitk.sitkUInt8)
    if not np.any(sitk.GetArrayViewFromImage(result)):
        raise ValueError('Brain mask is empty')
    return result


def crop_region(template_mask, target_size):
    """Find a bounded fixed crop without temporary-image dangling views.

    NumPy uses ZYX index order; SimpleITK GetSize uses XYZ.
    Hold the image and its array copy for the entire calculation.
    """
    stable_mask = binary_mask(template_mask)
    voxels = sitk.GetArrayFromImage(stable_mask) > 0  # owning copy
    if not voxels.any():
        raise ValueError('Template crop-reference brain mask is empty')

    # Determine bounding coordinates without allocating an N x 3
    # array for millions of foreground voxels.
    z = np.flatnonzero(np.any(voxels, axis=(1, 2)))
    y = np.flatnonzero(np.any(voxels, axis=(0, 2)))
    x = np.flatnonzero(np.any(voxels, axis=(0, 1)))
    lo = np.array([x[0], y[0], z[0]], dtype=np.int64)
    hi = np.array([x[-1], y[-1], z[-1]], dtype=np.int64)
    del voxels

    size = np.asarray(target_size, dtype=np.int64)
    bounds = np.asarray(template_mask.GetSize(), dtype=np.int64)
    if size.shape != (3,) or np.any(size <= 0) or np.any(size > bounds):
        raise ValueError(f'Requested crop {tuple(size)} exceeds template grid {tuple(bounds)}')
    center = (lo.astype(float) + hi.astype(float)) / 2.0
    start = np.floor(center - (size - 1) / 2.0).astype(np.int64)
    start = np.minimum(np.maximum(start, 0), bounds - size)
    return start.tolist(), size.tolist()


def elastix_parameters(initialization):
    p = itk.ParameterObject.GetDefaultParameterMap('rigid')
    if initialization == 'header':
        p['AutomaticTransformInitialization'] = ['false']
    else:
        p['AutomaticTransformInitialization'] = ['true']
        p['AutomaticTransformInitializationMethod'] = [
            'CenterOfGravity' if initialization == 'center-of-gravity' else 'GeometricalCenter'
        ]
    p['NumberOfResolutions'] = ['4']
    p['MaximumNumberOfIterations'] = ['750']
    p['NumberOfSpatialSamples'] = ['16000']
    p['NumberOfHistogramBins'] = ['64']
    p['ImageSampler'] = ['RandomSparseMask']
    p['MaximumNumberOfSamplingAttempts'] = ['0']
    p['FinalBSplineInterpolationOrder'] = ['1']
    p['DefaultPixelValue'] = ['-1024']
    p['ResultImagePixelType'] = ['float']
    obj = itk.ParameterObject.New()
    obj.SetParameterMap(p)
    return obj


def mask_parameters(fitted):
    m = dict(fitted.GetParameterMap(0))
    m['FinalBSplineInterpolationOrder'] = ['0']
    m['DefaultPixelValue'] = ['0']
    m['ResultImagePixelType'] = ['unsigned char']
    obj = itk.ParameterObject.New()
    obj.SetParameterMap(m)
    return obj


def main():
    print('[DEBUG] Main started', flush=True)
    args = parse_args()
    print('[DEBUG] Arguments parsed', flush=True)
    files = {
        'CT': args.ct, 'CT mask': args.ct_mask,
        'Original CTA': args.original_cta, 'Original CTA mask': args.original_cta_mask,
        'Aligned CTA': args.aligned_cta, 'Aligned CTA mask': args.aligned_cta_mask,
        'Step 3 transform': args.cta_to_template_transform,
        'Template crop mask': args.crop_reference_mask,
        'Crop size reference CT': args.crop_size_reference_ct,
    }
    for label, path in files.items():
        print(f'[DEBUG] Checking {label}: {path}', flush=True)
        require(path, label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {key: args.output_dir / name for key, name in OUTPUT_NAMES.items()}
    elastix_file = args.output_dir / 'CT_to_original_CTA_ElastixTransformParameters.txt'
    conflicts = [p for p in [*outputs.values(), elastix_file] if p.exists()]
    if conflicts and not args.overwrite:
        raise FileExistsError('Outputs exist; use --overwrite to replace:\n' + '\n'.join(map(str, conflicts)))

    print('[DEBUG] Loading CT...', flush=True)
    ct = sitk.ReadImage(str(args.ct))
    print('[DEBUG] Loaded CT', flush=True)
    print('[DEBUG] Loading CT mask...', flush=True)
    ct_mask = binary_mask(sitk.ReadImage(str(args.ct_mask)))
    print('[DEBUG] Loaded CT mask', flush=True)
    print('[DEBUG] Loading Original CTA...', flush=True)
    original_cta = sitk.ReadImage(str(args.original_cta))
    print('[DEBUG] Loaded Original CTA', flush=True)
    print('[DEBUG] Loading Original CTA mask...', flush=True)
    original_mask = binary_mask(sitk.ReadImage(str(args.original_cta_mask)))
    print('[DEBUG] Loaded Original CTA mask', flush=True)
    print('[DEBUG] Loading Aligned CTA...', flush=True)
    aligned_cta = sitk.ReadImage(str(args.aligned_cta))
    print('[DEBUG] Loaded Aligned CTA', flush=True)
    print('[DEBUG] Loading Aligned CTA mask...', flush=True)
    aligned_mask = binary_mask(sitk.ReadImage(str(args.aligned_cta_mask)))
    print('[DEBUG] Loaded Aligned CTA mask', flush=True)
    print('[DEBUG] Loading Template mask...', flush=True)
    reference_mask = sitk.ReadImage(str(args.crop_reference_mask))
    print('[DEBUG] Loaded Template mask', flush=True)
    print('[DEBUG] Loading Template CT...', flush=True)
    reference_ct = sitk.ReadImage(str(args.crop_size_reference_ct))
    print('[DEBUG] Loaded Template CT', flush=True)

    print('[DEBUG] Checking grids...', flush=True)
    require_grid(ct, ct_mask, 'CT', 'CT brain mask')
    require_grid(original_cta, original_mask, 'Original CTA', 'Original CTA brain mask')
    require_grid(aligned_cta, aligned_mask, 'Aligned CTA', 'Aligned CTA brain mask')
    require_grid(aligned_cta, reference_mask, 'Aligned CTA', 'Template crop mask')
    if reference_ct.GetDimension() != 3:
        raise ValueError('Crop-size reference CT must be 3D')
    print('[DEBUG] Calculating crop...', flush=True)
    crop_start, crop_size = crop_region(reference_mask, reference_ct.GetSize())
    print('[DEBUG] Loading saved transform...', flush=True)
    transform = sitk.ReadTransform(str(args.cta_to_template_transform))
    if transform.GetDimension() != 3:
        raise ValueError('Step 3 transform must be 3D')

    # Validate that the saved transform reproduces Step 3's aligned brain mask.
    print('[DEBUG] Validating transform by resampling mask...', flush=True)
    remapped_mask = sitk.Resample(original_mask, aligned_cta, transform,
                                 sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    print('[DEBUG] Calculating validation Dice...', flush=True)
    a = sitk.GetArrayViewFromImage(remapped_mask) > 0
    b = sitk.GetArrayViewFromImage(aligned_mask) > 0
    denom = int(a.sum()) + int(b.sum())
    dice = 2.0 * np.count_nonzero(a & b) / denom if denom else 0.0
    print(f'Step 3 transform validation (brain-mask Dice): {dice:.5f}', flush=True)
    if dice < 0.95:
        raise ValueError('The saved Step 3 transform does not reproduce the aligned mask. '
                         'Verify transform direction, patient and Step 3 version before continuing.')

    print('Stage A: Register CT to ORIGINAL patient CTA with Elastix...', flush=True)
    with tempfile.TemporaryDirectory(prefix='ct_to_original_cta_') as tmp:
        reg_ct, fitted = itk.elastix_registration_method(
            itk.imread(str(args.original_cta), itk.F),
            itk.imread(str(args.ct), itk.F),
            fixed_mask=itk.imread(str(args.original_cta_mask), itk.UC),
            moving_mask=itk.imread(str(args.ct_mask), itk.UC),
            parameter_object=elastix_parameters(args.initialization),
            output_directory=tmp,
            log_to_console=True,
            log_to_file=False,
        )
        reg_mask = itk.transformix_filter(
            itk.imread(str(args.ct_mask), itk.UC),
            mask_parameters(fitted),
            output_directory=tmp,
        )
        # ITK and SimpleITK share physical coordinates through NIfTI I/O.
        ct_tmp = Path(tmp) / 'registered_CT.nii.gz'
        mask_tmp = Path(tmp) / 'registered_CT_mask.nii.gz'
        itk.imwrite(reg_ct, str(ct_tmp), compression=True)
        itk.imwrite(reg_mask, str(mask_tmp), compression=True)
        registered_ct = sitk.ReadImage(str(ct_tmp))
        registered_mask = binary_mask(sitk.ReadImage(str(mask_tmp)))
        require_grid(original_cta, registered_ct, 'Original CTA', 'Elastix CT result')
        require_grid(original_cta, registered_mask, 'Original CTA', 'Elastix CT mask result')
        params = Path(tmp) / 'TransformParameters.0.txt'
        if not params.is_file():
            raise FileNotFoundError(f'Elastix transform was not written: {params}')
        shutil.copy2(params, elastix_file)

    print('Stage B: Resample registered CT and CT mask into Step 3 template grid...', flush=True)
    ct_on_template = sitk.Resample(registered_ct, aligned_cta, transform,
                                  sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    mask_on_template = sitk.Resample(registered_mask, aligned_cta, transform,
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    full = {
        'ct': ct_on_template,
        'cta': sitk.Cast(aligned_cta, sitk.sitkFloat32),
        'ct_mask': mask_on_template,
        'cta_mask': aligned_mask,
    }
    print(f'Fixed crop index XYZ: {tuple(crop_start)}; size XYZ: {tuple(crop_size)}')
    for key, img in full.items():
        cropped = sitk.RegionOfInterest(img, size=crop_size, index=crop_start)
        sitk.WriteImage(cropped, str(outputs[key]), useCompression=True)

    saved = {key: sitk.ReadImage(str(path)) for key, path in outputs.items()}
    for key, img in saved.items():
        require_grid(saved['cta'], img, 'Saved ready CTA', f'Saved {key}')
    print('Done. All four outputs share the same grid:')
    for path in outputs.values():
        print(f'  {path}')
    print(f'  {elastix_file}')
    print('Inspect CT/CTA overlays and both masks before downstream segmentation.')


if __name__ == '__main__':
    main()
