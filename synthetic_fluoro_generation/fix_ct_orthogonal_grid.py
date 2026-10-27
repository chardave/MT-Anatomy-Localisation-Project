#!/usr/bin/env python3
"""Resample a tilted/sheared CT and its brain mask onto one orthogonal grid."""
import argparse
from pathlib import Path
from itertools import product
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct', type=Path, required=True)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()

    ct = nib.load(str(args.ct))
    mask = nib.load(str(args.mask))
    if len(ct.shape) != 3 or len(mask.shape) != 3:
        raise ValueError('CT and mask must both be 3D')
    if ct.shape != mask.shape or not np.allclose(ct.affine, mask.affine, atol=1e-3):
        raise ValueError('CT and mask must have the same shape and affine BEFORE correction')

    original_affine = ct.affine
    axes = original_affine[:3, :3]
    voxel_spacing = np.linalg.norm(axes, axis=0)
    directions = axes / voxel_spacing
    orthogonality_error = np.max(np.abs(directions.T @ directions - np.eye(3)))
    print('Original CT orientation:', nib.aff2axcodes(original_affine))
    print('Original shape:', ct.shape)
    print('Original voxel spacing (mm):', voxel_spacing)
    print('Maximum orthogonality error:', orthogonality_error)

    # Polar decomposition: nearest orthonormal basis; preserve handedness.
    u, _, vt = np.linalg.svd(directions)
    rotation = u @ vt
    new_axes = rotation @ np.diag(voxel_spacing)

    # Find an orthogonal output grid covering the complete original CT FOV.
    corners = np.array(list(product(*[(-0.5, n - 0.5) for n in ct.shape])), dtype=float)
    world = (original_affine[:3, :3] @ corners.T).T + original_affine[:3, 3]
    coords = (np.linalg.inv(new_axes) @ world.T).T
    low = np.floor(coords.min(axis=0) + 0.5).astype(int)
    high = np.ceil(coords.max(axis=0) + 0.5).astype(int)
    new_shape = tuple((high - low).tolist())
    new_affine = np.eye(4)
    new_affine[:3, :3] = new_axes
    new_affine[:3, 3] = new_axes @ low
    target = (new_shape, new_affine)

    print('Corrected shape:', new_shape)
    print('Corrected orientation:', nib.aff2axcodes(new_affine))
    print('Resampling CT (linear interpolation)...')
    ct_out = resample_from_to(ct, target, order=1, mode='constant', cval=-1024)
    print('Resampling mask (nearest-neighbour interpolation)...')
    mask_out = resample_from_to(mask, target, order=0, mode='constant', cval=0)

    ct_result = nib.Nifti1Image(np.asarray(ct_out.dataobj, dtype=np.float32), new_affine)
    mask_result = nib.Nifti1Image((np.asarray(mask_out.dataobj) > 0).astype(np.uint8), new_affine)
    for result in (ct_result, mask_result):
        result.set_sform(new_affine, code=1)
        result.set_qform(new_affine, code=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ct_path = args.output_dir / '3_CT_LPS_orthogonal.nii.gz'
    mask_path = args.output_dir / '5_CT_LPS_brain_mask_orthogonal.nii.gz'
    nib.save(ct_result, str(ct_path))
    nib.save(mask_result, str(mask_path))
    print('Saved:', ct_path)
    print('Saved:', mask_path)
    print('Original files were not modified.')

if __name__ == '__main__':
    main()
