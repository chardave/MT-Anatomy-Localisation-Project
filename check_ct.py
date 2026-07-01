import numpy as np
import nibabel as nib
from pathlib import Path

in_path = Path(
    r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\Totalsegmentator_dataset_v201\s0224\ct.nii.gz"
)

out_path = in_path.with_name("ct_fixed_orthogonal.nii.gz")

# Load with nibabel, which is usually more permissive than SimpleITK/ITK
img = nib.load(str(in_path))

data = np.asanyarray(img.dataobj)
affine = img.affine.copy()

print("Original shape:", data.shape)
print("Original affine:\n", affine)

# Extract the 3x3 voxel-to-world orientation/spacing part
A = affine[:3, :3]

# Voxel spacing = length of each column vector
spacing = np.linalg.norm(A, axis=0)

# Direction matrix = columns normalised by spacing
direction = A / spacing

print("Original spacing:", spacing)
print("Direction dot-products:\n", direction.T @ direction)

# Find the closest valid orthonormal rotation matrix using SVD
U, _, Vt = np.linalg.svd(direction)
direction_ortho = U @ Vt

# Preserve handedness
if np.linalg.det(direction_ortho) * np.linalg.det(direction) < 0:
    U[:, -1] *= -1
    direction_ortho = U @ Vt

new_affine = affine.copy()
new_affine[:3, :3] = direction_ortho @ np.diag(spacing)

print("Fixed affine:\n", new_affine)
print("Fixed direction dot-products:\n", direction_ortho.T @ direction_ortho)

# Save a repaired NIfTI
fixed_img = nib.Nifti1Image(data, new_affine)
fixed_img.set_qform(new_affine, code=1)
fixed_img.set_sform(new_affine, code=1)

nib.save(fixed_img, str(out_path))

print("Saved fixed CT to:", out_path)