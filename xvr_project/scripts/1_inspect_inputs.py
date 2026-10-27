
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"


def inspect_nifti(name):
    path = INPUT / name

    if not path.exists():
        print(f"\n[NOT FOUND] {path}")
        return

    nii = nib.load(str(path))

    print(f"\n{'=' * 50}")
    print(f"FILE: {name}")
    print(f"{'=' * 50}")

    print("Shape:", nii.shape)
    print("Voxel spacing (mm):", nii.header.get_zooms()[:3])
    print("Orientation:", nib.aff2axcodes(nii.affine))
    print("Affine matrix:")
    print(np.array2string(nii.affine, precision=4))

    # Read data for sanity checking only
    data = np.asarray(nii.dataobj)

    print("Intensity range:",
          float(np.nanmin(data)),
          float(np.nanmax(data)))

    print("Image centre in world coordinates (mm):")
    centre = (np.array(nii.shape[:3]) - 1) / 2
    print(nib.affines.apply_affine(nii.affine, centre))


def inspect_fluoroscopy():
    path = INPUT / "fluoroscopy_AP.png"

    print(f"\n{'=' * 50}")
    print("FLUOROSCOPY")
    print(f"{'=' * 50}")

    if not path.exists():
        print(f"[NOT FOUND] {path}")
        return

    with Image.open(path) as img:
        print("Image dimensions:", img.size)
        print("Image mode:", img.mode)

        arr = np.asarray(img.convert("L"))
        print("Intensity range:", arr.min(), arr.max())
        print("Aspect ratio:", img.width / img.height)


def check_gpu():
    print(f"\n{'=' * 50}")
    print("GPU")
    print(f"{'=' * 50}")

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:",
              torch.cuda.get_device_capability(0))

        x = torch.randn((256, 256), device="cuda")
        y = x @ x

        print("GPU computation successful:", bool(torch.isfinite(y).all().item()))


if __name__ == "__main__":
    inspect_nifti("CT.nii.gz")
    inspect_nifti("CTA.nii.gz")
    inspect_fluoroscopy()
    check_gpu()

    print("\nInspection complete.")
