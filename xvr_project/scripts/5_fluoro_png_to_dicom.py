"""Wrap a masked fluoroscopy snapshot in a DICOM file that xvr can register.

xvr only reads X-rays from DICOM, because it takes the C-arm geometry from the
DICOM tags. This script writes those tags from the values on the angiography
system's status panel (the *_details.png screenshot):

    angles   e.g. "RAO 90 / 0"   -> --primary -90 --secondary 0
             LAO is positive, RAO negative; CRAN positive, CAUD negative
    zoom     e.g. "32 cm"        -> --fov-cm 32 (Siemens quotes the detector DIAGONAL)
    SID      e.g. "108 cm"       -> --sid-cm 108 (source-to-image distance)

The source-to-isocentre distance is not on the panel. It is fixed by the C-arm
design; --sod-mm (default 785) is only the starting guess, xvr refines it.
No patient information is written to the file.

Example:
    python 5_fluoro_png_to_dicom.py --image ..\\input\\fluoro_43\\fluoro_AP_non_contrast_masked.png ^
        --primary 0 --secondary 0 --fov-cm 32 --sid-cm 108
"""

import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import XRayAngiographicImageStorage, ExplicitVRLittleEndian, generate_uid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Masked PNG from 4_mask_fluoro_overlays.py")
    parser.add_argument("--primary", required=True, type=float, help="LAO(+)/RAO(-) angle in degrees")
    parser.add_argument("--secondary", required=True, type=float, help="CRAN(+)/CAUD(-) angle in degrees")
    parser.add_argument("--fov-cm", required=True, type=float, help="Zoom / field size from the panel (cm)")
    parser.add_argument("--fov-type", choices=["diagonal", "width"], default="diagonal",
                        help="Whether --fov-cm is the field diagonal (Siemens convention) or its width")
    parser.add_argument("--sid-cm", required=True, type=float, help="Source-to-image distance from the panel (cm)")
    parser.add_argument("--sod-mm", type=float, default=785.0,
                        help="Source-to-isocentre distance (mm); starting guess only")
    parser.add_argument("--output", type=Path, help="Default: same name as --image with .dcm")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    rows, cols = gray.shape

    # ---- 1. detector pixel size from the field size ----
    fov_mm = args.fov_cm * 10
    if args.fov_type == "diagonal":
        pixel_mm = fov_mm / np.hypot(rows, cols)
    else:
        pixel_mm = fov_mm / cols
    sid_mm = args.sid_cm * 10
    if not 0 < args.sod_mm < sid_mm:
        parser.error("--sod-mm must be positive and smaller than the SID")

    # ---- 2. build a minimal, anonymous X-ray angiography DICOM ----
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    output = args.output or args.image.with_suffix(".dcm")
    ds = FileDataset(str(output), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "XA"
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "ANONYMOUS"
    ds.ContentDate = datetime.date.today().strftime("%Y%m%d")
    ds.ImageComments = f"Converted from {args.image.name}"

    ds.DistanceSourceToDetector = round(sid_mm, 3)
    ds.DistanceSourceToPatient = round(args.sod_mm, 3)
    ds.ImagerPixelSpacing = [round(pixel_mm, 6), round(pixel_mm, 6)]
    ds.PixelSpacing = ds.ImagerPixelSpacing
    ds.PositionerPrimaryAngle = args.primary
    ds.PositionerSecondaryAngle = args.secondary

    # 16-bit, bright = more X-rays reached the detector (air white), as on the screen
    pixels = (gray.astype(np.uint16) * 257)
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()

    ds.save_as(str(output), enforce_file_format=True)
    print(f"Saved: {output}")
    print(f"Image {cols} x {rows} px | pixel size {pixel_mm:.4f} mm | field width {pixel_mm * cols:.1f} mm")
    print(f"SID {sid_mm:.0f} mm | SOD guess {args.sod_mm:.0f} mm | "
          f"angles primary {args.primary:+.1f}, secondary {args.secondary:+.1f}")


if __name__ == "__main__":
    main()
