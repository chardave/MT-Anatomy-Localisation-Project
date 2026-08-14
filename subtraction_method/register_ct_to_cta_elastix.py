import argparse
from pathlib import Path
import itk

# paths: one case folder containing the CT series in 3/ and the CTA series in 4/
parser = argparse.ArgumentParser(description="Rigidly register a CT to its CTA with Elastix.")
parser.add_argument("case_dir", type=Path, help="case folder, e.g. <KCH_scans>/<case-id>")
parser.add_argument("--cta", default="4/4_CTA_Neck_and_COW_0.625mm.nii.gz", help="CTA path relative to case_dir")
parser.add_argument("--ct", default="3/3_Brain_0.625mm_Tilt_1.nii.gz", help="CT path relative to case_dir")
parser.add_argument("--cta-mask", default="4/brain.nii.gz", help="CTA brain mask relative to case_dir")
parser.add_argument("--ct-mask", default="3/brain.nii.gz", help="CT brain mask relative to case_dir")
parser.add_argument("--output", type=Path, default=None, help="output folder (default: case_dir/output)")
args = parser.parse_args()

CTA_PATH = args.case_dir / args.cta
CT_PATH = args.case_dir / args.ct

CTA_MASK_PATH = args.case_dir / args.cta_mask
CT_MASK_PATH = args.case_dir / args.ct_mask

OUTPUT_PATH = args.output if args.output is not None else args.case_dir / "output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# read images - need to be in float32 for registration, masks loaded as unsigned 8-bit integer images
print("Reading CTA image...")
cta = itk.imread(CTA_PATH, itk.F)
print("Reading CT image...")
ct = itk.imread(CT_PATH, itk.F)
print("Reading CTA mask...")
cta_mask = itk.imread(CTA_MASK_PATH, itk.UC)
print("Reading CT mask...")
ct_mask = itk.imread(CT_MASK_PATH, itk.UC)

# create rigid parameter map allowing 3 translations, 3 rotations but no scaling or shearing that would deform the anatomy

parameter_map = itk.ParameterObject.GetDefaultParameterMap("rigid")

# physical coordinate headers should already provide approximate initial alignment
parameter_map["AutomaticTransformInitialization"] = ["false"]

# four level coarse to fine registration 
parameter_map["NumberOfResolutions"] = ["4"]

# iterations at each resolution level
parameter_map["MaximumNumberOfIterations"] = ["750"]

# three dimensional mutual information calculation 
parameter_map["NumberOfSpatialSamples"] = ["16000"]

# mutual information histogram resolution
parameter_map["NumberOfHistogramBins"] = ["64"]

# linear interpolation preserves the original image intensities, which is important for quantitative analysis
parameter_map["FinalBSplineInterpolationOrder"] = ["1"]

# treat areas outside the transformed CT as air
parameter_map["DefaultPixelValue"] = ["-1024"]

# keep registered output as floating point
parameter_map["ResultsImagePixelType"] = ["float"]

parameter_object = itk.ParameterObject.New()
parameter_object.SetParameterMap(parameter_map)

# run registration 
# fixed - CTA, moving - CT, output - CT resampled to CTA space

print("\nRunning registration...")
registered_ct, transform_parameters = itk.elastix_registration_method(
    cta,
    ct,
    fixed_mask = cta_mask,
    moving_mask = ct_mask,
    parameter_object = parameter_object,
    output_directory = str(OUTPUT_PATH),
    log_to_console = True,
    log_to_file = True,
    log_file_name = "elastix_log.txt"
)

print("\nRegistration complete. Registered CT saved to output directory.")

# save registered CT image
registered_ct_path = OUTPUT_PATH / "CT_registered_to_CTA.nii.gz"

itk.imwrite(registered_ct,
            str(registered_ct_path),
            compression = True)

print(f"Registration CT mask saved to: \n{registered_ct_path}")
print("\nAll done.")
