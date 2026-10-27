# Create one static six-panel image

This adds a static projection step after your existing preprocessing. It does not replace registration or TopBrain. It is a standalone simplified PyTorch cone-beam ray integrator, not nanoDRR or DeepDRR. CT attenuation is approximated from HU; vessels are projected as path length in mm. Bright and dark vessel panels are opposite display polarities of the same signal, not separate acquisitions or a true live roadmap. No contrast-flow or catheter simulation is included.

## 1. Finish the existing patient preprocessing

Keep your existing sequence: reorient CT/CTA, obtain brain masks, align CTA to template, register CT to CTA, resample/crop both to the common grid, segment ready CTA with TopBrain, download labels.

Required final inputs in the patient folder:

- 9_ready_CT.nii.gz: registered, resampled CT, still in HU.
- 13_CTA_vessel_labels.nii.gz: integer vessel labels on exactly the same physical grid.

10_ready_CTA.nii.gz is useful for checking segmentation but not required for this rendering step. 14_vessel_enhanced_CT.nii.gz is NOT needed: vessels are projected separately. Use the unenhanced CT for the middle column.

Load CT and labels together in Slicer or ITK-SNAP and verify alignment first. The script checks dimensions and NIfTI affines; it does not repair registration or resample labels silently. Missing arteries in the segmentation will be missing in the output.

## 2. Put the script in your scripts folder

Extract make_biplane_image.py into:

<path-to-repo>\synthetic_fluoro_generation

Open Windows PowerShell:

```powershell
$project = "<path-to-repo>"
$scripts = Join-Path $project "synthetic_fluoro_generation"
$caseId = "<case-id>"
$case = Join-Path (Join-Path $project "synth_fluoro_gen_test") $caseId
cd $scripts
Test-Path "$case\9_ready_CT.nii.gz"
Test-Path "$case\13_CTA_vessel_labels.nii.gz"
```

Both checks must return True. Keep this PowerShell session open so the variables remain defined.

## 3. Create a separate Windows Python environment once

Python 3.11 is already installed from your OCR setup. This new environment keeps rendering dependencies separate:

```powershell
py -3.11 -m venv .render_env
.\.render_env\Scripts\python.exe -m pip install --upgrade pip
.\.render_env\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.render_env\Scripts\python.exe -m pip install numpy nibabel scipy Pillow
```

No Conda, Ubuntu or environment activation is required. Internet is needed for installation. CUDA PyTorch can also run on CPU if GPU access fails.

Check GPU:

```powershell
.\.render_env\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The rendering script also tests an actual GPU interpolation kernel at startup. --device auto uses CPU if this startup test fails. CUDA memory exhaustion later is not silently retried: reduce --batch-rays or use CPU.

## 4. Make a quick 256-pixel trial

```powershell
.\.render_env\Scripts\python.exe .\make_biplane_image.py `
  --ct "$case\9_ready_CT.nii.gz" `
  --labels "$case\13_CTA_vessel_labels.nii.gz" `
  --output-dir "$case\15_static_biplane_trial" `
  --size 256 `
  --step-mm 1.0 `
  --device auto
```

Defaults generate an AP-like view and a view rotated 90 degrees about LPS Z. They deliberately do not claim to reproduce the screenshot's exact RAO direction. The script prints label IDs present and progress through rays.

Open the output:

```powershell
Invoke-Item "$case\15_static_biplane_trial\biplane_six_panel.png"
```

Verify the CT and vessel outlines align, the head is not clipped, and both views show vessels. Projection pixels are square and the canvas does not stretch anatomy.

## 5. Make the full image

```powershell
.\.render_env\Scripts\python.exe .\make_biplane_image.py `
  --ct "$case\9_ready_CT.nii.gz" `
  --labels "$case\13_CTA_vessel_labels.nii.gz" `
  --output-dir "$case\15_static_biplane" `
  --size 512 `
  --step-mm 0.5 `
  --a-angles 0 0 0 `
  --b-angles 90 0 0 `
  --sod-mm 850 `
  --a-sdd-mm 1000 `
  --b-sdd-mm 1170 `
  --a-field-mm 320 `
  --b-field-mm 400 `
  --device auto
```

SDD and field-size defaults are inspired by the supplied display (100/117 cm and 32/40 cm). Source-to-target distance 850 mm is an assumed starting value, not a measured value from the screenshot. The square field model treats the displayed detector input-format size as a square width; this is a first approximation, not a calibration of detector format, zoom, display cropping or collimation.

Open:

```powershell
Invoke-Item "$case\15_static_biplane\biplane_six_panel.png"
```

## 6. Select vessels, if desired

Without --label-ids, every positive label is projected. This is easiest for checking the method, but a selective angiogram normally shows an injected territory. Read the KDH mapping JSON and choose the appropriate territory IDs; do not reuse TopAneu IDs without confirming the KDH mapping. Add, for example:

```powershell
--label-ids 4 5 17
```

These numbers demonstrate syntax only; they are NOT a verified territory selection. The script rejects absent IDs and reports all IDs in your input. Territory selection by anatomical labels is still an approximation: collateral filling and flow are not modelled.

## 7. Adjust appearance or framing

- Reduce --noise to 0 for clean images; default .008 adds independent Gaussian display noise, not photon noise.
- --blur-px .7 is the default vessel blur in output pixels; use 0 for no blur.
- --target-shift-mm 0 0 20 moves the target 20 mm in superior LPS Z. Shift is in world LPS mm, not pixels.
- Larger field-mm shows more anatomy; smaller shows less. Match real detector geometry before interpreting magnification.
- --a-angles and --b-angles are three intrinsic ZXY Euler angles in LPS, in degrees. Zero has the source anterior to the target (LPS -Y), beam +Y, up +Z. +90 Z puts the source on LPS +X. These are renderer controls; do not insert machine RAO/LAO numbers as if their signs and axes had been calibrated. Your previous nanoDRR/DeepDRR scripts may use different baseline directions.
- GPU out of memory: add --batch-rays 256. CPU will be slower.

Output files are replaced if you rerun into the same output directory. Use different output directories for comparisons.

## Outputs

biplane_six_panel.png has two rows (A and B) and three columns (bright vessel projection / CT DRR / dark vessel projection).

Six separate PNGs are also saved, plus plane_A_raw.npz and plane_B_raw.npz containing CT attenuation integrals and vessel path lengths before display scaling. geometry.json records source, beam, target, distances, pixel size and selections. Display contrast is normalised independently for each view; it is not quantitative iodine concentration or radiographic calibration.

## Corrections to the existing Notion instructions

- Use deactivate, not deactive.
- After deactivation, explicitly cd back to the script directory before executing scripts.
- Keep the SAME case ID for CREATE folder creation, Windows upload, SLURM submission and download. The pasted instructions alternated between two different case IDs.
- Check the case ID has no accidental trailing characters (one copy ended in an extra e).
- Do not put comments after a PowerShell backtick or Bash backslash, or between continued arguments. Put comments before the command.
- The vessel-enhancement command needs a backtick after python ".\6_prepare_vessel_enhanced_ct.py" if you use that separate step.
- TotalSegmentator default output is a directory: check whether brain.nii.gz is INSIDE the path named 5_CT_LPS_brain_mask.nii.gz or 6_CTA_LPS_brain_mask.nii.gz. A directory with a .nii.gz suffix is not itself a NIfTI file. Supply the actual brain.nii.gz to registration or move it to the intended filename after confirming output.
- When the first patient is itself the chosen template, moving/template paths can be the same intentionally. For later patients, keep the template paths fixed and change only moving paths.
- The pasted CT-registration/resampling step is a placeholder, not an executable command. Keep using your tested registration script; its exact filename/options were not supplied here and are not invented in this guide.
- CREATE commands are Bash after SSH; local upload/download and rendering commands are Windows PowerShell.

## Validation limits and method references

The script passed Python syntax checking and numerical ray-intersection checks. No patient NIfTI volumes or usable local PyTorch runtime were supplied, so full rendering and GPU performance are not validated here. First test the 256-pixel trial and compare manually.

NiBabel affines map voxels to RAS world coordinates even for LPS-oriented voxel storage; the script explicitly converts world coordinates to LPS:
https://nipy.org/nibabel/coordinate_systems.html

PyTorch grid_sample performs the trilinear interpolation with align_corners=True and zero outside the volume:
https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html

Real DSA subtracts pre-contrast and contrast signals in the appropriate log-intensity space. These vessel-only panels are an idealised projection surrogate:
https://dicom.nema.org/medical/DICOM/2021b/output/chtml/part04/sect_N.2.5.html
