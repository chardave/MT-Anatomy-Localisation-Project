# Two nanoDRR biplane versions using enhanced CT

Use 14_vessel_enhanced_CT.nii.gz as --ct and 13_CTA_vessel_labels.nii.gz as --labels. Use the existing working .render_env. Replace make_biplane_nanodrr.py with this version; no dependency changes or preprocessing reruns are needed if the enhanced CT already exists.

Both outputs share exactly the same enhanced CT projection. Without labels means no coloured overlay; vessels remain visible from the enhancement. With labels adds a cyan overlay of the union of selected foreground labels on the centre panels only. The left/right vessel projections stay the same. This is a single-colour vessel annotation, not separate anatomical class colours.

Output: biplane_without_labels.png and biplane_with_labels.png, individual panels, raw arrays and geometry.json. --overlay-alpha .4 controls overlay opacity (0 to 1). --label-ids selects which anatomical IDs to project; omit for all positive IDs. --ct must be enhanced by the preceding enhancement script; this renderer does not perform enhancement itself.

Use a short output path to avoid the Windows save error:

```powershell
.\.render_env\Scripts\python.exe .\make_biplane_nanodrr.py `
  --ct "$case\14_vessel_enhanced_CT.nii.gz" `
  --labels "$case\13_CTA_vessel_labels.nii.gz" `
  --output-dir "C:\fluoro_output\000118_enhanced" `
  --size 512 `
  --device auto `
  --overlay-alpha 0.4
```

Camera/render API follows your previously tested version. Rotations and translations remain nanoDRR GUI inputs; machine-angle mapping is unverified. Default CT window is 2nd to 99th percentile, inverted. --no-invert-ct changes polarity. All static simulation limitations remain: no flow/catheter, approximate detector field size, projected segmentation panels are not true DSA subtraction. Syntax and overlay logic were checked; patient rendering of this update was not tested here.
