# MT-Anatomy-Localisation-Project
Live anatomy localisation on fluoroscopy videos during Mechanical Thrombectomy procedures

In nanodrr-test folder:

**fluoroscopy_sim.py** 

    - run to open fluoroscopy simulation app using GPU
    - simulates C-Arm Control 
    - can change input ct image and segmentation by browsing folders
    - can choose which segmentations to impose

**Build scan index.py** 

    - saves the details of each scan from multiple datasets in one csv file 
    - saves case_id, dataset, image_path, label_path (to input into the subject class in the nanodrr script), anatomical regiod, labels present, and annotation type (manual/automatic) 
    - each dataset will have its own adapter script that will output the image and label path - in the adapter folder

**Check_ct.py** 

    - repairs the NIfTI header geometry so that ITK accepts the file. 
    - current scans probably have affine matrix however ITK only supports orthonormal direction cosines 
    - this is just to view, as nanodrr can accept affine (stick to original as we would need to change the segmentation file too) 

**Find_composite_candidate.py**
    - go through all of the segmentations and find which region each scan belongs mostly to (which will help us to put together body parts with the most labels present) 
    - all outputs in composite_candidate_output folder in metadata folder 
