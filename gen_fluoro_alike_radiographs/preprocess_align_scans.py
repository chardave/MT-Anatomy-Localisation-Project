from pathlib import Path
import SimpleITK as sitk

def print_image_info(name, img):
    print(f"{name}:")
    print(f"  Size: {img.GetSize()}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")
    print(f"  Direction: {img.GetDirection()}")
    print()

def almost_equal_tuple(a, b, tolerence = 1e-4):
    if len(a) != len(b):
        return False
    
    for x, y in zip(a,b):
        if abs(x-y) > tolerence:
            return False
    return True

def same_geometry(img1, img2, tolerence=1e-4):
    # check if the image and mask have the same size, spacing, origin, and direction
    same_size = img1.GetSize() == img2.GetSize()
    same_spacing = almost_equal_tuple(img1.GetSpacing(), img2.GetSpacing(), tolerence)
    same_origin = almost_equal_tuple(img1.GetOrigin(), img2.GetOrigin(), tolerence)
    same_direction = almost_equal_tuple(img1.GetDirection(), img2.GetDirection(), tolerence)

    return same_size and same_spacing and same_origin and same_direction

def make_binary_mask(mask):
    # temporary registration mask so labelmask is not registered as an intensity image
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return mask

def make_distance_map(mask):
    # distance map = how far each pixel is from the nearest non-zero pixel in the mask (vessel label)
    distance = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False, # inside vessel is negative distance, outside is positive distance, boundary = 0
        squaredDistance=False, # norrmal distance, not squared
        useImageSpacing=True, # distance in mm, not pixels
    )
    return sitk.Cast(distance, sitk.sitkFloat32)

# Main Registration Function

def register_case(
        template_cta_path,
        template_vessel_path,
        moving_cta_path,
        moving_vessel_path,
        output_cta_path,
        output_vessel_path,
):
    #load images
    print("Loading images...")

    template_cta = sitk.ReadImage(str(template_cta_path))
    template_vessel = sitk.ReadImage(str(template_vessel_path))

    moving_cta = sitk.ReadImage(str(moving_cta_path))
    moving_vessel = sitk.ReadImage(str(moving_vessel_path))

    print_image_info("Template CTA", template_cta)
    print_image_info("Template vessel mask", template_vessel)
    print_image_info("Moving CTA", moving_cta)
    print_image_info("Moving vessel mask", moving_vessel)

    # check CTA and vessel masks have the same geometry

    if not same_geometry(template_cta, template_vessel):
        raise ValueError("Template CTA and vessel mask do not have the same geometry.")
    
    if not same_geometry(moving_cta, moving_vessel):
        raise ValueError("Moving CTA and vessel mask do not have the same geometry.")
    
    # make binary masks for registration
    # keep original labels for resampling after registration
    template_vessel_labels = template_vessel
    moving_vessel_labels = moving_vessel

    template_vessel_registration = make_binary_mask(template_vessel)
    moving_vessel_registration = make_binary_mask(moving_vessel)

    fixed_distance = make_distance_map(template_vessel_registration)
    moving_distance = make_distance_map(moving_vessel_registration)

    #create initial transform - create starting guess for registration
    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(template_vessel_registration, sitk.sitkFloat32),
        sitk.Cast(moving_vessel_registration, sitk.sitkFloat32),
        sitk.Euler3DTransform(), # rigid transform (rotation + translation)
        sitk.CenteredTransformInitializerFilter.MOMENTS, # initialise based on object distribution (center of mass and principal axes)
    )

    print("Setting up registration...")
    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMeanSquares() # use mean squares metric for distance maps
    registration.SetInterpolator(sitk.sitkLinear) # use linear interpolation for distance maps
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate = 2.0,
        minStep = 1e-4,
        numberOfIterations=200,
        relaxationFactor = 0.5, # if optimiser overshoots/slows down, reduce step size by this factor
        gradientMagnitudeTolerance = 1e-6, # stops when gradient becomes small enough (converged)
    )
    registration.SetOptimizerScalesFromPhysicalShift() # helps with convergence by scaling the optimizer steps based on physical distance rather than pixel distance

    # Multiresolution framework - register at lower resolution first, then refine at higher resolution
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() # smoothing in mm, not pixels

    registration.SetInitialTransform(initial_transform, inPlace=False) # returns new final transform instead of the original transform object directly

    print("Running registration...")
    final_transform = registration.Execute(fixed_distance, moving_distance)

    print("Registration finished.")
    print("Final metric value: ", registration.GetMetricValue())
    print("Stop Condition: ", registration.GetOptimizerStopConditionDescription())
    print()

    # resample moving CTA and vessel mask to template space using final transform
    print("Resampling moving CTA to template space...")
    aligned_cta = sitk.Resample(
        moving_cta,
        template_cta,
        final_transform,
        sitk.sitkLinear, # good for CT intensities
        -1024, # default background value for CT
        moving_cta.GetPixelID(), # keep CTA pixel type
    )
    print("Resampling moving vessel mask to template space...")
    aligned_vessel = sitk.Resample(
        moving_vessel,
        template_vessel,
        final_transform,
        sitk.sitkNearestNeighbor, # good for vessel mask - copies the nearest existing label value rather than interpolating
        0, # default background value for vessel mask
        moving_vessel.GetPixelID(), # keep vessel mask pixel type
    )

    # save outputs
    print("Saving aligned CTA and vessel mask... ")
    output_cta_path = Path(output_cta_path)
    output_vessel_path = Path(output_vessel_path)

    output_cta_path.parent.mkdir(parents=True, exist_ok=True)
    output_vessel_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(aligned_cta, str(output_cta_path))
    sitk.WriteImage(aligned_vessel, str(output_vessel_path))

    print("Done.")
    print("Saved aligned CTA to:", output_cta_path)
    print("Saved aligned vessel mask to:", output_vessel_path)


if __name__ == "__main__":
    # register_case(
    #     template_cta_path=r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\images\topaneu_center4_ct_002_0000.nii.gz",
    #     template_vessel_path=r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\vessel_masks\topaneu_center4_ct_002.nii.gz",
    #     moving_cta_path=r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\images\topaneu_center4_ct_005_0000.nii.gz",
    #     moving_vessel_path=r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\vessel_masks\topaneu_center4_ct_005.nii.gz",
    #     output_cta_path=r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\output\topaneu_center4_ct_005_0000_registered.nii.gz",
    #     output_vessel_path=r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\output\topaneu_center4_ct_005_0000_registered_vessels.nii.gz",
    # )

    images_dir = Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\images"
    )

    vessel_masks_dir = Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\vessel_masks"
    )

    aligned_images_dir = Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\aligned_images"
    )

    aligned_vessel_masks_dir = Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment\aligned_vessel_masks"
    )


    #fixed template files - 002 
    template_cta_path = images_dir / "topaneu_center4_ct_002_0000.nii.gz"
    template_vessel_path = vessel_masks_dir / "topaneu_center4_ct_002.nii.gz"

    image_paths = sorted(images_dir.glob("*_0000.nii.gz"))
    print("Found {} images to register.".format(len(image_paths)))

    for moving_cta_path in image_paths:

        # skip the template image
        if moving_cta_path.name == template_cta_path.name:
            print("Skipping template image:", moving_cta_path.name)
            continue

        # get the corresponding vessel mask path
        vessel_name = moving_cta_path.name.replace("_0000.nii.gz", ".nii.gz")
        moving_vessel_path = vessel_masks_dir / vessel_name

        # check if the vessel mask exists
        if not moving_vessel_path.exists():
            print("Vessel mask not found for image:", moving_cta_path.name)
            continue

        # set output paths
        output_cta_path = aligned_images_dir / moving_cta_path.name.replace(".nii.gz", "_registered.nii.gz")
        output_vessel_path = aligned_vessel_masks_dir / vessel_name.replace(".nii.gz", "_registered.nii.gz")


        print()
        print("=" * 80)
        print("Registering image:", moving_cta_path.name)
        print("Corresponding vessel mask:", moving_vessel_path.name)
        print("Output image:", output_cta_path.name)
        print("Output vessel mask:", output_vessel_path.name)
        print("=" * 80)

        try:
            register_case(
                template_cta_path=template_cta_path,
                template_vessel_path=template_vessel_path,
                moving_cta_path=moving_cta_path,
                moving_vessel_path=moving_vessel_path,
                output_cta_path=output_cta_path,
                output_vessel_path=output_vessel_path,
            )
        
        except Exception as e:
            print()
            print("Error registering image:", moving_cta_path.name)
            print("Error message:", str(e))
            continue

    print()
    print("All images registered. Aligned images saved to:", aligned_images_dir)