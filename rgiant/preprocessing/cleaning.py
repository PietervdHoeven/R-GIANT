import nibabel as nib
import numpy as np
import ants
import os
import shutil
import time
import datetime
import subprocess
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.gradients import read_bvals_bvecs
from nibabel.processing import resample_from_to, resample_to_output
from nibabel.funcs import as_closest_canonical
from scipy.io import loadmat
from dipy.core.gradients import reorient_bvecs, gradient_table
from rgiant.utils.logging import setup_logger


def load_data(paths: dict) -> tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, np.ndarray, int]:
    """
    Loads DWI and structural MRI data along with corresponding b-values and b-vectors.

    Args:
        paths (dict): A dictionary containing file paths with the following keys:
            - "dwi": Path to the DWI NIfTI file.
            - "bval": Path to the .bval file.
            - "bvec": Path to the .bvec file.
            - "smri": Path to the structural MRI NIfTI file.

    Returns:
        tuple:
            dwi_img (nib.Nifti1Image): Loaded diffusion-weighted imaging (DWI) NIfTI image.
            smri_img (nib.Nifti1Image): Loaded structural MRI NIfTI image.
            bvals (np.ndarray): Array of b-values.
            bvecs (np.ndarray): Array of b-vectors.
            num_volumes (int): Number of 3D volumes in the DWI image.
    """

    # Load the DWI image and reorient it to a canonical orientation (RAS+)
    dwi_img = as_closest_canonical(nib.load(paths["dwi"]))

    # Load the b-values and b-vectors from their respective text files
    bvals, bvecs = read_bvals_bvecs(paths["bval"], paths["bvec"])

    # Load the structural MRI image
    smri_img = nib.load(paths["smri"])

    # Extract the number of volumes from the last dimension of the DWI data
    num_volumes = dwi_img.get_fdata().shape[-1]

    # Return the loaded data and metadata
    return dwi_img, smri_img, bvals, bvecs, num_volumes



def denoise_img(img: nib.Nifti1Image, out_path: str) -> nib.Nifti1Image:
    """
    Applies non-local means denoising to a diffusion MRI volume.

    Args:
        img (nib.Nifti1Image): The input NIfTI image to be denoised.
        out_path (str): File path to save the denoised image if 'save' is True.
        save (bool, optional): Whether to save the denoised image to disk. Defaults to True.

    Returns:
        nib.Nifti1Image: The denoised image as a NIfTI object.
    """
    # Extract the image data as a NumPy array
    data = img.get_fdata()

    # Estimate noise standard deviation using DIPY's method
    sigma = estimate_sigma(data, N=4)

    # Apply non-local means denoising with a small patch and block radius
    denoised_data = nlmeans(data, sigma=sigma, patch_radius=1, block_radius=1)

    # Wrap the denoised data in a new NIfTI image using the original affine and header
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)

    # Save the image to disk
    nib.save(denoised_img, out_path)

    # Return the denoised image
    return denoised_img



def extract_b0_img(dwi_img: nib.Nifti1Image, bvals: np.ndarray, out_path: str) -> tuple[nib.Nifti1Image, int]:
    """
    Extracts the first b=0 (non-diffusion-weighted) image from a DWI dataset.

    Args:
        dwi_img (nib.Nifti1Image): The full DWI image containing multiple volumes.
        bvals (np.ndarray): 1D array of b-values corresponding to each volume.
        out_path (str): File path to save the extracted b=0 image if 'save' is True.
        save (bool, optional): Whether to save the extracted image to disk. Defaults to True.

    Returns:
        tuple:
            b0_img (nib.Nifti1Image): The extracted b=0 image as a NIfTI object.
            b0_index (int): The index of the first b=0 volume in the DWI data.
    """
    # Find the index of the first volume with b-value == 0
    b0_index = np.where(bvals == 0)[0][0]

    # Extract the corresponding 3D image data from the 4D DWI volume
    b0_data = dwi_img.get_fdata()[..., b0_index]

    # Create a new NIfTI image using the original affine and header
    b0_img = nib.Nifti1Image(b0_data, dwi_img.affine, dwi_img.header)

    # Save the image to disk
    nib.save(b0_img, out_path)

    # Return the image and the b0 volume index
    return b0_img, b0_index



def calculate_mc_transformations(
    dwi_ants: ants.ANTsImage,
    b0_ants: ants.ANTsImage,
    num_volumes: int,
    b0_index: int,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    out_path: str,
    save: bool = True
) -> tuple[dict, nib.Nifti1Image, nib.Nifti1Image]:
    """
    Performs motion correction on a 4D DWI volume using rigid registration to the b0 image.

    Args:
        dwi_ants (ants.ANTsImage): The full DWI image in ANTs format.
        b0_ants (ants.ANTsImage): The extracted b0 image in ANTs format.
        num_volumes (int): Total number of volumes in the DWI image.
        b0_index (int): Index of the b0 image in the 4D DWI volume.
        affine (np.ndarray): Affine transformation matrix to use in the output NIfTI.
        header (nib.Nifti1Header): Header from the original DWI NIfTI to preserve metadata.
        out_path (str): File path to save the motion-corrected image if 'save' is True.
        save (bool, optional): Whether to save the motion-corrected image to disk. Defaults to True.

    Returns:
        tuple:
            mc_affine_paths (dict): A dictionary mapping volume indices to paths of forward affine transforms.
            dwi_ants (ants.ANTsImage): The original input DWI in ANTs format (unchanged).
            b0_ants (ants.ANTsImage): The original b0 reference image in ANTs format.
    """
    # Initialize an empty array to store motion-corrected DWI data
    mc_dwi_data = np.zeros_like(dwi_ants.numpy())

    # Dictionary to hold the file paths to affine transforms per volume
    mc_affine_paths = {}

    # Loop over all volumes to perform rigid registration to the b0 image
    for i in range(num_volumes):
        if i == b0_index:
            # If current volume is the b0 image, no registration needed
            mc_dwi_data[..., i] = b0_ants.numpy()
            continue

        # Extract the i-th volume from the 4D DWI ANTs image
        vol_denoised_ants = dwi_ants[:, :, :, i]

        # Set the direction to match the b0 image to ensure spatial consistency
        vol_denoised_ants.set_direction(b0_ants.direction)

        # Register the current volume to the b0 using rigid body transformation
        registration = ants.registration(
            fixed=b0_ants,
            moving=vol_denoised_ants,
            type_of_transform="Rigid"
        )

        # If saving is enabled, store the registered volume into the output array
        if save:
            mc_dwi_data[..., i] = registration['warpedmovout'].numpy()

        # Store the path to the forward affine transformation matrix
        mc_affine_paths[i] = registration['fwdtransforms'][0]

    # Save the motion-corrected DWI image to disk if requested
    if save:
        mc_dwi_img = nib.Nifti1Image(mc_dwi_data, affine, header)
        nib.save(mc_dwi_img, out_path)

    # Return the affine transform paths and the input ANTs images for reference
    return mc_affine_paths, dwi_ants, b0_ants



def skullstrip_img(in_path: str, out_path: str, device: str = "cuda") -> nib.Nifti1Image:
    """
    Applies skull stripping to a NIfTI image using the HD-BET tool via command-line.

    Args:
        in_path (str): Path to the input NIfTI image to be skull-stripped.
        out_path (str): Path to save the skull-stripped output image.
        device (str, optional): Device to run HD-BET on ('cuda' for GPU or 'cpu'). Defaults to "cuda".

    Returns:
        nib.Nifti1Image: The skull-stripped image loaded as a NIfTI object.
    """
    # Construct the HD-BET command with input, output, and device flags
    hd_bet_cmd = f"hd-bet -i {in_path} -o {out_path} -device {device}"
    #print(f"Running skullstripping command: {hd_bet_cmd}")

    # Record the start time for performance logging
    start_time = time.time()

    try:
        # Run the skull stripping command as a subprocess
        result = subprocess.run(
            hd_bet_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        # Print stdout from the subprocess if successful
        print("Command output:", result.stdout)

    except subprocess.CalledProcessError as e:
        # Print any error message if the subprocess fails
        print("Error during skullstripping:", e.stderr)

    finally:
        # Report total time taken for skull stripping
        end_time = time.time()
        print(f"Skullstripping completed in {round(end_time - start_time, 2)} seconds.")

    # Load and return the skull-stripped output image
    return nib.load(out_path)



def align_resolution_and_FOV(
    b0_img: nib.Nifti1Image, 
    smri_img: nib.Nifti1Image, 
    b0_out_path: str,
    smri_out_path: str
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Resamples the b0 and sMRI images to have aligned resolution and field of view (FOV).

    Args:
        b0_img (nib.Nifti1Image): The b0 image extracted from the DWI data.
        smri_img (nib.Nifti1Image): The structural MRI image to align with the b0 image.
        b0_out_path (str): Path to save the upsampled b0 image.
        smri_out_path (str): Path to save the resampled sMRI image.

    Returns:
        tuple:
            upsampled_b0_img (nib.Nifti1Image): The b0 image upsampled to 1mm isotropic resolution.
            resampled_smri_img (nib.Nifti1Image): The sMRI image resampled to match the b0 image's space.
    """
    # Upsample the b0 image to 1.0 mm isotropic voxel size
    upsampled_b0_img = resample_to_output(b0_img, voxel_sizes=(1.0, 1.0, 1.0))

    # Resample the sMRI image to the upsampled b0 image space and resolution
    resampled_smri_img = resample_from_to(smri_img, upsampled_b0_img)

    # Save the resampled sMRI image to disk
    nib.save(resampled_smri_img, smri_out_path)

    # Save the upsampled b0 image to disk
    nib.save(upsampled_b0_img, b0_out_path)

    # Return both resampled images
    return upsampled_b0_img, resampled_smri_img



def calculate_nonlinear_transformations(
    b0_ants: ants.ANTsImage,
    smri_ants: ants.ANTsImage,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    out_path: str,
    save: bool = True
) -> dict:
    """
    Performs nonlinear registration of the b0 image to the structural MRI using ANTs SyN transformation.

    Args:
        b0_ants (ants.ANTsImage): The b0 image in ANTs format (moving image).
        smri_ants (ants.ANTsImage): The structural MRI in ANTs format (fixed image).
        affine (np.ndarray): Affine transformation matrix to use in the output NIfTI image.
        header (nib.Nifti1Header): NIfTI header to retain in the saved warped image.
        out_path (str): Path to save the nonlinearly registered b0 image.
        save (bool, optional): Whether to save the warped image to disk. Defaults to True.

    Returns:
        dict: A dictionary containing registration results, including transformation paths and warped image.
    """
    # Perform nonlinear registration using the Symmetric Normalization (SyN) algorithm
    registration = ants.registration(
        fixed=smri_ants,
        moving=b0_ants,
        type_of_transform="SyN"
    )

    # Save the warped moving image as a NIfTI file if requested
    if save:
        nib.save(nib.Nifti1Image(registration["warpedmovout"].numpy(), affine, header), out_path)

    # Return the full ANTs registration dictionary
    return registration



def downsample_smri_and_parcellation(
    smri_img: nib.Nifti1Image, 
    parc_img: nib.Nifti1Image, 
    smri_out_path: str,
    parc_out_path: str
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Downsamples the structural MRI and its corresponding parcellation to 2mm isotropic resolution.

    Args:
        smri_img (nib.Nifti1Image): The high-resolution structural MRI image.
        parc_img (nib.Nifti1Image): The brain parcellation image aligned with the structural MRI.
        smri_out_path (str): Path to save the downsampled structural MRI.
        parc_out_path (str): Path to save the downsampled parcellation image.

    Returns:
        tuple:
            downsampled_smri_img (nib.Nifti1Image): The structural MRI resampled to 2mm resolution.
            downsampled_parc_img (nib.Nifti1Image): The parcellation image resampled to match the downsampled MRI.
    """
    # Downsample the structural MRI to 2mm isotropic resolution
    downsampled_smri_img = resample_to_output(smri_img, voxel_sizes=(2.0, 2.0, 2.0))

    # Resample the parcellation image to match the downsampled MRI using nearest-neighbor interpolation
    downsampled_parc_img = resample_from_to(parc_img, downsampled_smri_img, order=0)

    # Save the downsampled structural MRI to disk
    nib.save(downsampled_smri_img, smri_out_path)

    # Save the downsampled parcellation image to disk
    nib.save(downsampled_parc_img, parc_out_path)

    # Return both downsampled images
    return downsampled_smri_img, downsampled_parc_img



def apply_transformations(
    smri_ants: ants.ANTsImage,
    dwi_ants: ants.ANTsImage,
    b0_ants: ants.ANTsImage,
    b0_index: int,
    num_volumes: int,
    nl_registration: dict,
    mc_transforms: dict,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    out_path: str
) -> nib.Nifti1Image:
    """
    Applies both nonlinear and motion correction transformations to each volume in a DWI image.
    Load images directly as LPS-aligned ANTS images so that ANTS handles image orientations correctly
    Warp transformation <- Non-linear affine <- Motion correct affine
    Result is stored as RAS-aligned NIfTI to be compatible with later processing

    Args:
        smri_ants (ants.ANTsImage): The structural MRI image used as the final space.
        dwi_ants (ants.ANTsImage): The original 4D DWI image in ANTs format.
        b0_ants (ants.ANTsImage): The b0 reference image in ANTs format.
        b0_index (int): Index of the b0 image in the DWI sequence.
        num_volumes (int): Total number of DWI volumes.
        nl_registration (dict): Dictionary containing forward transforms from b0 to sMRI space.
        mc_transforms (dict): Dictionary of motion correction affine paths per volume index.
        affine (np.ndarray): Affine matrix to assign to the output NIfTI image.
        header (nib.Nifti1Header): Header to use for the output NIfTI image.
        out_path (str): Path to save the fully corrected DWI image.

    Returns:
        nib.Nifti1Image: A 4D NIfTI image with all volumes transformed into the structural MRI space.
    """
    # Initialize an empty array to store the transformed DWI data
    corrected_dwi_data = np.zeros((*smri_ants.shape, dwi_ants.shape[-1]))

    # Loop through each volume in the DWI dataset
    for i in range(num_volumes):
        # Extract the current 3D volume as an ANTs image
        vol_ants = dwi_ants[:, :, :, i]

        # Set the spatial orientation to match the b0 image
        vol_ants.set_direction(b0_ants.direction)

        # Start building the transformation list with nonlinear warps from b0 to sMRI
        transformlist = [
            nl_registration["fwdtransforms"][0],  # warp
            nl_registration["fwdtransforms"][1]   # affine
        ]

        # If not the b0 volume, include motion correction transform for this volume
        if i != b0_index:
            transformlist.append(mc_transforms[i])

        # Apply the full transformation chain to this volume using bspline interpolation. Transformations are applied from last to first in the list.
        corrected_vol = ants.apply_transforms(
            fixed=b0_ants,
            moving=vol_ants,
            transformlist=transformlist,
            interpolator='bSpline'
        )

        # Store the corrected volume in the result array
        corrected_dwi_data[..., i] = corrected_vol.numpy()

    # Wrap the corrected data in a NIfTI image with the target affine and header
    corrected_dwi_img = nib.Nifti1Image(corrected_dwi_data, affine, header)

    # Save the result to disk
    nib.save(corrected_dwi_img, out_path)

    # Return the corrected image object
    return corrected_dwi_img



def apply_bvec_rotations(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    esc_affine_path: str,
    mc_affine_paths: dict,
    bvecs_out_path: str
) -> None:
    """
    Applies rotation corrections to b-vectors based on susceptibility and motion correction affine transforms.
    Original bvecs are in LAS -> ANTS works in LPS so we first reorient bvecs to LPS and apply affines -> We finally reorient to RAS
    LAS2LPS -> Motion correct affine -> Non-linear affine -> LPS2RAS

    Args:
        bvals (np.ndarray): Array of b-values for the DWI volumes.
        bvecs (np.ndarray): Array of original b-vectors (3 x N).
        esc_affine_path (str): Path to the susceptibility correction affine matrix (.mat file).
        mc_affine_paths (dict): Dictionary mapping volume indices to motion correction affine transform paths.
        bvecs_out_path (str): Path to save the reoriented b-vectors.

    Returns:
        None
    """
    # Create a gradient table with a mask to identify non-b0 volumes
    gtab = gradient_table(bvals=bvals, bvecs=bvecs, b0_threshold=1)

    # Get indices of all non-b0 volumes (we don't need to rotate b0 volumes)
    non_b0_indices = np.where(~gtab.b0s_mask)[0]
    num_non_b0 = non_b0_indices.shape[0]

    # Initialize a container for the composite 3x3 rotation matrices
    composite_affines = np.empty((3, 3, num_non_b0), dtype=np.float32)

    # Load the susceptibility correction affine transform matrix from .mat file
    susceptibility_dict = loadmat(esc_affine_path)
    susceptibility_array = susceptibility_dict['AffineTransform_float_3_3'].flatten()
    R_susceptibility = susceptibility_array[:9].reshape((3, 3))

    # Define coordinate system correction matrices
    R_lps2ras = np.diag([-1, -1, 1])
    R_las2lps = np.diag([1, -1, 1])

    # Loop over each non-b0 volume to construct composite affine rotations
    for i, idx in enumerate(non_b0_indices):
        # Load the motion correction affine matrix for the given volume
        motion_dict = loadmat(mc_affine_paths[idx])
        motion_array = motion_dict['AffineTransform_float_3_3'].flatten()
        R_motion = motion_array[:9].reshape((3, 3))

        # Compute the total rotation matrix for this volume (last to first)
        R_total = R_lps2ras @ R_susceptibility @ R_motion @ R_las2lps

        # Store the result in the composite array8
        composite_affines[:, :, i] = R_total

    # Apply the composite rotation matrices to reorient the b-vectors
    gtab = reorient_bvecs(gtab, composite_affines)

    # Save the rotated b-vectors to file in FSL-style (3 x N, transposed)
    np.savetxt(bvecs_out_path, gtab.bvecs.T, fmt="%.8f", delimiter=" ")



def run_cleaning_pipeline(patient_id: str, session_id: str, data_dir: str = "data/", save: bool = True, clear_temp: bool = False, stream = True, log_dir = "logs/") -> None:

    # Setup logger
    logger = setup_logger(name="cleaning_logger", prefix="cleaning", patient_id=patient_id, session_id=session_id, stream=stream, log_dir=log_dir)

    start_time = time.time()
    logger.info(f"Starting DWI cleaning pipeline for patient {patient_id} | Session {session_id}")

    # Create the directory structure for the patient and session if it doesn't exist yet
    os.makedirs(f"{data_dir}/temp/{patient_id}_{session_id}", exist_ok=True)
    os.makedirs(f"{data_dir}/clean/{patient_id}_{session_id}", exist_ok=True)

    paths = {
        "dwi": f"{data_dir}/mr/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi.nii.gz",
        "bval": f"{data_dir}/mr/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi.bval",
        "bvec": f"{data_dir}/mr/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi.bvec",
        "smri": f"{data_dir}/fs/{patient_id}_{session_id}/brain.mgz",
        "parc": f"{data_dir}/fs/{patient_id}_{session_id}/aparc+aseg.mgz",

        "denoised_dwi": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi_denoised.nii.gz",
        "denoised_mc_dwi": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi_denoised_mc.nii.gz",
        "resampled_smri": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_smri_resampled.nii.gz",
        "downsampled_smri": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_smri_downsampled.nii.gz",
        "denoised_b0": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_b0_denoised.nii.gz",
        "denoised_brain_b0": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_b0_denoised_brain.nii.gz",
        "denoised_brain_upsampled_b0": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_b0_denoised_brain_upsampled.nii.gz",
        "b0_to_smri": f"{data_dir}/temp/{patient_id}_{session_id}/{patient_id}_{session_id}_b0_to_smri.nii.gz",

        "corrected_dwi": f"{data_dir}/clean/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi_corrected.nii.gz",
        "downsampled_parc": f"{data_dir}/clean/{patient_id}_{session_id}/{patient_id}_{session_id}_parc_downsampled.nii.gz",
        "rotated_bvec": f"{data_dir}/clean/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi_rotated.bvec"
    }

    # Move the bvals file to the clean_mri directory
    try:
        logger.info("Moving bvals file to clean directory")
        shutil.copy(paths["bval"], f"{data_dir}/clean/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi.bval")
    except Exception:
        logger.exception("Failed to move bvals file to clean directory")
        return

    try:
        logger.info("Step 0: Loading input data")
        dwi_img, smri_img, bvals, bvecs, num_volumes = load_data(paths)
    except Exception:
        logger.exception("Failed at Step 0: Loading input data")
        return

    try:
        logger.info("Step 1: Denoising DWI")
        denoised_dwi_img = denoise_img(img=dwi_img, out_path=paths["denoised_dwi"])
    except Exception:
        logger.exception("Failed at Step 1: Denoising DWI")
        return

    try:
        logger.info("Step 2: Extracting b=0 image")
        denoised_b0_img, b0_index = extract_b0_img(dwi_img=denoised_dwi_img, bvals=bvals, out_path=paths["denoised_b0"])
    except Exception:
        logger.exception("Failed at Step 2: Extracting b=0 image")
        return

    try:
        logger.info("Step 3: Calculating Motion Correction transformations")
        mc_transforms, denoised_dwi_ants, denoised_b0_ants = calculate_mc_transformations(
            dwi_ants=ants.image_read(paths["denoised_dwi"]),
            b0_ants=ants.image_read(paths["denoised_b0"]),
            num_volumes=num_volumes,
            b0_index=b0_index,
            affine=denoised_dwi_img.affine,
            header=denoised_dwi_img.header,
            out_path=paths["denoised_mc_dwi"],
            save=save
        )
    except Exception:
        logger.exception("Failed at Step 3: Motion correction")
        return

    try:
        logger.info("Step 4: Skull stripping")
        denoised_brain_b0_img = skullstrip_img(in_path=paths["denoised_b0"], out_path=paths["denoised_brain_b0"], device="cuda")
    except Exception:
        logger.exception("Failed at Step 4: Skull stripping")
        return

    try:
        logger.info("Step 5: Aligning resolution and FOV")
        denoised_brain_upsampled_b0_img, resampled_smri_img = align_resolution_and_FOV(
            b0_img=denoised_brain_b0_img,
            smri_img=smri_img,
            b0_out_path=paths["denoised_brain_upsampled_b0"],
            smri_out_path=paths["resampled_smri"]
        )
    except Exception:
        logger.exception("Failed at Step 5: Aligning resolution and FOV")
        return

    try:
        logger.info("Step 6: Nonlinear registration")
        nl_registration = calculate_nonlinear_transformations(
            b0_ants=ants.image_read(paths["denoised_brain_upsampled_b0"]),
            smri_ants=ants.image_read(paths["resampled_smri"]),
            affine=denoised_brain_upsampled_b0_img.affine,
            header=denoised_brain_upsampled_b0_img.header,
            out_path=paths["b0_to_smri"],
            save=save
        )
    except Exception:
        logger.exception("Failed at Step 6: Nonlinear registration")
        return

    try:
        logger.info("Step 7: Downsampling sMRI and parcellation")
        downsampled_smri_img, downsampled_parc_img = downsample_smri_and_parcellation(
            smri_img=resampled_smri_img,
            parc_img=nib.load(paths["parc"]),
            smri_out_path=paths["downsampled_smri"],
            parc_out_path=paths["downsampled_parc"]
        )
    except Exception:
        logger.exception("Failed at Step 7: Downsampling")
        return

    try:
        logger.info("Step 8: Applying full transformations to DWI volumes")
        corrected_dwi_img = apply_transformations(
            smri_ants=ants.image_read(paths["downsampled_smri"]),
            dwi_ants=denoised_dwi_ants,
            b0_ants=denoised_b0_ants,
            b0_index=b0_index,
            num_volumes=num_volumes,
            nl_registration=nl_registration,
            mc_transforms=mc_transforms,
            affine=downsampled_smri_img.affine,
            header=downsampled_smri_img.header,
            out_path=paths["corrected_dwi"]
        )
    except Exception:
        logger.exception("Failed at Step 8: Applying full transformations")
        return

    try:
        logger.info("Step 9: Rotating b-vectors")
        apply_bvec_rotations(
            bvals=bvals,
            bvecs=bvecs,
            esc_affine_path=nl_registration["fwdtransforms"][1],
            mc_affine_paths=mc_transforms,
            bvecs_out_path=paths["rotated_bvec"]
        )
    except Exception:
        logger.exception("Failed at Step 9: Rotating b-vectors")
        return
    
    if clear_temp:
        try:
            logger.info("Clearing temporary files")
            shutil.rmtree(f"{data_dir}/temp/{patient_id}_{session_id}")
        except:
            logger.exception("Failed to clear temporary files")

    logger.info(f"Cleaning pipeline completed for patient {patient_id} | session {session_id} in {round(time.time() - start_time, 2)} seconds.")



# Example use case for debugging or running directly
if __name__ == "__main__":
    run_cleaning_pipeline(
        patient_id="0001", 
        session_id="0757", 
        data_dir=r"C:\Users\piete\Documents\Projects\R-GIANT\data",
        clear_temp=True, 
        save=False,
        stream=True
        )