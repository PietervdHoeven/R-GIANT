import time
import datetime
import logging
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from filelock import FileLock
from src.utils.json_loading import load_special_fs_labels, load_parcellation_mappings
from scipy.ndimage import binary_dilation
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.dti import TensorModel
from dipy.viz import window, actor
from dipy.data import get_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines, values_from_volume
from dipy.tracking.streamlinespeed import length
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.segment.clustering import QuickBundles
from dipy.tracking.utils import target, connectivity_matrix



def setup_debug_logger(patient_id, session_id):
    """
    Setup a logger for debugging.
    """
    logger = logging.getLogger(f"debug_connectome_{patient_id}_{session_id}")
    logger.setLevel(logging.INFO)
    
    # Ensure logger does not propagate to avoid duplicate log entries when a master logger is present.
    logger.propagate = False  

    # Create the directory if it doesn't exist.
    os.makedirs("logs/connectomes", exist_ok=True)
    
    # Create a timestamped log file name.
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    log_filename = f"{patient_id}_{session_id}_{timestamp}.log"
    log_filepath = os.path.join("logs/connectomes", log_filename)
    
    # Setup file handler.
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Check if handlers already exist; if not, add file and stream handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        
        # Optional: add a stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger



def load_data(paths: dict) -> tuple[np.ndarray, np.ndarray, gradient_table, np.ndarray]:
    """
    Loads DWI data, gradient table, and FreeSurfer parcellation.

    Args:
        paths (dict): Dictionary with paths to:
            - "dwi": NIfTI file of DWI volume
            - "bval": b-values file
            - "bvec": b-vectors file
            - "smri_parc": structural MRI parcellation volume (e.g. aparc+aseg)

    Returns:
        Tuple containing:
            - dwi_data (np.ndarray): 4D DWI image
            - dwi_affine (np.ndarray): Affine matrix of the DWI image
            - gtab (GradientTable): DIPY gradient table object
            - fs_parcellation (np.ndarray): 3D volume of the parcellated brain
    """
    # Load DWI volume and affine transform
    dwi_data, dwi_affine = load_nifti(paths["dwi"])
    
    # Load b-values and b-vectors and construct gradient table
    bvals, bvecs = read_bvals_bvecs(paths["bval"], paths["bvec"])
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    
    # Load FreeSurfer parcellation image (e.g., aparc+aseg)
    smri_parc_img = nib.load(paths["smri_parc"])
    fs_parcellation = smri_parc_img.get_fdata()
    
    return dwi_data, dwi_affine, gtab, fs_parcellation



def create_wm_mask(fs_parcellation: np.ndarray, wm_labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a binary white matter mask from a parcellation volume.

    Args:
        fs_parcellation (np.ndarray): 3D parcellation volume (e.g., aparc+aseg) with FreeSurfer labels.
        wm_labels (list[int]): List of integer labels representing white matter regions.

    Returns:
        Tuple containing:
            - wm_mask (np.ndarray): Binary mask where WM voxels are True.
            - dilated_wm_mask (np.ndarray): Binary mask with one iteration of dilation applied.
    """
    # Identify voxels that belong to white matter based on the label list
    wm_mask = np.isin(fs_parcellation, wm_labels)

    # Slightly expand the WM mask to ensure tractography seeds can reach edges
    dilated_wm_mask = binary_dilation(wm_mask, iterations=1)

    return wm_mask, dilated_wm_mask


def fit_dti_model(dwi_data: np.ndarray, gtab: gradient_table, mask: np.ndarray) -> tuple[TensorModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a diffusion tensor model to the DWI data and extracts diffusion metrics.

    Args:
        dwi_data (np.ndarray): 4D DWI image.
        gtab (gradient_table): DIPY gradient table.
        mask (np.ndarray): Binary mask to restrict fitting (usually dilated white matter mask).

    Returns:
        Tuple containing:
            - dti_fit (TensorModel): Fitted diffusion tensor model.
            - fa_data (np.ndarray): Fractional anisotropy volume.
            - md_data (np.ndarray): Mean diffusivity volume.
            - ad_data (np.ndarray): Axial diffusivity volume.
            - rd_data (np.ndarray): Radial diffusivity volume.
    """
    # Fit the diffusion tensor model within the provided mask
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(dwi_data, mask=mask)

    # Extract scalar maps from the fit
    fa_data = dti_fit.fa  # Fractional Anisotropy
    md_data = dti_fit.md  # Mean Diffusivity
    ad_data = dti_fit.ad  # Axial Diffusivity
    rd_data = dti_fit.rd  # Radial Diffusivity

    # Replace NaNs with zeros to avoid issues downstream
    fa_data[np.isnan(fa_data)] = 0
    md_data[np.isnan(md_data)] = 0
    ad_data[np.isnan(ad_data)] = 0
    rd_data[np.isnan(rd_data)] = 0

    return dti_fit, fa_data, md_data, ad_data, rd_data



def generate_streamlines(
    dti_fit: TensorModel,
    wm_mask: np.ndarray,
    fa_data: np.ndarray,
    dwi_affine: np.ndarray,
    fa_thresh: float = 0.3,
    sphere_str: str = 'symmetric362'
) -> Streamlines:
    """
    Generates streamlines using deterministic tractography from a fitted DTI model.

    Args:
        dti_fit (TensorModel): Fitted diffusion tensor model.
        wm_mask (np.ndarray): White matter binary mask.
        fa_data (np.ndarray): Fractional anisotropy map for thresholding.
        dwi_affine (np.ndarray): Affine of the DWI volume for spatial reference.
        fa_thresh (float, optional): FA threshold for seeding. Defaults to 0.3.
        sphere_str (str, optional): Sphere used for orientation distribution. Defaults to 'symmetric362'.

    Returns:
        Streamlines: A DIPY `Streamlines` object containing all generated streamlines.
    """
    # Define seed points in high-FA white matter regions
    seed_mask = wm_mask & (fa_data > fa_thresh)
    seeds = utils.seeds_from_mask(seed_mask, affine=dwi_affine, density=1)

    # Load orientation sphere for directional modeling
    sphere = get_sphere(name=sphere_str)

    # Compute the orientation distribution function and clip to positive values
    odf_data = dti_fit.odf(sphere)
    pmf_data = odf_data.clip(min=0)

    # Create a direction getter using the deterministic peak of the ODF
    det_dg = DeterministicMaximumDirectionGetter.from_pmf(
        pmf_data, max_angle=30.0, sphere=sphere
    )

    # Define stopping criterion as remaining within the white matter
    stopping_criterion = BinaryStoppingCriterion(wm_mask)

    # Perform local deterministic streamline tracking
    streamlines_generator = LocalTracking(
        det_dg, stopping_criterion, seeds, dwi_affine, step_size=0.5
    )

    return Streamlines(streamlines_generator)



def filter_by_length(streamlines: Streamlines, min_length: float = 20.0, max_length: float = 300.0) -> Streamlines:
    """
    Filters streamlines based on their length.

    Args:
        streamlines (Streamlines): Collection of streamlines to filter.
        min_length (float, optional): Minimum allowed length in mm. Defaults to 20.0.
        max_length (float, optional): Maximum allowed length in mm. Defaults to 300.0.

    Returns:
        Streamlines: A new Streamlines object with only those that fall within the specified length range.
    """
    # Compute the length of each streamline
    streamline_lengths = length(streamlines)

    # Create a boolean mask for streamlines within the allowed range
    valid_mask = (streamline_lengths > min_length) & (streamline_lengths <= max_length)

    # Return filtered streamlines using the valid mask
    return Streamlines(streamlines[valid_mask])



def filter_anatomical(
    streamlines: Streamlines, 
    fs_aseg: np.ndarray, 
    fs_affine: np.ndarray, 
    excl_labels: list[int]
) -> Streamlines:
    """
    Removes streamlines that pass through unwanted anatomical regions.

    Args:
        streamlines (Streamlines): Input set of streamlines.
        fs_aseg (np.ndarray): Parcellation or segmentation volume (e.g., aparc+aseg).
        fs_affine (np.ndarray): Affine matrix for the segmentation volume.
        excl_labels (list[int]): List of labels to exclude (e.g., CSF or ventricles).

    Returns:
        Streamlines: Filtered streamlines that do not enter the excluded regions.
    """
    # Create a binary mask where excluded labels are True
    exclude_mask = np.isin(fs_aseg, excl_labels)

    # Use DIPY's `target` to exclude streamlines intersecting those regions
    filtered_streamlines = list(target(streamlines, fs_affine, exclude_mask, include=False))

    return Streamlines(filtered_streamlines)



def filter_outliers(streamlines: Streamlines, min_cluster_size: int = 5, threshold: float = 5.0) -> Streamlines:
    """
    Removes outlier streamlines by clustering and keeping only large clusters.

    Args:
        streamlines (Streamlines): Input streamlines to be filtered.
        min_cluster_size (int, optional): Minimum number of streamlines required to keep a cluster. Defaults to 5.
        threshold (float, optional): Clustering distance threshold (in mm). Lower values yield tighter clusters. Defaults to 5.0.

    Returns:
        Streamlines: Streamlines belonging to clusters of at least `min_cluster_size` streamlines.
    """
    # Cluster the streamlines using the QuickBundles algorithm
    qb = QuickBundles(threshold=threshold)
    clusters = qb.cluster(streamlines)

    # Extract clusters with enough streamlines
    large_clusters = clusters.get_large_clusters(min_cluster_size)

    # Flatten the list of streamlines from all large clusters
    filtered_streamlines = [sl for cluster in large_clusters for sl in cluster]

    return Streamlines(filtered_streamlines)



def preprocess_parcellation(fs_parcellation: np.ndarray, fs_idxs2graph_idxs: dict) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """
    Removes excluded labels from a FreeSurfer parcellation and remaps remaining labels to a compact range.
    The parcellatioin mapping from JSON only contains labels that we want as nodes in the connectome.
    We take the zeros matrix and relabel at all positions in the old parcellation when we find that label as key in the dict.
    So if we can't find the label in the dict, the label is not translated and copied to the zeros matrix, leaving the value as 0. Which is what we want, exclude all labels not in the dict.

    Args:
        fs_parcellation (np.ndarray): 3D volume with original parcellation labels (e.g., aparc+aseg).

    Returns:
        Tuple containing:
            - reduced_parcellation (np.ndarray): Parcellation with excluded labels set to 0 and remaining relabeled to a dense index.
            - fs_idxs2graph_idxs (dict[int, int]): Mapping from original FreeSurfer label → compact index.
            - reduced2fs (dict[int, int]): Mapping from compact index → original FreeSurfer label.
    """
    # Initialize output parcellation
    reduced_parcellation = np.zeros_like(fs_parcellation, dtype=int)

    # Relabel all retained regions (json loaded dicts always have str keys)
    for label, new_label in fs_idxs2graph_idxs.items():
        reduced_parcellation[fs_parcellation == int(label)] = new_label

    return reduced_parcellation



def sl_count_connectivity(
    streamlines: Streamlines, 
    affine: np.ndarray, 
    parcellation: np.ndarray
) -> tuple[np.ndarray, dict[tuple[int, int], Streamlines]]:
    """
    Computes a streamline count-based structural connectivity matrix.

    Args:
        streamlines (Streamlines): The set of filtered streamlines.
        affine (np.ndarray): Affine transformation matrix of the parcellation volume.
        parcellation (np.ndarray): 3D volume with reduced parcellation labels.

    Returns:
        Tuple containing:
            - connectome (np.ndarray): Square matrix (N x N) where each entry [i,j] is the number of streamlines connecting ROI i and j.
            - streamline_mapping (dict[tuple[int, int], Streamlines]): Mapping of ROI pairs to their corresponding streamlines.
    """
    # Compute the full connectivity matrix and mapping from streamlines to ROI pairs
    connectome, streamline_mapping = connectivity_matrix(
        streamlines,
        affine=affine,
        label_volume=parcellation,
        inclusive=True,
        symmetric=True,
        return_mapping=True,
        mapping_as_streamlines=True
    )

    # Return connectome excluding background (row/col 0)
    return connectome[1:, 1:], streamline_mapping



def compute_mean_metric(volume: np.ndarray, streamlines: Streamlines, affine: np.ndarray) -> float:
    """
    Computes the average value of a scalar map (e.g., FA, MD) along a set of streamlines.

    Args:
        volume (np.ndarray): 3D scalar volume.
        streamlines (Streamlines): Streamlines along which to sample.
        affine (np.ndarray): Affine for spatial alignment.

    Returns:
        float: Mean scalar value sampled along streamlines, or 0.0 if no values found.
    """
    values = [val for sl in values_from_volume(volume, streamlines, affine=affine) for val in sl]
    return float(np.mean(values)) if values else 0.0



def multiview_connectivities(
    streamline_mapping: dict[tuple[int, int], Streamlines],
    affine: np.ndarray,
    fa_volume: np.ndarray,
    md_volume: np.ndarray,
    rd_volume: np.ndarray,
    ad_volume: np.ndarray,
    num_rois: int = 102
) -> dict[str, np.ndarray]:
    """
    Computes multiview connectivity matrices (FA, MD, RD, AD, streamline length).

    Args:
        streamline_mapping (dict): Mapping of ROI pairs to streamlines.
        affine (np.ndarray): Affine matrix for mapping to voxel space.
        *_volume (np.ndarray): Scalar diffusion maps (FA, MD, etc.).
        num_rois (int): Number of regions (excluding label 0).

    Returns:
        dict[str, np.ndarray]: Dictionary of connectivity matrices per metric.
    """
    # Initialize all output matrices with zeros
    fa_matrix = np.zeros((num_rois, num_rois), dtype=np.float32)
    md_matrix = np.zeros_like(fa_matrix)
    rd_matrix = np.zeros_like(fa_matrix)
    ad_matrix = np.zeros_like(fa_matrix)
    length_matrix = np.zeros_like(fa_matrix)

    # Loop over all ROI pairs and their connecting streamlines
    for (roi1, roi2), streamlines in streamline_mapping.items():
        # Skip background label (0) and empty streamline sets
        if roi1 == 0 or roi2 == 0 or len(streamlines) == 0:
            continue

        # Convert label IDs to zero-based indices for matrix access
        i, j = roi1 - 1, roi2 - 1

        # Compute and store average scalar values for each metric
        fa_matrix[i, j] = fa_matrix[j, i] = compute_mean_metric(fa_volume, streamlines, affine)
        md_matrix[i, j] = md_matrix[j, i] = compute_mean_metric(md_volume, streamlines, affine)
        rd_matrix[i, j] = rd_matrix[j, i] = compute_mean_metric(rd_volume, streamlines, affine)
        ad_matrix[i, j] = ad_matrix[j, i] = compute_mean_metric(ad_volume, streamlines, affine)

        # Compute and store average streamline length
        length_matrix[i, j] = length_matrix[j, i] = float(np.mean(length(streamlines))) if len(streamlines) > 0 else 0.0

    # Return all matrices in a dictionary
    return {
        'fa': fa_matrix,
        'md': md_matrix,
        'rd': rd_matrix,
        'ad': ad_matrix,
        'length': length_matrix
    }



def plot_multiview_connectomes(views_dict: dict, patient_id: str, session_id: str, save_dir: str = "plots/connectivity_matrices/") -> None:
    """
    Saves multiple connectivity matrices side by side in a single image file using log-scaled color maps.

    Args:
        views_dict (dict): Dictionary where keys are metric names (e.g., 'fa', 'md') 
                           and values are square connectivity matrices (np.ndarray).
        patient_id (str): Identifier for the patient.
        session_id (str): Identifier for the session.
        save_dir (str): Directory to save the plots. Defaults to "plots/connectivity_matrices/".
    """
    views = list(views_dict.keys())

    # Create one subplot per view (metric)
    fig, axes = plt.subplots(1, len(views), figsize=(25, 5))

    for i, view in enumerate(views):
        ax = axes[i]
        ax.set_title(view.replace('_', ' ').upper())
        im = ax.imshow(np.log1p(views_dict[view]), interpolation='nearest', cmap='viridis')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()

    # Save all views into a single file
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{patient_id}_{session_id}_multiview_connectomes.png"))
    plt.close()



def visualize_streamlines(streamlines: Streamlines) -> None:
    """
    Displays a 3D interactive visualization of streamlines using DIPY's built-in viewer.

    Args:
        streamlines (Streamlines): The set of streamlines to visualize.
    """
    # Create a visual actor from the streamline data
    streamline_actor = actor.line(streamlines)

    # Initialize a new 3D rendering scene
    scene = window.Scene()
    scene.add(streamline_actor)

    # Open an interactive visualization window
    window.show(scene)



def run_connectome_pipeline(patient_id: str, session_id: str, base_dir: str = "data/", special_fs_labels: dict = None, fs_idxs2graph_idxs: dict = None, external_logger = None) -> None:
    """
    Full pipeline for building multiview structural connectomes from DWI and parcellation data.

    Args:
        patient_id (str): Subject ID (e.g., "s0001").
        session_id (str): Session ID (e.g., "d0757").
        base_dir (str): Directory where all input data is stored.
    """

    # Setup logger
    if external_logger:
        logger = external_logger
    else:
        logger = setup_debug_logger(patient_id, session_id)

    # Start the timer
    start_time = time.time()

    # Log the start of the pipeline
    logger.info(f"Starting pipeline for patient {patient_id} | session {session_id} at {start_time}")

    # Create the directory structure for the patient and session if it doesn't exist yet
    os.makedirs(f"{base_dir}adj_matrices/{patient_id}_{session_id}", exist_ok=True)

    # Define paths to input files
    paths = {
        "dwi": f"{base_dir}clean_mri/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi_corrected.nii.gz",
        "bval": f"{base_dir}clean_mri/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi.bval",
        "bvec": f"{base_dir}clean_mri/{patient_id}_{session_id}/{patient_id}_{session_id}_dwi_rotated.bvec",
        "smri": f"{base_dir}clean_mri/{patient_id}_{session_id}/{patient_id}_{session_id}_smri_downsampled.nii.gz",
        "smri_parc": f"{base_dir}clean_mri/{patient_id}_{session_id}/{patient_id}_{session_id}_parc_downsampled.nii.gz",
    }

    # Load parcellation mappings for FreeSurfer to reduced labels
    if fs_idxs2graph_idxs is None:
        fs_idxs2graph_idxs = load_parcellation_mappings()['fs_idxs2graph_idxs']

    # Unpack specific FreeSurfer labels for filtering
    if special_fs_labels is None:
        special_fs_labels = load_special_fs_labels()
    WM_LABELS = special_fs_labels["wm_labels"]
    FLUID_LABELS = special_fs_labels["fluid_labels"]

    try:
        logger.info("Loading data")
        dwi_data, dwi_affine, gtab, fs_parcellation = load_data(paths)
    except Exception:
        logger.exception("Failed loading data")
        raise

    try:
        logger.info("Creating white matter masks")
        wm_mask, dilated_wm_mask = create_wm_mask(fs_parcellation, WM_LABELS)
    except Exception:
        logger.exception("Failed creating white matter masks")
        raise

    try:
        logger.info("Fitting DTI model")
        dti_fit, fa_data, md_data, rd_data, ad_data = fit_dti_model(dwi_data, gtab, dilated_wm_mask)
    except Exception:
        logger.exception("Failed fitting DTI model")
        raise

    try:
        logger.info("Generating streamlines")
        streamlines = generate_streamlines(dti_fit, dilated_wm_mask, fa_data, dwi_affine)
    except Exception:
        logger.exception("Failed generating streamlines")
        raise

    try:
        logger.info("Filtering streamlines by length")
        streamlines = filter_by_length(streamlines, min_length=20, max_length=300)
    except Exception:
        logger.exception("Failed filtering streamlines by length")
        raise

    try:
        logger.info("Performing anatomical filtering")
        streamlines = filter_anatomical(streamlines, fs_parcellation, dwi_affine, FLUID_LABELS)
    except Exception:
        logger.exception("Failed anatomical filtering")
        raise

    try:
        logger.info("Filtering outliers")
        streamlines = filter_outliers(streamlines, min_cluster_size=5, threshold=10.0)
    except Exception:
        logger.exception("Failed filtering outliers")
        raise

    try:
        logger.info("Reducing parcellation")
        reduced_parcellation = preprocess_parcellation(fs_parcellation, fs_idxs2graph_idxs)
    except Exception:
        logger.exception("Failed reducing parcellation")
        raise

    try:
        logger.info("Building streamline count connectome")
        connectome, streamline_mapping = sl_count_connectivity(streamlines, dwi_affine, reduced_parcellation)
    except Exception:
        logger.exception("Failed building streamline count connectome")
        raise

    try:
        logger.info("Computing multiview connectomes")
        multiview_connectomes = multiview_connectivities(
            streamline_mapping, dwi_affine, fa_data, md_data, rd_data, ad_data,
            num_rois=reduced_parcellation.max()
        )
        multiview_connectomes['count'] = connectome
    except Exception:
        logger.exception("Failed computing multiview connectomes")
        raise

    try:
        logger.info("Plotting multiview connectomes")
        plot_multiview_connectomes(multiview_connectomes, patient_id=patient_id, session_id=session_id)
    except Exception:
        logger.exception("Failed plotting multiview connectomes")
        raise

    try:
        logger.info("Saving adjacency matrices to intermediate data")
        np.savez_compressed(
            f"{base_dir}adj_matrices/{patient_id}_{session_id}_As.npz",
            **multiview_connectomes
        )
    except Exception:
        logger.exception("Failed saving adjacency matrices to intermediate data")
        raise

    logger.info(f"Pipeline completed for patient {patient_id} | session {session_id} in {time.time() - start_time:.2f} seconds.")



if __name__ == "__main__":
    run_connectome_pipeline(patient_id="0001", session_id="0757")