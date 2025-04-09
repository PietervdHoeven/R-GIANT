import numpy as np
from src.preprocessing.utils.json_loading import load_parcellation_mappings, load_special_fs_labels

def parse_aseg_stats(aseg_path: str):
    """
    Parse a FreeSurfer aseg.stats file.
    
    Parameters:
        aseg_path (str): Path to the aseg.stats file.
        
    Returns:
        tuple: (etiv, rows)
            etiv (float): The estimated Total Intracranial Volume (eTIV) extracted from the header.
            rows (list): A list of dictionaries, one for each data row in the file.
                        Each dictionary contains keys:
                        "Index", "SegId", "StructName", "NumVert", and "Volume_mm3".
    """
    etiv = None
    rows = []
    
    with open(aseg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Look for the eTIV line; this line starts with "MeasureEstimatedTotalIntracranialVol"
            if line.startswith("# Measure EstimatedTotalIntraCranialVol"):
                parts = line.split(',')
                if len(parts) > 1:
                    try:
                        etiv = float(parts[-2].strip())
                    except ValueError:
                        etiv = None
                continue
            # Skip all comment lines that start with "#"
            if line.startswith("#"):
                continue
            # Split the line into fields
            fields = line.split()
            # We expect at least 5 fields per data row.
            if len(fields) < 5:
                continue
            # Index SegId #NVoxels Volume_mm3 StructName #normMean #normStdDev #normMin #normMax #normRange
            row_dict = {
                "Index": int(fields[0]),
                "SegId": fields[1],  # Keep as string for consistency with mappings
                "Volume_mm3": float(fields[3]),
                "StructName": fields[4],  # Keep as string for consistency with mappings
            }
            rows.append(row_dict)

    return (rows, etiv)



def encode_sub_ctx_features(X: np.array, aseg_stats: list[dict], eTIV: float, fs2reduced: dict, fluid_labels: list) -> np.array:
    """
    Encode sub-cortical features from aseg.stats into the node feature matrix X.
    
    Parameters:
        X (np.array): The node feature matrix to be updated.
        aseg_stats (list): List of dictionaries containing aseg stats.
        eTIV (float): Estimated Total Intracranial Volume.
        mapping (dict): Mapping from FreeSurfer segment IDs to reduced IDs.
        
    Returns:
        np.array: Updated node feature matrix X with sub-cortical features.
    """
    for row in aseg_stats:
        fs_label = row["SegId"]
        if fs_label in fs2reduced:
            idx = fs2reduced[fs_label] - 1 # -1 for 0-based indexing
            if int(fs_label) in fluid_labels:
                X[idx, 2] = 1  # index 2=True iff sub-cortical structure is fluid
            else:
                X[idx, 0] = 1  # index 0=True iff sub-ctx-struct is normal
            X[idx, 3] = row["Volume_mm3"] / eTIV  # Volume_mm3 normalized by eTIV
    return X



def parse_aparc_stats(lh_parc_path: str, rh_parc_path: str) -> list[dict]:
    """
    Parse a FreeSurfer aseg.stats file.
    
    Parameters:
        aseg_path (str): Path to the aseg.stats file.
        
    Returns:
        rows (list): A list of dictionaries, one for each data row in the file.
                    Each dictionary contains keys:
                    'StructName', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv'.
    """
    rows = []
    for i, parc_path in enumerate([lh_parc_path, rh_parc_path]):
        with open(parc_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Skip all comment lines that start with "#"
                if line.startswith("#"):
                    continue
                # Split the line into fields
                fields = line.split()
                # We expect at least 5 fields per data row.
                if len(fields) < 5:
                    continue
                # StructName #NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv #GausCurv #FoldInd #CurvInd
                row_dict = {
                    "StructName": "ctx-lh-"+fields[0] if i == 0 else "ctx-rh-"+fields[0],
                    "SurfArea": int(fields[2]),
                    "GrayVol": int(fields[3]),
                    "ThickAvg": float(fields[4]),
                    "ThickStd": float(fields[5]),
                    "MeanCurv": float(fields[6]),
                }
                rows.append(row_dict)

    return rows



def encode_ctx_features(X: np.array, aparc_stats: list[dict], names2reduced: dict) -> np.array:
    """
    Encode cortical features from aparc.stats into the node feature matrix X.
    
    Parameters:
        X (np.array): The node feature matrix to be updated.
        aparc_stats (list): List of dictionaries containing aparc stats.
        names2reduced (dict): Mapping from FreeSurfer structure names to reduced IDs.
        
    Returns:
        np.array: Updated node feature matrix X with cortical features.
    """
    for row in aparc_stats:
        fs_name = row["StructName"]
        if fs_name in names2reduced:
            idx = names2reduced[fs_name] - 1 # -1 for 0-based indexing
            X[idx, 1] = 1 # index 1=True iff it is a cortical structure
            X[idx, 4] = row["SurfArea"]
            X[idx, 5] = row["GrayVol"]
            X[idx, 6] = row["ThickAvg"]
            X[idx, 7] = row["ThickStd"]
            X[idx, 8] = row["MeanCurv"]
    return X


def extract_node_features(
        patient_id: str,
        session_id: str,
        base_dir: str = "C:/Users/piete/Documents/Projects/R-GIANT/data/",
        fs2reduced: dict = None,
        ctx_names2reduced: dict = None,
        fluid_labels: dict = None
):
    """
    Extract node features from FreeSurfer segmentation stats and save them to a numpy file.

    This function processes FreeSurfer segmentation statistics to extract features for brain regions
    and encodes them into a node feature matrix. The features include sub-cortical and cortical
    properties such as volume, surface area, thickness, and curvature. The resulting matrix is saved
    as a `.npy` file for further analysis.

    Parameters:
        patient_id (str): The ID of the patient (e.g., "0001").
        session_id (str): The session ID for the patient (e.g., "0757").
        base_dir (str): The base directory where the data is stored. Defaults to
                        "C:/Users/piete/Documents/Projects/R-GIANT/data/".
        fs2reduced (dict): A mapping from FreeSurfer segment IDs to reduced IDs for sub-cortical regions.
        ctx_names2reduced (dict): A mapping from FreeSurfer cortical structure names to reduced IDs.
        fluid_labels (dict): A dictionary of fluid-related labels to identify fluid compartments.

    Returns:
        None: The function saves the node feature matrix as a `.npy` file in the intermediate directory.

    Outputs:
        - A numpy file containing the node feature matrix is saved to:
          `{base_dir}/intermediate/{patient_id}_{session_id}_node_features.npy`.
        - The matrix is printed to the console for verification.

    Node Feature Matrix (X):
        - Shape: (n_nodes, 9)
        - Columns:
            0: is_sub_ctx (1 if sub-cortical structure, 0 otherwise)
            1: is_ctx (1 if cortical structure, 0 otherwise)
            2: is_fluid (1 if fluid compartment, 0 otherwise)
            3: Volume_mm3 (normalized by eTIV)
            4: SurfArea (surface area)
            5: GrayVol (gray matter volume)
            6: ThickAvg (average thickness)
            7: ThickStd (thickness standard deviation)
            8: MeanCurv (mean curvature)

    """
    # Build empty feature matrix with shape (n_nodes, n_features)
    n_nodes = len(fs2reduced)
    n_features = 9 # is_sub-ctx, is_ctx, is_fluid, Volume_mm3, SurfArea, GrayVol, ThickAvg, ThickStd, MeanCurv
    X = np.zeros((n_nodes, n_features))

    # Load parcellation label mappings
    mappings = load_parcellation_mappings()

    # Extract asegs stats and estimated Total Intracranial Volume (eTIV) by parsing aseg.stats
    aseg_stats, eTIV = parse_aseg_stats(aseg_path=f"{base_dir}raw/{patient_id}_{session_id}/{patient_id}_{session_id}_aseg.stats")

    # Encode the sub cortical features from the parsed siub cortical segmentation stats into the node feature matrix X
    X = encode_sub_ctx_features(X, aseg_stats, eTIV, fs2reduced=mappings["fs2reduced"], fluid_labels=fluid_labels)

    # Extract aparc stats by parsing lh.aparc.stats and rh.aparc.stats
    aparc_stats = parse_aparc_stats(
        lh_parc_path=f"{base_dir}raw/{patient_id}_{session_id}/{patient_id}_{session_id}_lh.aparc.stats",
        rh_parc_path=f"{base_dir}raw/{patient_id}_{session_id}/{patient_id}_{session_id}_rh.aparc.stats"
    )

    # Encode the cortical features from the parsed cortical parcellation stats into the node feature matrix X
    X = encode_ctx_features(X, aparc_stats, names2reduced=ctx_names2reduced)

    # Save the node feature matrix X to a numpy file
    np.save(f"{base_dir}intermediate/{patient_id}_{session_id}_node_features.npy", X)
    np.set_printoptions(precision=4, suppress=True)
    for i in range(X.shape[0]):
        print(f"{X[i]}")



# Example usage:
if __name__ == "__main__":
    # Load mappings and special labels
    mappings = load_parcellation_mappings()
    special_labels = load_special_fs_labels()
    # Extract node features
    extract_node_features(
        patient_id="0001",
        session_id="0757",
        base_dir="C:/Users/piete/Documents/Projects/R-GIANT/data/",
        fs2reduced=mappings["fs2reduced"],
        ctx_names2reduced=mappings["ctx_names2reduced"],
        fluid_labels=special_labels["fluid_labels"]
    )

