import numpy as np
from utils.mappings import load_parcellation_mappings

def parse_aseg_stats(aseg_path):
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
                print(line)
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
            print(fields)
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

    return (etiv, rows)



def parse_aparc_stats(lh_parc_path, rh_parc_path):
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
                print(fields)
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



def extract_node_features(
        patient_id: str,
        session_id: str,
):
    # Build empty feature matrix with shape (n_nodes, n_features)
    n_nodes = len(load_parcellation_mappings()['fs2reduced'])
    n_features = 8 # is_sub-ctx, is_ctx, is_fluid, Volume_mm3, SurfArea, GrayVol, ThickAvg, ThickStd, MeanCurv
    X = np.zeros(n_nodes, n_features)

    # Extract sub-cortical features from aseg.stats
    aseg_stats = parse_aseg_stats(




# Example usage:
if __name__ == "__main__":
    aseg_file = "C:/Users/piete/Documents/Projects/R-GIANT/packed_data/OAS30001_MR_d0757/OAS30001_Freesurfer53_d0757/DATA/OAS30001_MR_d0757/stats/aseg.stats"  # update with your file path
    etiv, seg_rows = parse_aseg_stats(aseg_file)
    parc_rows = parse_aparc_stats(
        "C:/Users/piete/Documents/Projects/R-GIANT/packed_data/OAS30001_MR_d0757/OAS30001_Freesurfer53_d0757/DATA/OAS30001_MR_d0757/stats/lh.aparc.stats",
        "C:/Users/piete/Documents/Projects/R-GIANT/packed_data/OAS30001_MR_d0757/OAS30001_Freesurfer53_d0757/DATA/OAS30001_MR_d0757/stats/rh.aparc.stats"
    )
    print("eTIV:", etiv)
    for row in seg_rows:
        print(row)
    for row in parc_rows:
        print(row)
