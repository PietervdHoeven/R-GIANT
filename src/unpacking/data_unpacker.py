import os
import shutil
import json
import argparse
from tqdm import tqdm

def unpack_mr(
        packed_dir, target_dir, remove: bool = False
):
    """
        Unpacks magnetic resonance (MR) session directories from a packed directory 
    into a target directory, organizing them by patient and session IDs.
    Args:
        packed_dir (str): The path to the directory containing packed session 
            directories. Each session directory name should include patient 
            and session IDs.
        target_dir (str): The path to the directory where unpacked files 
            will be stored. Subdirectories will be created for each session 
            using the format "{patient_id}_{session_id}".
        remove (bool, optional): If True, the original session directories 
            in the packed directory will be deleted after unpacking. Defaults 
            to False.
    Raises:
        FileNotFoundError: If the `packed_dir` does not exist or if any 
            required file is missing during the unpacking process.
        OSError: If there is an issue creating directories or copying files.
    Notes:
        - Files with a `.json` extension are ignored during the unpacking 
            process.
        - The function assumes that the `get_ids` function is available and 
            correctly extracts patient and session IDs from the session 
            directory name.
    """
    for session_dir in os.listdir(packed_dir):
        print(session_dir)
        # get the patient and session IDs from the directory name
        p_id, s_id = get_ids(session_dir)
        # create the target directory if it doesn't exist
        os.makedirs(
            os.path.join(target_dir, f"{p_id}_{s_id}"), 
            exist_ok=True
            )
        
        dwi_dir = sorted(os.listdir(os.path.join(packed_dir, session_dir)))[-1]

        for file_name in os.listdir(os.path.join(packed_dir, session_dir, dwi_dir)):
            # move all files except .json to the target directory
            _, ext = os.path.splitext(file_name)
            if ext == ".json":
                continue
            if ext == ".gz":
                ext = ".nii.gz"
            src_path = os.path.join(packed_dir, session_dir, dwi_dir, file_name)
            dst_path = os.path.join(target_dir, f"{p_id}_{s_id}", f"{p_id}_{s_id}"+ext)
            shutil.copy(src_path, dst_path)
        if remove:
            # remove the packed directory
            shutil.rmtree(os.path.join(packed_dir, session_dir))



def unpack_fs(
        packed_dir, target_dir, remove: bool = False
):
    """
    Unpacks session directories from a packed directory into a target directory.

    This function processes each session directory within the `packed_dir`, extracts
    specific files, and organizes them into a new directory structure in the `target_dir`.
    Optionally, it can remove the original packed directories after unpacking.

    Args:
        packed_dir (str): Path to the directory containing packed session directories.
        target_dir (str): Path to the directory where unpacked session directories will be created.
        remove (bool, optional): If True, the original packed directories will be deleted 
                                    after unpacking. Defaults to False.

    Raises:
        FileNotFoundError: If any of the required files to move are missing in the packed directory.
        OSError: If there are issues creating directories or copying files.

    Notes:
        - Each session directory in `packed_dir` is expected to have a specific structure
            containing subdirectories like "mri" and "stats".
        - The function extracts specific files such as "aparc+aseg.mgz", "brain.mgz",
            and various stats files, and places them in a new directory named using
    """
    for session_dir in tqdm(os.listdir(packed_dir)):
        # Extract the patient and session IDs from the directory name
        p_id, s_id = get_ids(session_dir)
        
        # Create a new session directory in the target directory
        new_session_dir = os.path.join(target_dir, f"{p_id}_{s_id}")
        os.makedirs(
            new_session_dir,
            exist_ok=True
        )
        
        # Define the paths of the files to be moved
        prefix = os.path.join(packed_dir, session_dir)
        files_to_move = [
            os.path.join(prefix, "mri", "aparc+aseg.mgz"),  # MRI segmentation file
            os.path.join(prefix, "mri", "brain.mgz"),       # Brain image file
            os.path.join(prefix, "stats", "aseg.stats"),    # Subcortical stats file
            os.path.join(prefix, "stats", "lh.aparc.stats"), # Left hemisphere stats file
            os.path.join(prefix, "stats", "rh.aparc.stats") # Right hemisphere stats file
        ]
        
        # Copy each file to the new session directory
        for src_path in files_to_move:
            file_name = os.path.basename(src_path)
            dst_path = os.path.join(new_session_dir, file_name)
            shutil.copy(src_path, dst_path)
        
        if remove:
            # Remove the original packed directory after processing
            shutil.rmtree(os.path.join(packed_dir, session_dir))



def unpack_pup(
        packed_dir, target_dir, pup_ids2mr_ids: dict, remove: bool = False
):
    """
        Unpacks and processes PUP (PET Unified Pipeline) session directories, translating their IDs to 
    corresponding MR session IDs, and moves specific files to a target directory.
    Args:
        packed_dir (str): Path to the directory containing packed PUP session directories.
        target_dir (str): Path to the directory where processed session directories will be created.
        pup_ids2mr_ids (dict): A mapping of PUP session IDs (tuple of participant ID and session ID) 
        to MR session IDs (tuple of participant ID and session ID).
        remove (bool, optional): If True, removes the original packed directories after processing. 
        Defaults to False.
    Raises:
        KeyError: If a PUP session ID from the directory name does not exist in the `pup_ids2mr_ids` mapping.
        FileNotFoundError: If the expected `pet_proc` directory or required files are missing.
    Notes:
        - The function handles both zipped and unzipped session directories.
        - Only files ending with "RSF.suvr" from the deepest `pet_proc` directory are moved to the target directory.
        - The target directory structure is created based on the translated MR session IDs.
    """
    for session_dir in tqdm(os.listdir(packed_dir)):
        # Unzip the session directory if it is a zip file
        if session_dir.endswith(".zip"):
            zip_path = os.path.join(packed_dir, session_dir)
            extract_path = os.path.join(packed_dir, session_dir[:-4])  # Remove ".zip" extension
            shutil.unpack_archive(zip_path, extract_path)
            session_dir = session_dir[:-4]  # Update session_dir to the extracted folder name

        # Get the ids from the dir name, translate to matched MR session and create target session directory
        pup_p_id, pup_s_id = get_ids(session_dir, is_pup=True)
        p_id, s_id = pup_ids2mr_ids[(pup_p_id, pup_s_id)]
        new_session_dir = os.path.join(target_dir, f"{p_id}_{s_id}")
        os.makedirs(
            new_session_dir,
            exist_ok=True
        )

        # walk through all dirs untill you find a folder named pet_proc
        pet_proc_dir = None
        for root, dirs, files in os.walk(os.path.join(packed_dir, session_dir), topdown = False):
            if "pet_proc" in dirs:
                pet_proc_dir = os.path.join(root, "pet_proc")
                break

        if not pet_proc_dir:
            raise FileNotFoundError(f"'pet_proc' directory not found in {session_dir}")

        # Move only the files that end with RSF.suvr from the deepest directory in session_dir named pet_proc
        for file_name in os.listdir(pet_proc_dir):
            if file_name.endswith("RSF.suvr"):
                src_path = os.path.join(pet_proc_dir, file_name)
                dst_path = os.path.join(new_session_dir, file_name)
                shutil.copy(src_path, dst_path)



def load_pup2mr_json(path: str = 'pup_ids2mr_ids.json'):
    with open(path, 'r') as f:
        loaded_dict = json.load(f)

    # Convert keys and values back to tuples
    pup_ids2mr_ids = {
        tuple(k.split('_')): tuple(v.split('_'))
        for k, v in loaded_dict.items()
    }

    return pup_ids2mr_ids



def get_ids(session_dir, is_pup: bool = False):
    """
    Get the patient and session IDs from the directory name.
    The directory name is expected to be in the format "OAS3{p_id}_<type>_d{s_id}",
    where {p_id} is the patient ID and {s_id} is the session ID.
    If is_pup is True, s_id is extracted from parts[3][1:] instead.
    """
    # split the directory name by "_"
    parts = session_dir.split("_")
    # extract the patient ID
    p_id = parts[0][4:]  # Remove "OAS3" prefix to get the patient ID
    # extract the session ID based on is_pup
    if is_pup:
        s_id = parts[3][1:]  # Remove "d" prefix to get the session ID
    else:
        s_id = parts[2][1:]  # Remove "d" prefix to get the session ID
    return p_id, s_id



if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Unpack session directories.")
    parser.add_argument("type", choices=["pup", "mr", "fs"], help="Type of unpacking to perform.")
    parser.add_argument("packed_dir", help="Path to the packed directory.")
    parser.add_argument("target_dir", help="Path to the target directory.")
    parser.add_argument("--json_path", help="Path to the PUP to MR JSON mapping file (required if type is 'pup').")
    parser.add_argument("--remove", help="Remove the original packed directories after unpacking.")

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate unpacker based on the type
    if args.type == "pup":
        if not args.json_path:
            raise ValueError("The --json_path argument is required when type is 'pup'.")
        pup_ids2mr_ids = load_pup2mr_json(args.json_path)
        unpack_pup(args.packed_dir, args.target_dir, pup_ids2mr_ids, args.remove)
    elif args.type == "mr":
        unpack_mr(args.packed_dir, args.target_dir, args.remove)
    elif args.type == "fs":
        unpack_fs(args.packed_dir, args.target_dir, args.remove)
    