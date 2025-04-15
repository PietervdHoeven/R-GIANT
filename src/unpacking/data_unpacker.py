import os
import shutil
import json
import argparse
import tarfile
import zipfile
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



def unpack_pup(packed_dir, target_dir, pup_ids2mr_ids, remove=False):
    for session in tqdm(os.listdir(packed_dir)):
        session_path = os.path.join(packed_dir, session)

        is_zip = session.endswith(".zip")
        is_tar = session.endswith(".tar.gz")
        is_dir = os.path.isdir(session_path)

        # Determine session name (used to retrieve IDs)
        if is_zip:
            session_name = session[:-4]
        elif is_tar:
            session_name = session[:-7]
        elif is_dir:
            session_name = session
        else:
            print(f"Skipping unsupported file: {session}")
            continue

        try:
            pup_p_id, pup_s_id = get_ids(session_name, is_pup=True)
            p_id, s_id = pup_ids2mr_ids[(pup_p_id, pup_s_id)]
        except KeyError:
            print(f"No matching MR session found for {session_name}")
            continue

        new_session_dir = os.path.join(target_dir, f"{p_id}_{s_id}")
        os.makedirs(new_session_dir, exist_ok=True)

        # --- CASE 1: ZIP file ---
        if is_zip:
            with zipfile.ZipFile(session_path, 'r') as archive:
                for member in archive.namelist():
                    if "pet_proc" in member and member.endswith("RSF.suvr"):
                        with archive.open(member) as src_file, open(os.path.join(new_session_dir, os.path.basename(member)), 'wb') as out_file:
                            shutil.copyfileobj(src_file, out_file)

        # --- CASE 2: TAR.GZ file ---
        elif is_tar:
            with tarfile.open(session_path, 'r:gz') as archive:
                for member in archive.getmembers():
                    if "pet_proc" in member.name and member.name.endswith("RSF.suvr"):
                        extracted = archive.extractfile(member)
                        if extracted:
                            with open(os.path.join(new_session_dir, os.path.basename(member.name)), 'wb') as out_file:
                                shutil.copyfileobj(extracted, out_file)

        # --- CASE 3: Already-unpacked folder ---
        elif is_dir:
            pet_proc_dir = None
            for root, dirs, files in os.walk(session_path):
                if os.path.basename(root) == "pet_proc":
                    pet_proc_dir = root
                    break

            if pet_proc_dir:
                for file_name in os.listdir(pet_proc_dir):
                    if file_name.endswith("RSF.suvr"):
                        src_path = os.path.join(pet_proc_dir, file_name)
                        dst_path = os.path.join(new_session_dir, file_name)
                        shutil.copy(src_path, dst_path)
            else:
                print(f"'pet_proc' folder not found in {session}")

        if remove and (is_zip or is_tar):
            os.remove(session_path)


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
    