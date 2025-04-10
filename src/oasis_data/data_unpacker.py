import os
import shutil

def unpack_mr(
        packed_dir, target_dir, remove: bool = False
):
    for session_dir in os.listdir(packed_dir):
        # get the patient and session IDs from the directory name
        p_id, s_id = get_ids(session_dir)
        # create the target directory if it doesn't exist
        os.makedirs(
            os.path.join(target_dir, f"{p_id}_{s_id}"), 
            exist_ok=True
            )
        for file_name in os.listdir(os.path.join(packed_dir, session_dir)):
            # move all files except .json to the target directory
            _, ext = os.path.splitext(file_name)
            if ext == ".json":
                continue
            src_path = os.path.join(packed_dir, session_dir, file_name)
            dst_path = os.path.join(target_dir, f"{p_id}_{s_id}", file_name)
            shutil.copy(src_path, dst_path)
        if remove:
            # remove the packed directory
            shutil.rmtree(os.path.join(packed_dir, session_dir))



def unpack_fs(
        packed_dir, target_dir, remove: bool = False
):
    for session_dir in os.listdirs(packed_dir):
        p_id, s_id = get_ids(session_dir)
        new_session_dir = os.path.join(target_dir, f"{p_id}_{s_id}")
        os.makedirs(
            new_session_dir,
            exist_ok=True
        )
        prefix = os.path.join(packed_dir, session_dir)
        files_to_move = [
            os.path.join(prefix, "mri", "aparc+aseg.mgz"),
            os.path.join(prefix, "mri", "brain.mgz"),
            os.path.join(prefix, "stats", "aseg.stats"),
            os.path.join(prefix, "stats", "lh-aparc.stats"),
            os.path.join(prefix, "stats", "rh-aparc.stats")
        ]
        for src_path in files_to_move:
            file_name = os.path.basename(src_path)
            dst_path = os.path.join(new_session_dir, file_name)
            shutil.copy(src_path, dst_path)
        if remove:
            # remove the packed directory
            shutil.rmtree(os.path.join(packed_dir, session_dir))



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