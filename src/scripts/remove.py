import os
import shutil
import pandas as pd

# Set your paths
csv_path = "mr_ids.csv"         # Your .csv file
target_dir = "packed_mr"       # Folder containing many subdirectories

# Read the list of valid folder names (assuming 1 column with folder names)
valid_names = pd.read_csv(csv_path, header=None).iloc[:, 0].tolist()

# Create a dictionary with valid names as keys and 0 as values
valid_names_dict = {name: 0 for name in valid_names}

# Loop through subdirectories
for folder in os.listdir(target_dir):
    valid_names_dict[folder] = 1

# Find missing names
missing_names = [name for name, count in valid_names_dict.items() if count == 0]

# Save missing names to a CSV file
missing_names_df = pd.DataFrame(missing_names, columns=["Missing Names"])
missing_names_df.to_csv("missing_mr_ids.csv", index=False)


        

# Loop through subdirectories
# for folder in os.listdir(target_dir):
#     if folder in valid_names: print("folder")
#     print(folder)
#     folder_path = os.path.join(target_dir, folder)
#     if os.path.isdir(folder_path) and folder in valid_names:
#         print(f"Removing: {folder}")
#         shutil.move(folder_path,
#                 os.path.join(os.path.expanduser("~/Desktop/packed_mr"), folder))