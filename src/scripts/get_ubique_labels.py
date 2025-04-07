import nibabel as nib
import numpy as np

# Path to the .mgz file
file_path = 'C:/Users/piete/Documents/Projects/R-GIANT/data/0001_0757_aparc+aseg.mgz'

# Load the .mgz file
mgz_data = nib.load(file_path)

# Get the data array
data = mgz_data.get_fdata()

# Find unique labels
unique_labels = np.unique(data)

# Print the number of unique labels
print(f"Number of unique labels: {len(unique_labels)}")