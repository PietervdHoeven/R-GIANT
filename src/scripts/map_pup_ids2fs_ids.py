import pandas as pd
import json

# Load the CSV files without headers
pup_ids_df = pd.read_csv('C:/Users/piete/Documents/Projects/R-GIANT/data/pup_ids.csv', header=None, names=['pup_id'])
fs_ids_df = pd.read_csv('C:/Users/piete/Documents/Projects/R-GIANT/data/fs_ids.csv', header=None, names=['fs_id'])

# Function to clean and extract participant ID and session ID
def clean_ids(full_id):
    parts = full_id.split('_')
    participant_id = parts[0].replace('OAS3', '')  # e.g. 'OAS30001' -> '0001'
    session_id = parts[-1].replace('d', '')        # e.g. 'd2438' -> '2438'
    return participant_id, session_id

# Apply the cleaning function
cleaned_pup_ids = pup_ids_df['pup_id'].apply(clean_ids)
cleaned_fs_ids = fs_ids_df['fs_id'].apply(clean_ids)

# Create the dictionary
pup_ids2mr_ids = dict(zip(cleaned_pup_ids, cleaned_fs_ids))

# Convert tuple keys and values to string format
serializable_dict = {
    f"{k[0]}_{k[1]}": f"{v[0]}_{v[1]}"
    for k, v in pup_ids2mr_ids.items()
}

# Save to JSON
with open('pup_ids2mr_ids.json', 'w') as f:
    json.dump(serializable_dict, f, indent=2)
