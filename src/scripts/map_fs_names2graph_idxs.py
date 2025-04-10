from src.utils.json_loading import load_parcellation_mappings
import json

fs_names2graph_idxs = {}

mappings = load_parcellation_mappings()

fs_idxs2graph_idxs = mappings['fs_idxs2graph_idxs']
fs_idxs2fs_names = mappings['fs_idxs2fs_names']

for fs_idx, graph_idx in fs_idxs2graph_idxs.items():
    fs_name = fs_idxs2fs_names[fs_idx]
    fs_names2graph_idxs[fs_name] = graph_idx

    with open('fs_names2graph_idxs.json', 'w') as f:
        json.dump(fs_names2graph_idxs, f, indent=4)

print("fs_names2graph_idxs.json file created successfully.")