# rgiant/data/dataset.py

import os
import glob
import torch
from torch_geometric.data import InMemoryDataset

class ConnectomeDataset(InMemoryDataset):
    """
    InMemoryDataset for connectome graphs.
    
    - On first instantiation: scans `root/*.pt`, applies pre_transform (if any),
      collates them into one big `data, slices` tuple, and saves to `processed/all_data.pt`.
    - On subsequent loads: skips scanning and just loads the collated file.
    - __len__ / __getitem__ are inherited and use `self.data` + `self.slices`.
    """

    def __init__(
            self,
            root: str,
            processed_filename: str = "all_graphs.pt",
            transform=None,
            pre_transform=None
            ):
        """
        Args:
            root: directory containing your `*.pt` graph files (e.g. "data/processed")
            transform: run on each Data object at __getitem__ (for on-the-fly aug, if needed)
            pre_transform: run once on each Data in `process()` (e.g. fold-specific normalization)
        """
        self.processed_filename = processed_filename
        super().__init__(root, transform, pre_transform)
        # load the collated dataset into memory
        path = os.path.join(self.processed_dir, self.processed_filename)
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        # name of the single collated file
        return [self.processed_filename]

    def process(self):
        """
        1) Scan self.root for all `*_G.pt` graphs
        2) torch.load each one into a Data object
        3) Optionally apply self.pre_transform (e.g. normalization)
        4) Collate into (data, slices) and torch.save to processed/all_data.pt
        """
        graph_list = []
        # adjust the pattern if your PT files live in subfolders
        for pt_path in glob.glob(os.path.join(self.raw_dir, "*_G.pt")):
            graph = torch.load(pt_path)
            if self.pre_transform:
                graph = self.pre_transform(graph)
            graph_list.append(graph)

        # collate into InMemoryDataset format
        data, slices = self.collate(graph_list)

        # ensure the processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_filename))
