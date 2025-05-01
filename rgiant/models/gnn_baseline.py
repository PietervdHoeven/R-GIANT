import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class GNNClassifier(torch.nn.Module):
    def __init__(self, in_node_feats=11, in_edge_feats=7, hidden=64, out_hidden=32, num_classes=2):
        super().__init__()

        # GraphConv layer 1: input → hidden
        self.conv1 = GCNConv(in_node_feats, hidden)

        # GraphConv layer 2: hidden → hidden
        self.conv2 = GCNConv(hidden, hidden)

        # MLP classifier head
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, out_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(out_hidden, num_classes)     # Binary output
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # GCNConv ignores edge_attr unless you use a model that supports it
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pool over nodes to get graph-level embedding
        x = global_mean_pool(x, batch)   # shape: [batch_size, hidden]

        out = self.mlp(x)                # shape: [batch_size, num_classes]
        return out
