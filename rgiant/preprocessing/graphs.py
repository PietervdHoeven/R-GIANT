import torch
from torch_geometric.data import Data
import numpy as np

def adjacency_to_edge_list_torch(A_r: torch.Tensor, relation_type: int):
    """
    Converts one adjacency matrix to edge_index, edge_type, and edge_weight
    using torch operations.

    - `edge_index`: A 2 x num_of_edges tensor where each column represents an edge.
      The first row contains the source node indices, and the second row contains
      the target node indices for all edges in the graph.

    - `edge_type`: A 1D tensor of length num_of_edges, where each element specifies
      the type of the corresponding edge. In this case, all edges in the adjacency
      matrix A^r are assigned the same relation type `relation_type`.

    - `edge_weight`: A 1D tensor of length num_of_edges, where each element contains
      the weight associated with the corresponding edge a_{ij}^r in the adjacency
      matrix A^r.
    """
    # 1. Get all (i, j) for non-zero entries
    edge_indices = torch.nonzero(A_r, as_tuple=False)  # shape [E, 2]
    
    # 2. Get edge weights
    edge_weights = A_r[edge_indices[:, 0], edge_indices[:, 1]]  # shape [E]

    # 3. Transpose edge_indices to shape [2, E]
    edge_index = edge_indices.t().contiguous()

    # 4. Create edge type tensor: one value per edge
    edge_type = torch.full((edge_index.size(1),), relation_type, dtype=torch.long)

    return edge_index, edge_type, edge_weights



def build_pyg_data(patient_id, session_id, data_dir, label=None):
    """
    Constructs a PyTorch Geometric `Data` object from adjacency matrices, node features, 
    and an optional label.
    Args:
        As (dict): A dictionary where keys are relation types (e.g., integers) and values 
            are adjacency matrices (2D numpy arrays or similar) representing the graph 
            structure for each relation type. Each adjacency matrix should have shape 
            `[num_nodes, num_nodes]`.
        X (numpy.ndarray or similar): A 2D array of node features with shape `[num_nodes, num_features]`.
        label (int, optional): An optional integer label for the graph. Defaults to `None`.
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric `Data` object containing:
            - `x` (torch.Tensor): Node feature matrix of shape `[num_nodes, num_features]`.
            - `edge_index` (torch.Tensor): Edge indices in COO format of shape `[2, E_total]`, 
              where `E_total` is the total number of edges across all relation types.
            - `edge_type` (torch.Tensor): Edge type identifiers of shape `[E_total]`.
            - `edge_weight` (torch.Tensor): Edge weights of shape `[E_total]`.
            - `y` (torch.Tensor, optional): Graph label tensor of shape `[1]` if `label` is provided.
    Notes:
        - The adjacency matrices in `As` are converted to edge lists using the 
          `adjacency_to_edge_list_torch` function, which is assumed to return edge indices, 
          edge types, and edge weights.
        - All inputs are converted to PyTorch tensors with appropriate data types.
    """
    As_path = f"{data_dir}/matrices/{patient_id}_{session_id}_As.npz"
    X_path = f"{data_dir}/matrices/{patient_id}_{session_id}_X.npy"

    As = np.load(As_path)  
    X = np.load(X_path)

    all_edge_indices = []
    all_edge_types = []
    all_edge_weights = []

    for r, A in enumerate(As.values()):
        A = torch.tensor(A, dtype=torch.float32)
        edge_index_r, edge_type_r, edge_weight_r = adjacency_to_edge_list_torch(A, r)
        all_edge_indices.append(edge_index_r)
        all_edge_types.append(edge_type_r)
        all_edge_weights.append(edge_weight_r)

    # Concatenate everything
    edge_index = torch.cat(all_edge_indices, dim=1)       # shape [2, E_total]
    edge_type = torch.cat(all_edge_types, dim=0)          # shape [E_total]
    edge_weight = torch.cat(all_edge_weights, dim=0)      # shape [E_total]

    # Cast node features to torch
    X = torch.tensor(X, dtype=torch.float32)  # Ensure float dtype

    data = Data(
        x = X,  # Ensure float dtype
        edge_index = edge_index,
        edge_type = edge_type,
        edge_weight = edge_weight,
    )

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    data.id = f"{patient_id}_{session_id}"

    # Save the graph
    torch.save(data, f"{data_dir}/graphs/{patient_id}_{session_id}_G.pt")



def label_pyg_data(data: Data, label: int):
    data.y = torch.tensor([label], dtype=torch.long)
    return data



def test_graph(data: Data, expected_num_nodes=98, expected_feat_dim=10):
    assert isinstance(data, Data), "Not a PyG Data object!"

    # Check node features
    assert hasattr(data, 'x'), "Missing node features!"
    assert data.x.shape == (expected_num_nodes, expected_feat_dim), f"Unexpected shape for x: {data.x.shape}"
    assert data.x.dtype == torch.float32, f"x should be float, got {data.x.dtype}"

    # Check edge_index
    assert hasattr(data, 'edge_index'), "Missing edge_index!"
    assert data.edge_index.shape[0] == 2, "edge_index should have shape [2, E]"
    E = data.edge_index.shape[1]
    assert E > 0, "No edges found!"
    assert data.edge_index.dtype == torch.long, "edge_index should be long"

    # Check edge_weight and edge_type
    for name in ['edge_weight', 'edge_type']:
        assert hasattr(data, name), f"Missing {name}!"
        tensor = getattr(data, name)
        assert tensor.shape[0] == E, f"{name} must match number of edges"
        assert tensor.dtype in [torch.float32, torch.long], f"{name} has unexpected dtype: {tensor.dtype}"

    # Optional: check metadata
    try:
        assert hasattr(data, 'id'), "Missing session_id for tracking"
    except AssertionError:
        print("No session_id found, but this is optional.")

    # Optional: try sending to GPU
    try:
        data_cuda = data.cuda()
        assert data_cuda.is_cuda, "Data did not move to GPU!"
        print("Data moved to GPU successfully.")
    except Exception as e:
        print(f"Could not move to GPU: {e}")

    print("Graph passed all checks!")



if __name__ == "__main__":
    # Example usage
    data_dir = "C:/Users/piete/Documents/Projects/R-GIANT/data"
    patient_id = "0001"
    session_id = "0757"
    build_pyg_data(patient_id, session_id, data_dir)