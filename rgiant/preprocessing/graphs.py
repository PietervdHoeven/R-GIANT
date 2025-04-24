import torch
from torch_geometric.data import Data
import numpy as np

def adjacency_to_edge_attributes(A_r: torch.Tensor, relation_type: int):
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
    Create a PyTorch Geometric Data object from multi-metric adjacency matrices and node features.

    This routine loads:
        1. A `.npz` archive of R adjacency matrices (one per DWI metric) for a given subject/session,
            each of shape [N, N].
        2. A `[N, F]` NumPy array of node features for the same subject/session.
    It then:
        • Identifies every unique node-pair (i, j) present in any metric matrix.
        • Builds a single edge_index tensor of shape [2, E], listing those pairs once.
        • Packs the R metric values for each edge into an `edge_attr` tensor of shape [E, R].
        • Attaches the node-feature matrix `x`, an optional label `y`, and an identifier `id`.

    Parameters
    ----------
    patient_id : str
        Identifier for the subject.
    session_id : str
        Identifier for the scanning session.
    data_dir : str
        Base directory containing:
            - `matrices/{patient_id}_{session_id}_As.npz`
            - `matrices/{patient_id}_{session_id}_X.npy`
    label : int, optional
        Class label for the graph (e.g., diagnosis). If provided, stored in `data.y`.

    Returns
    -------
    torch_geometric.data.Data
        A Data object with attributes:
            - x         (Tensor[N, F])   : Node feature matrix.
            - edge_index (LongTensor[2, E]) : COO indices of all unique edges.
            - edge_attr  (Tensor[E, R])  : Edge-metric vectors (one column per adjacency matrix).
            - y         (LongTensor[1], optional) : Graph label, if `label` given.
            - id        (str)            : Combined patient+session identifier.

    Notes
    -----
    • The order of the R metrics in `edge_attr[:, r]` follows the sorted keys of the `.npz` file,
        ensuring consistent column-to-metric mapping across runs.
    • All inputs and outputs use `torch.float32` for features and attributes, and `torch.long`
        for indices and labels.
    • Existing parallel-edge logic is replaced: each node-pair appears exactly once, carrying
        a multi-dimensional feature vector.
    """
    As_path = f"{data_dir}/matrices/{patient_id}_{session_id}_As.npz"
    X_path = f"{data_dir}/matrices/{patient_id}_{session_id}_X.npy"

    # load all matrices from the .np* files
    As = np.load(As_path)  
    X = np.load(X_path)

    for k in sorted(As.files):
        print(k)
    # Load the individual adjacency matrices into tensors and store them in a list of tensors list[tensor[N,N]]
    # sort keys to fix metric order (e.g. [FA, MD, RD, AD, LEN, COUNT] -> ad count fa length md norm_count rd
    As = [torch.tensor(As[k], dtype=torch.float32) for k in sorted(As.files)]


    # Find every unique edge from ROI_i to ROI_j, these are the [i,j] in A that have nonzero values
    edge_coords = torch.cat([A.nonzero(as_tuple=False) for A in As], dim=0)  # A.nonzero returns a [E,2] tensor with on each row the column and row indices of nonzero elements in A
    edge_coords   = torch.unique(edge_coords, dim=0)   # shape [E, 2], We remove duplicate edges that have been found

    # build edge_index and edge_attr
    src_coords, dst_coords = edge_coords[:,0], edge_coords[:,1] # take two lists, one for the source index of each edge and one for destination index for each edge
    edge_index = torch.stack([src_coords, dst_coords], dim=0)  # Stack them into a tensor of shape [2,E], a tensor list indicating the source and destination node of each edge
    E = edge_coords.size(0);  R = len(As)
    edge_attr = torch.zeros(E, R, dtype=torch.float32)  # Initialise the edge_attr tensor that will contain all the edge attributes, i.e., the metrics for each edge

    # fill each metric column
    for r, A in enumerate(As):
        # gather metrics A[i,j] for each edge (i,j)
        edge_attr[:, r] = A[src_coords, dst_coords]

    # Cast node features to torch
    X = torch.tensor(X, dtype=torch.float32)  # Ensure float dtype

    data = Data(
        x = X,  # Ensure float dtype
        edge_index = edge_index,
        edge_attr = edge_attr,
    )

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    data.id = f"{patient_id}_{session_id}"

    # Save the graph
    #torch.save(data, f"{data_dir}/graphs/{patient_id}_{session_id}_G.pt")




def label_pyg_data(data: Data, label: int):
    data.y = torch.tensor([label], dtype=torch.long)
    return data



def test_graph(
    data: Data,
    expected_num_nodes: int = 98,
    expected_feat_dim: int = 11,
    expected_edge_dim: int = 7,   # number of metrics in edge_attr
):
    # — basic type check —
    assert isinstance(data, Data), "Not a PyG Data object!"

    # — node features —
    assert hasattr(data, 'x'), "Missing node features!"
    assert data.x.shape == (expected_num_nodes, expected_feat_dim), \
        f"Unexpected x shape: {data.x.shape}"
    assert data.x.dtype == torch.float32, \
        f"x should be float32, got {data.x.dtype}"

    # — edge_index —
    assert hasattr(data, 'edge_index'), "Missing edge_index!"
    ei = data.edge_index
    assert ei.ndim == 2 and ei.shape[0] == 2, \
        f"edge_index should be [2, E], got {ei.shape}"
    E = ei.shape[1]
    assert E > 0, "No edges found!"
    assert ei.dtype == torch.long, \
        f"edge_index should be long, got {ei.dtype}"

    # — edge_attr —
    assert hasattr(data, 'edge_attr'), "Missing edge_attr!"
    ea = data.edge_attr
    # must be a 2D tensor with E rows and expected_edge_dim columns
    assert ea.ndim == 2, f"edge_attr must be 2D, got {ea.ndim}D"
    assert ea.shape[0] == E, \
        f"edge_attr rows ({ea.shape[0]}) must match number of edges ({E})"
    assert ea.shape[1] == expected_edge_dim, \
        f"edge_attr should have {expected_edge_dim} columns, got {ea.shape[1]}"
    assert ea.dtype == torch.float32, \
        f"edge_attr should be float32, got {ea.dtype}"

    # — optional metadata —
    if not hasattr(data, 'id'):
        print("Warning: no `data.id` field (optional)")

    # — optional GPU test —
    try:
        data_cuda = data.cuda()
        assert data_cuda.is_cuda, "Data did not move to GPU!"
        print("Data moved to GPU successfully.")
    except Exception as e:
        print(f"Could not move to GPU (OK if no GPU): {e}")

    print("Graph passed all checks!")



if __name__ == "__main__":
    # Example usage
    data_dir = "C:/Users/piete/Documents/Development/Projects/R-GIANT/data/connectome_pipeline"
    patient_id = "0001"
    session_id = "0757"
    graph = build_pyg_data(patient_id, session_id, data_dir)
    test_graph(graph)
