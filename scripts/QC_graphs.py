from rgiant.preprocessing.graphs import test_graph
import os
import torch

for graph_filename in os.listdir("data/graphs/"):
    file_path = os.path.join("data/graphs", graph_filename)
    print(file_path)
    graph = torch.load(file_path)

    try:
        test_graph(graph, expected_num_nodes=98, expected_feat_dim=10)
    except Exception as e:
        print(f"{graph_filename}: failed QC: {e}")
