import os
import torch
from torch_geometric.data import Data
import numpy as np

# 由于是无向图，因此有 4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
# 节点的特征
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

def load_data(graph_dir, label_dir):
    dataset = []
    graph_files = os.listdir(graph_dir)
    # label_files = os.listdir(label_dir)
    for file in graph_files:
        if not file.endswith(".txt"):
            continue

        # Get edge data
        edge=[]
        f_graph = open(os.path.join(graph_dir, file), 'r')
        for line in f_graph:
            a,b = line.strip().split('\t')
            edge.append([int(a), int(b)])
        edge = torch.tensor(edge, dtype=torch.long)
        edge_index = edge.t().contiguous()
        f_graph.close()

        # Get label data
        label = []
        f_label = open(os.path.join(label_dir, file), 'r')
        for line in f_label:
            a,b = line.strip().split('\t')
            label.append([int(a), int(b)])
        label = np.array(label)
        idex=np.lexsort([label[:,0]])
        sorted_label = label[idex, :]
        print(sorted_label)
        x = torch.tensor(sorted_label[:,1], dtype=torch.float)
        f_label.close()
        print("label")
        print(x)
        print("edge")
        print(edge_index)
        data = Data(x=x, edge_index=edge_index)
        dataset.append(data)
    return dataset

load_data("graph", "label")