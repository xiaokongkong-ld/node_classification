from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from visual import visualize, plot_graph
import numpy as np
from ultils import edge_2_adj, adj_2_edge
import networkx as nx
from hyperbolicity import hyperbolicity_sample


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())  # PubMed, Cora

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
# print('ori')
# print(data.edge_index)
# adj = edge_2_adj(data.edge_index)
# adj_2_edge(adj)

print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# G = nx.Graph()
edges = data.edge_index.numpy().T
# G.add_edges_from(edges)
# hyp = hyperbolicity_sample(G)

# visualize(data.x, color=data.y)

plot_graph(edges)
