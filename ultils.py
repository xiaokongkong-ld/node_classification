from scipy.sparse import coo_matrix
import numpy as np
import torch
from visual import plot_mat

def rd_mask(mat_size, true_perc):

    mask = np.full(mat_size * mat_size, 0)
    true_num = int(mat_size*mat_size*true_perc)
    mask[:true_num] = 1
    np.random.shuffle(mask)
    mask = mask.reshape([mat_size, mat_size])

    return mask

def matrix_filter_value(matrix, filter):
    f_mat = matrix.copy()
    np.fill_diagonal(f_mat, 0)
    f_mat2 = f_mat.copy()
    f_mat2[f_mat >= filter] = 1
    f_mat2[f_mat < filter] = 0
    return f_mat2

def matrix_filter_percentage(mat, perc):
    x, y = mat.shape
    mac = mat.copy()
    np.fill_diagonal(mac, -10)
    mac = mac.reshape(-1)
    k = int(len(mac) * perc)
    idx = mac.argsort()
    idx = idx[::-1]
    top_k_idx = idx[:k]
    down_k_id = idx[k:]
    mac[top_k_idx] = 1
    mac[down_k_id] = 0
    return mac.reshape(x, -1)

def edge_2_adj(edges):
    edges = edges.numpy().T

    adj_len = edges.max() + 1
    adj = np.zeros([adj_len, adj_len])

    for edge in edges:

        left = edge[0]
        right = edge[1]

        adj[left][right] = 1

    adj = torch.tensor(adj, dtype=torch.float32)

    return adj

# ed = np.array([[0,1],[1,3],[2,3],[3,4],[4,5],[4,6]])
# ad = edge_2_adj(ed)
# print(ad)

def adj_2_edge(adj):
    adj = adj.detach().numpy()

    edge = coo_matrix(adj)
    edge = torch.tensor(np.vstack((edge.row, edge.col)), dtype=torch.long)
    return edge


