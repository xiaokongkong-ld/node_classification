import os
import pickle as pkl
import sys
import time
from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
from tqdm import tqdm
import heapq

# from utils.data_utils import load_data_lp
def matrix_filter_topK(matrix, K):
    """ filt matrix with top K max"""
    m_l = len(matrix[0])
    index = [x for x in range(m_l)]
    f_mat = np.zeros(m_l)
    mat = matrix.copy()
    np.fill_diagonal(mat, -100)

    for x in mat:
        y = heapq.nlargest(K, range(len(x)), x.take)
        y_else = np.setdiff1d(index, y)
        x[y] = 1
        x[y_else] = 0
        f_mat = np.vstack((f_mat, x))
    f_mat = np.delete(f_mat, np.s_[0], axis=0)
    # print('=====================================================================================')
    # print(f_mat)
    return f_mat

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


def matrix_filter_value(matrix, filter):
    f_mat = matrix.copy()
    np.fill_diagonal(f_mat, 0)
    f_mat2 = f_mat.copy()
    f_mat2[f_mat >= filter] = 1
    f_mat2[f_mat < filter] = 0
    return f_mat2

def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)


if __name__ == '__main__':
    print('hello')
    # G = nx.Graph()
    #
    # G.add_edges_from(ed)
    # hyp = hyperbolicity_sample(G)

