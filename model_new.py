import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
from torch.nn.modules.instancenorm import _InstanceNorm
from hgcn_conv import HGCNConv, HypAct
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TopKPooling
import torch.nn as nn
import torch.nn.functional as F
import manifolds
import layers.hyp_layers_new as hyp_layers
import numpy as np
from layers.att_layers import GraphAttentionLayer
import utils_hpy.math_utils as pmath
from torch.nn import Linear
from hgcn_conv import HypLinear
from scipy.sparse import coo_matrix
from ultils import edge_2_adj, adj_2_edge, matrix_filter_value, matrix_filter_percentage


class GCN_adj(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GCN_adj, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_channels = hidden_channels

        self.conv1 = GATConv(self.channel_in, self.hidden_channels)
        self.conv2 = GATConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GATConv(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, self.channel_out)
        self.lin_adj = Linear(400, 400)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.ln = torch.nn.LayerNorm(self.hidden_channels)

        self.short_cut = nn.Sequential()
        # self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        # self.bn2 = torch.nn.BatchNorm1d(self.channel_out)

        # self.in1 = torch.nn.InstanceNorm1d(hidden_channels)
    def forward(self, x, edge, adj, batch):
        # 1. Obtain node embeddings
        print('input')
        print(edge.shape)
        adj = self.lin_adj(adj)
        adj = torch.sigmoid(adj)
        print('sigmoid')
        print(adj.shape)

        # print(adj[0])

        print('detached')
        adj = adj.detach().numpy()
        # adj = matrix_filter_percentage(adj, 0.04)
        adj = matrix_filter_value(adj, 0.6) # 0.6
        print(adj.shape)
        np.save('./adj.npy', adj)
        edge_index_coo = coo_matrix(adj)

        edge = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)
        # print(edge.shape)
        # print(edge)

        x = self.conv1(x, edge)
        x = self.ln(x)

        x = x.relu()
        # x = self.pool1(x, edge_index)

        # x = self.in1(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # adj = torch.tensor(adj, dtype=torch.float32)
        # adj = self.lin_adj(adj)
        # adj = torch.sigmoid(adj)
        # adj = adj.detach().numpy()
        # adj = matrix_filter_value(adj, 0.8)
        # edge_index_coo = coo_matrix(adj)
        # edge = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)

        x = self.conv2(x, edge)
        x = self.ln(x)

        x = x.relu()
        # x = self.pool1(x, edge_index)

        # adj = torch.tensor(adj, dtype=torch.float32)
        # adj = self.lin_adj(adj)
        # adj = torch.sigmoid(adj)
        # adj = adj.detach().numpy()
        # adj = matrix_filter_value(adj, 0.8)
        # edge_index_coo = coo_matrix(adj)
        # edge = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long)

        x = self.conv2(x, edge)
        # x = self.ln(x)

        # x3 = x.relu()

        # x = (x1 + x2)

        # out = self.conv2(x, edge_index)
        # out = out.relu()

        # out = self.conv2(x, edge_index)
        # out = out.relu()

        # x = out + self.short_cut(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)

        # return F.log_softmax(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GCN, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_channels = hidden_channels

        self.conv1 = GCNConv(self.channel_in, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, self.channel_out)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.ln = torch.nn.LayerNorm(self.hidden_channels)

        self.short_cut = nn.Sequential()
        # self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        # self.bn2 = torch.nn.BatchNorm1d(self.channel_out)

        # self.in1 = torch.nn.InstanceNorm1d(hidden_channels)
    def forward(self, x, edge, batch):
        # 1. Obtain node embeddings


        x = self.conv1(x, edge)
        x = self.ln(x)

        x = x.relu()

        x = self.conv2(x, edge)
        x = self.ln(x)

        x = x.relu()
        # x = self.pool1(x, edge_index)

        # x = self.conv2(x, edge)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)

        # return F.log_softmax(x)
        return x


class HGCN_pyg(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, hidden_channels, channel_in, channel_out):
        super(HGCN_pyg, self).__init__()
        torch.manual_seed(1234567)
        self.c = c
        self.hidden_channels = hidden_channels
        self.channel_in = channel_in
        self.channel_out = channel_out
        # self.manifold = getattr(manifolds, 'Euclidean')()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # self.manifold = getattr(manifolds, 'PoincareBall')()
        act = getattr(F, 'relu')
        self.hconv1 = HGCNConv(self.manifold, self.channel_in, self.hidden_channels, self.c)
        self.hconv2 = HGCNConv(self.manifold, self.hidden_channels, self.hidden_channels, self.c)
        # self.hconv3 = HGCNConv(self.manifold, 64, 64, self.c)
        self.hyp_act = HypAct(self.manifold, self.c, act)
        # self.lin1 = HypLinear(self.manifold, hidden_channels, self.channel_out, self.c, 0.5, True)
        # self.lin2 = Linear(64, 512)
        # self.lin3 = Linear(512, 64)
        self.ln = torch.nn.LayerNorm(self.hidden_channels)
        # self.insn = torch.nn.instanceNorm(self.hidden_channels)
        self.lin4 = Linear(self.hidden_channels, self.channel_out)

        self.short_cut = nn.Sequential()

    def forward(self, x, edge_index, batch):
        # x = self.manifold.proj_tan0(x, self.c)
        # x = self.manifold.expmap0(x, self.c)
        # x = self.manifold.proj(x, self.c)

        x = self.hconv1(x, edge_index)
        x = self.hyp_act(x)
        # x = self.ln(x)

        # x = self.bn1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.hconv2(x, edge_index)
        x = self.hyp_act(x)
        # x = self.ln(x)

        # x = self.hconv2(x, edge_index)
        # x3 = self.hyp_act(x)

        # x = out + self.short_cut(x)
        #
        # out = self.hconv2(x, edge_index)
        # out = self.hyp_act(out)
        #
        # x = out + self.short_cut(x)
        #
        # out = self.hconv2(x, edge_index)
        # out = self.hyp_act(out)
        #
        # x = out + self.short_cut(x)
        # x = x1 + x2

        x = self.manifold.logmap0(x, c=self.c)
        x = self.manifold.proj_tan0(x, c=self.c)

        # x = F.log_softmax(x)
        x = global_mean_pool(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.lin2(x)
        # x = x.relu()
        # x = self.bn1(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.lin3(x)
        # x = x.relu()
        # x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin4(x)

        return x
        # return F.log_softmax(x)

class ResGCN(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(ResGCN, self).__init__()
        torch.manual_seed(1234567)

        self.hidden_channels = hidden_channels
        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = GraphConv(self.channel_in, self.hidden_channels)
        self.conv2 = GraphConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GraphConv(self.hidden_channels, self.hidden_channels)
        self.lin1 = Linear(self.hidden_channels, self.channel_out)

        # self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        # self.bn2 = torch.nn.BatchNorm1d(self.channel_out)

        # self.in1 = torch.nn.InstanceNorm1d(hidden_channels)
    def forward(self, x, edge_index, adj, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.in1(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        # x = self.bn2(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        # return F.log_softmax(x)
        x = self.lin1(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(1000, 600)
        self.lin2 = Linear(600, 200)
        self.lin3 = Linear(200, 86)
        self.lin4 = Linear(86, 16)
        self.lin5 = Linear(16, 7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin5(x)
        return x


# class HGCN(torch.nn.Module):
#     """
#     Hyperbolic-GCN.
#     """
#
#     def __init__(self, c):
#         super(HGCN, self).__init__()
#         self.c = c
#         # self.manifold = getattr(manifolds, 'Euclidean')()
#         self.manifold = getattr(manifolds, 'Hyperboloid')()
#         # self.manifold = getattr(manifolds, 'PoincareBall')()
#         act = getattr(F, 'relu')
#         # self.lin = Linear(64, 7, dropout=0.0, act=act, use_bias=True)
#         self.hgcov1 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 1433, 100, 1, 1, 0.0, act, 1, 0, 0)
#         self.hgcov2 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 100, 7, 1, 1, 0.0, act, 1, 0, 0)
#         self.encode_graph = True
#
#     def forward(self, x, adj):
#         # print('......................xtan............................')
#
#         x_tan = self.manifold.proj_tan0(x, 1)
#
#         x_hyp = self.manifold.expmap0(x_tan, 1)
#         # print(x_hyp)
#         x_hyp = self.manifold.proj(x_hyp, 1)
#         h = self.hgcov1.forward(x_hyp, adj)
#         h = self.hgcov2.forward(h, adj)
#
#         h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)
#         # h = self.lin(h)
#         return F.log_softmax(h)
#         # return h

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GAT, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = GATConv(self.channel_in, hidden_channels)
        self.conv2 = GATConv(hidden_channels, self.channel_out)
        self.lin1 = Linear(hidden_channels, self.channel_out)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        return F.log_softmax(x)
        # return x


class GraphSage(torch.nn.Module):
    def __init__(self, hidden_channels, channel_in, channel_out):
        super(GraphSage, self).__init__()
        torch.manual_seed(1234567)

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = SAGEConv(self.channel_in, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, self.channel_out)
        self.lin1 = Linear(hidden_channels, self.channel_out)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        return F.log_softmax(x)
        # return x
