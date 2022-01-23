import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, Linear
import torch.nn.functional as F
from ultils import edge_2_adj, adj_2_edge, matrix_filter_value, matrix_filter_percentage, rd_mask

from hgcn_conv import HGCNConv, HypAct
import torch.nn as nn
import manifolds
import numpy as np
from layers import hyp_layers
from scipy.sparse import coo_matrix, csc_matrix


class GCN_adj(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv0 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.fc_adj = Linear(2708, 2708)

    def forward(self, x, edge_index):

        adj = edge_2_adj(edge_index)

        identity = torch.eye(2708).float()
        mask = self.fc_adj(adj)

        mask = torch.sigmoid(mask)
        mask = torch.triu(mask)
        mask += mask.T - torch.diag(torch.diag(mask, 0), 0)

        mask = mask.mul(adj)
        mask = torch.gt(mask, 0.501)
        # mask = matrix_filter_value(mask, 0.6)
        # mask = torch.round(mask)
        # mask_adj = mask + adj

        edge_index = adj_2_edge(mask)
        print(edge_index.shape)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        return x


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x)
        return x


class HGCN_pyg(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, channel_in, hidden_channels, channel_out):
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
        self.hconv2 = HGCNConv(self.manifold, self.hidden_channels, self.channel_out, self.c)
        self.hyp_act = HypAct(self.manifold, self.c, act)
        self.ln = torch.nn.LayerNorm(self.hidden_channels)

    def forward(self, x, edge_index):
        x = self.hconv1(x, edge_index)
        x = self.hyp_act(x)

        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.hconv2(x, edge_index)

        x = self.manifold.logmap0(x, c=self.c)
        x = self.manifold.proj_tan0(x, c=self.c)

        # x = self.lin(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.log_softmax(x)
        return x


class HGCN(torch.nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c):
        super(HGCN, self).__init__()
        self.c = c
        # self.manifold = getattr(manifolds, 'Euclidean')()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # self.manifold = getattr(manifolds, 'PoincareBall')()
        act = getattr(F, 'relu')
        # self.lin = Linear(64, 7, dropout=0.0, act=act, use_bias=True)
        self.hgcov1 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 1433, 64, 1, 1, 0.0, act, 1, 0, 0)
        self.hgcov2 = hyp_layers.HyperbolicGraphConvolution(self.manifold, 64, 7, 1, 1, 0.0, act, 1, 0, 0)
        self.encode_graph = True

    def forward(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, 1)

        x_hyp = self.manifold.expmap0(x_tan, 1)
        # print(x_hyp)
        x_hyp = self.manifold.proj(x_hyp, 1)
        h = self.hgcov1.forward(x_hyp, adj)
        h = self.hgcov2.forward(h, adj)

        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)

        return h


class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(input_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        x = F.log_softmax(x)
        return x
