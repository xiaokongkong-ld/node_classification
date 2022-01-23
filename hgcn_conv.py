import math

from torch_geometric.nn import inits
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
import numpy as np
import torch.nn.init as init
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
# from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear
# from torch_geometric.nn.conv import MessagePassing
from hgcn_message_passing import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.nn.modules.module import Module


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class HGCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, manifold, in_channels: int, out_channels: int, c,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(HGCNConv, self).__init__(**kwargs)
        self.manifold = manifold
        self.c = c

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels)
        self.lin_hyp = HypLinear(self.manifold, self.in_channels, self.out_channels, self.c, 0.5, bias)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin.reset_parameters()
        self.lin_hyp.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        x = self.manifold.proj_tan0(x, self.c)
        x = self.manifold.expmap0(x, self.c)
        x = self.manifold.proj(x, self.c)

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        # x = self.manifold.proj_tan0(x, self.c)
        # x = self.manifold.expmap0(x, self.c)
        # x = self.manifold.proj(x, self.c)

        # x = self.lin(x)
        x = self.lin_hyp(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:

        if edge_weight is None:
            out = x_j
        else:
            out = edge_weight.view(-1, 1) * x_j
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # bound = 1.0 / math.sqrt(self.weight.size(-1))
        # init.uniform_(self.weight.data, -bound, bound)
        # init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        inits.glorot(self.weight)
        # inits.kaiming_uniform(self.weight, fan=self.in_features, a=math.sqrt(5))
        init.constant_(self.bias, 0)

    def forward(self, x):

        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c = c
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c))
        xt = self.manifold.proj_tan0(xt, c=self.c)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c), c=self.c)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c, self.c
        )