from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter as scatter
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import get_laplacian, remove_self_loops
from torch_sparse import SparseTensor, matmul
from typing import Union, Tuple


class TimeEncoder(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
            .float()
            .reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension))

    def reset_parameters(self):
        pass

    def forward(self, t):
        t = torch.log(t + 1)
        t = t.unsqueeze(dim=1)
        output = torch.cos(self.w(t))
        return output

class EdgeGatedConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize: bool = False,
        bias: bool = True,
        dropout: float = 0.3,
        heads: int = 3,
        concat: bool = False,
        edge_gated: bool = True,
        residual_gated: bool = True,
        gate_mechanism = None,
        gate_attr_channels = 0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.normalize = normalize
        self.concat = concat
        self.edge_gated = edge_gated
        self.residual_gated = residual_gated

        self.relu = nn.ReLU()
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_m = Linear(in_channels[0], out_channels * heads, bias=bias)
        if self.concat:
            self.lin_f = Linear(out_channels * heads, out_channels, bias=bias)

        if edge_gated:
            self.lin_g = Linear((in_channels[0]-in_channels[1])+gate_attr_channels, out_channels * heads, bias=bias) # berfungsi sebagai filter untuk menetukan mana yang bagus!!!!!
        

        if self.concat:
            self.lin_r = Linear(in_channels[1], out_channels * heads, bias=bias)
            # self.alpha_layer = nn.Linear(out_channels * heads, 1) # OLD
            if residual_gated:
                self.alpha_layer = nn.Linear(out_channels * 2 * heads, out_channels * heads) # NEW
        else:
            self.lin_r = Linear(in_channels[1], out_channels, bias=bias)
            # self.alpha_layer = nn.Linear(out_channels, 1)  # OLD
            if residual_gated:
                self.alpha_layer = nn.Linear(out_channels * 2, out_channels)  # NEW

        self.dropout = nn.Dropout(dropout)
        self.gate_mechanism = gate_mechanism

    def reset_parameters(self):
        self.lin_r.reset_parameters()

        self.lin_m.reset_parameters()
        if self.concat:
            self.lin_f.reset_parameters()
        
        if self.edge_gated:
            self.lin_g.reset_parameters()
            
        if self.residual_gated:
            self.alpha_layer.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_t: Tensor,
        gate_attr: Tensor
    ) -> Tensor:
        H, C = self.heads, self.out_channels
        
        row, col = edge_index
        if self.edge_gated:
            x_j = torch.cat([x[col], edge_attr, edge_t], dim=1)
            gate_attr = gate_attr[col]
            x_j = self.lin_m(x_j).view(-1, H, C)
            x_j = self.dropout(x_j)
            gate = self.lin_g(torch.cat([edge_attr, edge_t, gate_attr], dim=1)).view(-1, H, C)
            if self.gate_mechanism == "glu":
                gate = torch.sigmoid(gate)  # Nilai antara 0-1
            elif self.gate_mechanism == "gtu":
                x_j = torch.tanh(x_j)
                gate = torch.sigmoid(gate)
            elif self.gate_mechanism == "gtru":
                x_j = torch.tanh(x_j)
                gate = torch.relu(gate)
            else:
                raise Exception("Unknown gate mechanism")

            
            if self.concat:
                x_j = (x_j * gate).view(-1, H * C)  # Gated message
            else:
                x_j = (x_j * gate).mean(dim=1)  # Gated message
            
            
            x_j = scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce="sum") 
            x_i = self.lin_r(x)
        else:
            x_j = torch.cat([x[col], edge_attr, edge_t], dim=1)
            x_j = scatter.scatter(x_j, row, dim=0, dim_size=x.size(0), reduce="sum")
            x_j = self.lin_m(x_j)
            x_i = self.lin_r(x)

        if self.residual_gated:
            # alpha = torch.sigmoid(self.alpha_layer(x_j)) # OLD
            alpha = torch.sigmoid(self.alpha_layer(torch.cat([x_i,x_j], dim=1))) # NEW
            out = alpha * x_j + (1 - alpha) * x_i
            # out = alpha * x_j + x_i
            
            if self.concat:
                out = self.lin_f(out)
        else:
            # out = 0.5 * x_j + x_i 
            out = x_j # SEMENTARA

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out
