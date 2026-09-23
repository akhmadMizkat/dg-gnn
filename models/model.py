import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TemporalEncoding
from torch_geometric.nn.models.basic_gnn import GCNConv
from torch_geometric.nn.dense.linear import Linear
import math
import numpy as np

# custom module
from models.layers import (
    TimeEncoder,
    EdgeGatedConv
)

class DeterministicTemporalEncoding(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        sqrt_d = math.sqrt(out_channels)
        frequencies = 1.0 / sqrt_d ** np.linspace(0, sqrt_d, out_channels)
        self.linear = nn.Linear(1, out_channels, bias=False)

        # Set bobot manual dari luar, sama seperti sebelumnya
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.from_numpy(frequencies).float().view(out_channels, 1)
            )

        # Non-trainable: kita jadikan weight ini frozen
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        return torch.cos(self.linear(x))

class ImputationAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Linear(hidden_dim, in_dim)  # reconstruct original input dim

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class DualGatedSage(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_attr_channels=20, #50
        time_channels=20, #50
        num_layers=3,
        dropout=0.0,
        bias=True,
        bn=True,
        activation="elu",
        conv="edge_gated",
        time_encoding="temporal_encoding",
        data_imputation=True,
        loss_fn="bce_loss",
        edge_gated=True,
        residual_gated=True,
        intra_layer_gated=True,
        concat=False,
        heads=1,
        gate_mechanism=None,
        gate_attr_channels=0
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.edge_convs_a = nn.ModuleList()
        self.edge_convs_t = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.data_imputation = data_imputation
        self.loss_fn = loss_fn
        self.num_layers = num_layers
        self.embeddings = None
        self.lin_gate = Linear(17, 17, bias=True)

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            if conv == "bikernel":
                raise Exception("Conv not implemented")
            else:
                self.convs.append(
                    EdgeGatedConv(
                        (
                            first_channels + edge_attr_channels  + time_channels,
                            first_channels,
                        ),
                        second_channels,
                        concat=concat,
                        heads=heads,
                        edge_gated=edge_gated,
                        residual_gated=residual_gated,
                        gate_mechanism=gate_mechanism,
                        gate_attr_channels=gate_attr_channels
                    )
                )
            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)
        self.emb_type = nn.Embedding(12, edge_attr_channels)
        self.emb_direction = nn.Embedding(2, edge_attr_channels)
        if time_encoding == "temporal_encoding":
            self.t_enc = TemporalEncoding(time_channels)
        elif time_encoding == "deterministic":
            self.t_enc = DeterministicTemporalEncoding(time_channels)
        else:
            self.t_enc = TimeEncoder(time_channels)

        if data_imputation:
            self.lin_proj_null = ImputationAutoencoder(17, 64, dropout)
        # self.lin_proj_null = ImputationAutoencoder(in_channels, 64)

        if loss_fn == "bce_loss":
            self.classifier = Linear(out_channels, 1, bias=True)
            self.norm = nn.LayerNorm(1)

        self.intra_layer_gated = intra_layer_gated
        if intra_layer_gated:
            self.gate_linears = torch.nn.ModuleList()
            self.residual_linears = torch.nn.ModuleList()
            
            for i in range(num_layers):
                first_channel = in_channels if i == 0 else hidden_channels
                second_channel = hidden_channels
                self.residual_linears.append(torch.nn.Linear(first_channel, hidden_channels))
                self.gate_linears.append(torch.nn.Linear(hidden_channels, hidden_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.data_imputation:
            self.lin_proj_null.reset_parameters()
            
        for conv in self.convs:
            conv.reset_parameters()

        if self.intra_layer_gated:
            for l in self.residual_linears():
                l.reset_parameters()

            for l in self.gate_linears():
                l.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()
        if self.loss_fn == "bce_loss":
            self.classifier.reset_parameters()
            
        nn.init.xavier_uniform_(self.emb_type.weight)
        nn.init.xavier_uniform_(self.emb_direction.weight)

    def forward(self, x, edge_index, edge_attr, edge_t, edge_d, gate_attr=None):
        # tambahkan handling null values terhadap x
        
        # ========
        x_feat = x[:,:17]
        mask = (x[:, :17] == -1).float()

        # x_feat = x
        # mask = (x == -1).float()
        
        if self.data_imputation:
            assert mask.sum() > 0, "Mask kosong, tidak ada fitur yang diimputasi"
            
            x_feat_proj = self.lin_proj_null(x_feat)
            x_feat = x_feat + x_feat_proj * mask
            x = torch.cat([x_feat, x[:, 17:].clone()], dim=1)
            
        
        edge_attr = self.emb_type(edge_attr) + self.emb_direction(edge_d)
        edge_t = self.t_enc(edge_t.to(torch.float))
        
        for i, conv in enumerate(self.convs): 
            x_prev = x
            x = conv(x, edge_index, edge_attr, edge_t, gate_attr)

            if self.intra_layer_gated and i < self.num_layers-1:
                # Gating ke-3: residual gated fusion (mirip GRU-style update)
                beta = torch.sigmoid(
                    self.gate_linears[i](x) + self.residual_linears[i](x_prev)
                )
                x = beta * x + (1 - beta) * self.residual_linears[i](x_prev)  # gated residual fusion

            
            x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        self.embeddings = x
        
        if self.loss_fn == "bce_loss":
            return self.norm(self.classifier(x)), x_feat, mask
        return x.log_softmax(dim=-1), x_feat, mask