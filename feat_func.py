import torch
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter, scatter_min, scatter_max, scatter_add
from torch import Tensor
import numpy as np

def add_time_density(x: Tensor, edge_index: Tensor, edge_timestamp: Tensor) -> Tensor:
    row, col = edge_index  # asumsi edge_index [2, num_edges]
    num_nodes = x.size(0)

    # Step 1: Hitung min & max timestamp untuk edge keluar tiap node
    min_time, _ = scatter_min(edge_timestamp, row, dim=0, dim_size=num_nodes)
    max_time, _ = scatter_max(edge_timestamp, row, dim=0, dim_size=num_nodes)

    # Step 2: Hitung outdegree
    out_degree = torch_geometric.utils.degree(row, num_nodes, dtype=edge_timestamp.dtype)

    # Hindari pembagian nol
    out_degree = out_degree + 1e-8

    # Step 3: Hitung delta_time sebagai (max - min) / outdegree
    delta_time = (max_time - min_time) / out_degree

    # Bersihkan inf/nan
    delta_time[delta_time != delta_time] = 0  # NaN ke 0

    # Step 4: Hitung sum delta tetangga (anggap undirected)
    neighbor_delta_sum = scatter_add(delta_time[row], col, dim=0, dim_size=num_nodes)

    # Step 5: Final time density
    time_density = delta_time / (neighbor_delta_sum + 1e-8)
    time_density = time_density.unsqueeze(1)

    # normalisasi
    time_density = (time_density - time_density.mean()) / (time_density.std() + 1e-6)

    # Step 6: Tambahkan ke fitur
    x = torch.cat([x, time_density], dim=1)

    return x

def add_degree_feature(x: Tensor, edge_index: Tensor):
    row, col = edge_index
    in_degree = torch_geometric.utils.degree(col, x.size(0), x.dtype)

    out_degree = torch_geometric.utils.degree(row, x.size(0), x.dtype)
    return torch.cat([x, in_degree.view(-1, 1), out_degree.view(-1, 1)], dim=1)


def add_feature_flag(x, data_imputation=False):
    feature_flag = torch.zeros_like(x[:, :17])
    feature_flag[x[:, :17] == -1] = 1
    if not data_imputation:
        x[x == -1] = 0
    return torch.cat((x, feature_flag), dim=1)


def add_label_feature(x, y):
    y = y.clone()
    # All fraudulent nodes are temporarily considered as normal users to simulate the scenario of mining fraudulent users from normal users.
    y[y == 1] = 0
    y_one_hot = F.one_hot(y).squeeze()
    return torch.cat((x, y_one_hot[:, :-1]), dim=1)


def add_label_counts(x, edge_index, y):
    y = y.clone().squeeze()
    background_nodes = torch.logical_or(y == 2, y == 3)
    foreground_nodes = torch.logical_and(y != 2, y != 3)
    y[background_nodes] = 1
    y[foreground_nodes] = 0

    row, col = edge_index
    a = F.one_hot(y[col])
    b = F.one_hot(y[row])
    temp = scatter(a, row, dim=0, dim_size=y.size(0), reduce="sum")
    temp += scatter(b, col, dim=0, dim_size=y.size(0), reduce="sum")

    return torch.cat([x, temp.to(x)], dim=1)


def cos_sim_sum(x, edge_index):
    row, col = edge_index
    sim = F.cosine_similarity(x[row], x[col])
    sim_sum = scatter(sim, row, dim=0, dim_size=x.size(0), reduce="sum")
    return torch.cat([x, torch.unsqueeze(sim_sum, dim=1)], dim=1)


def to_undirected(edge_index, edge_attr, edge_timestamp):

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_timestamp = torch.cat([edge_timestamp, edge_timestamp], dim=0)
    return edge_index, edge_attr, edge_timestamp


def data_process(data, preprocess=True, feature_flag=True, data_imputation=False, time_density=True):
    edge_index, edge_attr, edge_timestamp = (
        data.edge_index,
        data.edge_attr,
        data.edge_timestamp,
    )

    x = data.x
    edge_index, edge_attr, edge_timestamp = to_undirected(
            edge_index, edge_attr, edge_timestamp
        )
    mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    edge_timestamp = edge_timestamp[mask]
    data.edge_index, data.edge_attr, data.edge_timestamp = to_undirected(
        edge_index, edge_attr, edge_timestamp
    )
    engineered_feat_idx = dict()
    if preprocess:
        degree_feature = add_degree_feature(x, edge_index)
        engineered_feat_idx["degree_feature"] = (x.shape[1], degree_feature.shape[1])
        x = degree_feature

        node_simililarity = cos_sim_sum(x, edge_index)
        engineered_feat_idx["node_simililarity"] = (x.shape[1], node_simililarity.shape[1])
        x = node_simililarity

        if feature_flag:
            missing_feature_flag = add_feature_flag(x, data_imputation) # 79.9793
            engineered_feat_idx["missing_feature_flag"] = (x.shape[1], missing_feature_flag.shape[1])
            x = missing_feature_flag
        
        label_counts = add_label_counts(x, edge_index, data.y) #80.13
        engineered_feat_idx["label_counts"] = (x.shape[1], label_counts.shape[1])
        x = label_counts

        label_feature = add_label_feature(x, data.y) #81.5648
        engineered_feat_idx["label_feature"] = (x.shape[1], label_feature.shape[1])
        x = label_feature
        
        if time_density:
            node_time_density = add_time_density(x, edge_index, edge_timestamp)
            engineered_feat_idx["node_time_density"] = (x.shape[1], node_time_density.shape[1])
            x = node_time_density

    data.edge_direct = torch.ones(data.edge_attr.size(0), dtype=torch.long)
    data.edge_direct[: data.edge_attr.size(0) // 2] = 0
    
    data.x = x
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)
    return data, engineered_feat_idx