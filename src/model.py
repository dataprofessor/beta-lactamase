import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch_geometric.nn as gnn 

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_dim, attention_heads = 4, dropout = 0.3):
        super(GNN, self).__init__()
        self.gat1 = gnn.GATConv(
            in_channels = in_features,
            out_channels = hidden_features,
            heads = attention_heads,
            concat = True,
            edge_dim=edge_dim,
            dropout = dropout)
        
        self.gat2 = gnn.GATConv(
            in_channels = attention_heads * hidden_features,
            out_channels = hidden_features,
            edge_dim=edge_dim,
            heads = 2,
            concat = False)
        
        self.regressor = nn.Linear(hidden_features, 1)

    def forward(self, x, edge_attr, edge_index, batch):
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = gnn.global_max_pool(x, batch)
        #x = torch.relu(x)
        x = self.regressor(x)
        return x