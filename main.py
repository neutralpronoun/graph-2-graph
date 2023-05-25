import torch
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn


U_net = pyg_nn.GraphUNet(in_channels=10,
                         hidden_channels=10,
                         out_channels=10,
                         depth=3)
print(U_net)
print(U_net.down_convs)