import torch
import networkx as nx
import numpy as np
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from tqdm import tqdm


# class DatasetFromNX(pyg.data.InMemoryDataset):
#     def __init__(self, networkx_graphs):
#         super().__init__(transform = None, process = None)
#         self.nx_graphs  = networkx_graphs
#         self.pyg_graphs = [pyg.utils.from_networkx(g, group_node_attrs=all) for g in self.nx_graphs]
#

def apply_noise(x, noise_variance=0.1):
    # Applies noise to a pyg Batch object

    x_shape = x.shape
    noise = (torch.randn(size=x_shape)) * (noise_variance ** 0.5)

    return noise + x
class Trainer:
    def __init__(self, model, data, diffusion_steps = 400):
        self.model = model
        self.data = data

        if torch.cuda.is_available():
            self.model.cuda()
        pass

    def init(self):
        # In the U-Net code - seems to instantiate data, loaders and an optimiser
        pass

    def to_cuda(self):
        # Again in U-Net, just sends a list of graphs to cuda if its available
        pass

    def run_epoch(self):
        # Runs an epoch. Takes epoch number, data, model and optimizer.
        # Calculates
        pass

    def apply_noise(self, x, noise_variance=0.1):
        # Applies noise to a pyg Batch object

        x_shape = x.shape
        noise = (torch.randn(size=x_shape)) * (noise_variance**0.5)

        return noise + x

if __name__ == "__main__":

    nx_dummy_graphs = [nx.grid_2d_graph(100,
                                        200) for _ in range(20)]
    for nx_g in nx_dummy_graphs:
        n_nodes = nx_g.order()
        for n in nx_g.nodes:
            nx_g.nodes[n]["color"] = np.arange(256) / 256

    pyg_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in nx_dummy_graphs],
                                     batch_size=5)

    model = pyg_nn.GraphUNet(in_channels=256,
                             hidden_channels=100,
                             out_channels=256,
                             depth=3).cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    n_epochs = 20

    pbar = tqdm(range(n_epochs))

    for e in pbar:
        loss = 1000

        for b in pyg_loader:
            optimizer.zero_grad()

            target = b.x.float().cuda()
            out = model(apply_noise(b.x).float().cuda(), b.edge_index.cuda())
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss}")
            # print(loss)

    pass

