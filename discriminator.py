import torch
from torch.nn import CosineSimilarity
import os
import wandb
import copy
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.datasets import QM9
from ogb.nodeproppred import PygNodePropPredDataset
from UNet import GraphUNet
from gat import GAT, EdgeCNN
# from extra_features import ExtraFeatures, SimpleNodeCycleFeatures
# print(os.getcwd())
# print(os.listdir())
# os.chdir("../")
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from ToyDatasets import *
from Metrics import *
from Datasets import *
from Diffusion import *
# os.chdir("graph-2-graph")

import sys,os
sys.path.append(os.getcwd())


class GCN(torch.nn.Module):
    def __init__(self, x_dim, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(x_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return torch.sigmoid(x)


class Discriminator(torch.nn.Module):
    def __init__(self, x_dim):
        super(Discriminator, self).__init__()

        # batch_size = int(cfg["batch_size"])
        # val_prop = float(cfg["val_prop"])
        # test_prop = float(cfg["test_prop"])
        #
        # hidden_dim = int(cfg["hidden_dim"])
        # num_layers = int(cfg["n_layers"])
        # extra_features = cfg["extra_features"]

        vis_fn = "pca"
        # self.loss_fn = torch.nn.MSELoss()
        self.x_dim = x_dim
        self.loss_fn = torch.nn.BCELoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = GCN(x_dim = x_dim, hidden_channels=64).double()
        print(self.model)

        # self.model = GraphUNet(in_channels=self.x_dim + 1 + self.features_dim, # Currently passing t as a node-level feature
        #                          hidden_channels=hidden_dim,
        #                          out_channels=self.x_dim,
        #                          depth=num_layers,
        #                          pool_ratios=0.25).to(self.device)



        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001)



    def get_xpreds(self, val_loader, wrapper):
        x_preds = []
        for ib_val, val_batch in enumerate(val_loader):
            val_batch_loss, x_pred = wrapper.sample_features(val_batch)
            x_preds.append(x_pred)
        return x_preds

    def epoch(self, val_loader, wrapper = None, x_generated = None):
        if x_generated is None:
            x_generated = self.get_xpreds(val_loader, wrapper)
        loader = self.prepare_data(val_loader, x_generated)
        self.train(loader)
        loader = self.prepare_data(val_loader, x_generated)
        loss, acc = self.test(loader)

        # print(f"Discriminator accuracy: {acc}, discriminator loss {loss}")
        wandb.log({"Discriminator-Accuracy":acc,
                   "Discriminator-Loss":loss})


    def prepare_data(self, val_loader, x_generated):

        edge_indices = []
        xs = []
        xs = []
        ys = []

        datalist = []

        counter = 0
        for ib, batch in enumerate(tqdm(val_loader, leave=False)):
            n_graphs = batch.num_graphs
            # print(f"batch has {n_graphs} graphs")
            x_slicing_indices = []
            running_count = 0
            for ig in range(n_graphs):
                example = batch.get_example(ig)
                # edge_indices += example
                # print(example.shape)
                # x_slicing_indices += [example.x.shape[0]]
                ran = np.random.random()
                if ran < 0.5:
                    x = torch.clone(example.x)
                    y = 0
                else:
                    x = torch.clone(x_generated[ib][running_count:running_count+example.x.shape[0], :])
                    y = 1

                running_count += example.x.shape[0]
            #
            #

            # for indices in x_slicing_indices:
            #     # print(x_pred.shape, indices, running_count, running_count + indices)
            #     i_x_pred = x_pred[running_count:running_count+indices, :]
            #     # print(x_pred)
            #     x_preds += [i_x_pred.detach().cpu()]

                data = pyg.data.Data(edge_index=torch.clone(example.edge_index).to(self.device), x = x.to(torch.double).to(self.device), y = y)
                # print(data)
                datalist.append(data)
                counter += 1


        return pyg.loader.DataLoader(datalist, batch_size = 32, shuffle=True)



    def train(self, loader, n_epochs = 5):
        self.model.train()

        self.model = GCN(x_dim = self.x_dim, hidden_channels=64).double().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001)

        for i_epoch in tqdm(range(n_epochs), leave = False):
            for data in loader:
                # print(data)

                # loss = 0.
                # print(data.x.dtype, data.x.to(torch.double).dtype,  data.edge_index.dtype, data.batch.dtype)
                out = self.model(data.x, data.edge_index, data.batch)

                # print(out.view(data.y.shape), data.y)
                loss = self.loss_fn(out.view(data.y.shape).to(self.device), data.y.double().to(self.device))
                # print(loss)


                loss.backward()#retain_graph  = True)
                self.optimizer.step()
                self.optimizer.zero_grad()


    def test(self, loader):
        self.model.eval()

        correct = 0
        total = 0
        loader_pbar = tqdm(loader)
        with torch.no_grad():
            for data in loader_pbar:
                out = self.model(data.x, data.edge_index, data.batch)
                # pred = out.argmax(dim = 1)
                pred = torch.round(out).flatten()
                # print(pred, data.y)
                loss = self.loss_fn(out.view(data.y.shape).to(self.device), data.y.double().to(self.device))
                correct += int((pred.cpu() == data.y).sum())
                total += data.y.shape[0]


        # Accuracy
        # print(correct, len(loader.dataset))
        return loss, correct / total





if __name__ == "__main__":
    disc = Discriminator(12, 12)# main()


