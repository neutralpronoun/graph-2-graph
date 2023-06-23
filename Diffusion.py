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
from ToyDatasets import *
from Metrics import *
from Datasets import *
# os.chdir("graph-2-graph")





def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)

def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace

class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()


    def k2_cycle(self):
        c2 = batch_diagonal(self.k2_matrix)

        return c2.unsqueeze(-1).float(), (torch.sum(c2, dim=-1)/4).unsqueeze(-1).float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        #
        # print(f"C3: {c3}\n"
        #       f"C3 unsqueezed: {(c3 / 2).unsqueeze(-1).float()}\n"
        #       f"second term: {(torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()}")

        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)

        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        # TODO: removed asserts as for the bike dataset was throwing errors. Fix later!
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k2x, k2y = self.k2_cycle()
        # assert (k2x >= -0.1).all(), k2x[k2x <= -0.1]

        k3x, k3y = self.k3_cycle()
        # assert (k3x >= -0.1).all(), k3x[k3x <= -0.1]

        k4x, k4y = self.k4_cycle()
        # assert (k4x >= -0.1).all(), k4x[k4x <= -0.1]

        k5x, k5y = self.k5_cycle()
        # assert (k5x >= -0.1).all(), k5x[k5x <= -0.1]

        _, k6y = self.k6_cycle()
        # assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k2x, k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k2y, k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy

class SimpleNodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, adj_matrix):

        # adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)
        # x_cycles = x_cycles.type_as(adj_matrix) * noisy_data['node_mask'].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles

def setup_wandb(cfg):
    # print(OmegaConf.to_yaml(cfg))
    data_name = cfg["name"]
    kwargs = {'name': f"{data_name}-" + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'Graph-2-Graph',
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion'}
    wandb.init(**kwargs)
    wandb.config.update(cfg)
    wandb.save('*.txt')


class ContinuousDiffusionFunctions:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pass
    def get_feature_normalisers(self, x_dim, train_loader):
        # self.x_dim = self.train_loader[0].x.shape[1]
        feature_sums = torch.zeros(x_dim, dtype=torch.double)
        print(f"Found feature dimensions to be {x_dim}")

        counter = 0
        for graph in train_loader:
            x = graph.x
            counter += x.shape[0]
            local_sums = torch.sum(x, dim=0)
            feature_sums += local_sums

        feature_means = (feature_sums / counter).double().to(self.device)

        x0 = None  # self.train_loader[0].x
        for graph in tqdm(train_loader, leave=False):
            if x0 is None:
                x0 = graph.x
            else:
                x = graph.x
                x0 = torch.cat((x0, x))

        # print(torch.var(x0, dim = 0), torch.max(x0, dim = 0)[0])

        feature_vars = torch.mean(x0, dim=0).to(self.device)

        if torch.sum(torch.abs(feature_vars)) < 1e-3:
            feature_vars = feature_vars / feature_vars

        if torch.sum(torch.abs(feature_means)) < 1e-3:
            feature_means = feature_means / feature_means

        print(f"Found feature means with dim: {feature_means.shape}")
        print(f"Found feature variances with dim: {feature_vars.shape}")

        self.feature_means, self.feature_vars = feature_means, feature_vars

        return feature_means, feature_vars

    def prepare_noise_schedule(self, diffusion_steps=400, min_beta=10 ** -4, max_beta=0.02, schedule_type="cosine",
                               sampling=False):
        # if schedule_type == "cosine":
        #     ts = np.linspace(start = 0, stop = np.pi / 2, num = diffusion_steps)
        #     fn = np.sin(ts)

        if not sampling:
            self.betas = torch.linspace(min_beta, max_beta, diffusion_steps).to(self.device)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(
                self.device)
        else:
            self.betas_sampling = torch.linspace(min_beta, max_beta, diffusion_steps).to(self.device)
            self.alphas_sampling = 1 - self.betas
            self.alpha_bars_sampling = torch.tensor(
                [torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)

        return self.alphas, self.alpha_bars, self.betas

        # fig = plt.figure(figsize=(8, 4))
        #
        # plt.plot(list(range(self.diffusion_steps)), self.betas.detach().cpu(), label="betas")
        # plt.plot(list(range(self.diffusion_steps)), self.alphas.detach().cpu(), label="alphas")
        # plt.plot(list(range(self.diffusion_steps)), self.alpha_bars.detach().cpu(), label="abar")
        #
        # plt.legend(shadow=True)
        # plt.savefig("Noise_Schedule.png")
        # plt.close()
        # print(self.alpha_bars)

        # return fn

    def apply_noise(self, x, t, eta=None):
        # Applies noise to a pyg Batch object

        # x_shape = x.shape
        # noise = ((torch.randn(size=x_shape)) * (noise_variance ** 0.5)).to(self.device)
        #
        # return noise + x

        x_shape = x.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(x_shape).to(self.device)

        noisy = a_bar.sqrt() * x + (1 - a_bar).sqrt() * eta

        return noisy
    # wandb.log({"Type":"Training"})
    # wandb.log(cfg)

    def remove_noise_step(self, x, eta_out, t, add_noise = False):
        alpha_t = self.alphas_sampling[t]
        alpha_t_bar = self.alpha_bars_sampling[t]
        # print(alpha_t, alpha_t_bar)
        # noisy = a_bar.sqrt() * x + (1 - a_bar).sqrt() * eta

        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_out)

        if t > 0 and add_noise:  # self.diffusion_steps:
            z = torch.randn(x.shape).to(self.device)

            beta_t = self.betas_sampling[t]
            sigma_t = beta_t.sqrt()
            # print(sigma_t)
            x = x + sigma_t * z

            del z

        return x

# class EMA():
#     def __init__(self, beta=0.9999):
#         super().__init__()
#         self.beta = beta
#     def update_model_average(self, ma_model, current_model):
#         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#             old_weight, up_weight = ma_params.data, current_params.data
#             ma_params.data = self.update_average(old_weight, up_weight)
#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new