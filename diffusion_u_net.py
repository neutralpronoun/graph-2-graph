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

import sys,os
sys.path.append(os.getcwd())



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
    print(OmegaConf.to_yaml(cfg))
    data_name = cfg["name"]
    kwargs = {'name': f"{data_name}-" + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'Graph-2-Graph',
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion'}
    wandb.init(**kwargs)
    wandb.config.update(cfg)
    wandb.save('*.txt')

    # wandb.log({"Type":"Training"})
    # wandb.log(cfg)

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

class DiffusionUNet(torch.nn.Module):
    def __init__(self, nx_graph_list, cfg):
        super(DiffusionUNet, self).__init__()

        batch_size = int(cfg["batch_size"])
        val_prop = float(cfg["val_prop"])
        test_prop = float(cfg["test_prop"])

        hidden_dim = int(cfg["hidden_dim"])
        num_layers = int(cfg["n_layers"])
        extra_features = cfg["extra_features"]


        min_beta, max_beta = float(cfg["min_beta"]), float(cfg["max_beta"])
        min_beta_sampling, max_beta_sampling = float(cfg["min_beta_sampling"]), float(cfg["max_beta_sampling"])

        diffusion_steps = int(cfg["diffusion_steps"])
        diffusion_steps_sampling = int(cfg["diffusion_steps_sampling"])

        if diffusion_steps_sampling >= diffusion_steps:
            diffusion_steps_sampling = diffusion_steps - 1

        vis_fn = "pca"

        self.val_fn = None


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if type(nx_graph_list) == str:
            if nx_graph_list == "protein":
                dataset = PygNodePropPredDataset(name = "ogbn-proteins")
                print(dataset)
                split_idx = dataset.get_idx_split()
                print(split_idx)
                self.train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
                self.val_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
                self.test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)

            elif nx_graph_list == "qm9":
                datadir = os.getcwd() + "/data/"
                dataset = QM9(root = datadir)#PygNodePropPredDataset(name = "ogbn-proteins")
                # print(dataset)
                # split_idx = dataset.get_idx_split()
                # print(split_idx)

                print(len(dataset))

                total_n = len(dataset)
                train_n, val_n, test_n = int(total_n * (1 - val_prop - test_prop)), int(total_n * (val_prop)), int(total_n * (test_prop))

                self.train_loader = DataLoader(dataset[:train_n], batch_size=batch_size, shuffle=True)
                self.val_loader = DataLoader(dataset[train_n : train_n + val_n], batch_size=batch_size, shuffle=True)
                self.test_loader = DataLoader(dataset[-test_n:], batch_size=batch_size, shuffle=False)



                self.x_dim = 11

            else:
                datadir = os.getcwd() + "/data/"
                print(datadir)
                self.train_loader = pyg.datasets.GNNBenchmarkDataset(root = datadir, name = "CIFAR10", split="train")[:1000] #.to_datapipe().batch_graphs(batch_size = batch_size)
                self.val_loader = pyg.datasets.GNNBenchmarkDataset(root=datadir, name="CIFAR10", split="val")[:20] # .to_datapipe().batch_graphs(batch_size = batch_size)
                self.test_loader = pyg.datasets.GNNBenchmarkDataset(root=datadir, name="CIFAR10", split="test") # .to_datapipe().batch_graphs(batch_size = batch_size)

                # print(self.train_loader[0].x)

                self.x_dim = 3


        else:
            n_graphs = len(nx_graph_list)
            n_train, n_val = int(n_graphs * (1 - val_prop - test_prop)), int(n_graphs * val_prop)
            train_graphs, val_graphs, test_graphs = nx_graph_list[:n_train], nx_graph_list[n_train:n_train+n_val], nx_graph_list[n_train+n_val:]

            print(train_graphs[0], pyg.utils.from_networkx(train_graphs[0]))
            # print(list(nx_graph_list[0].nodes(data=True)))
            self.x_dim = list(nx_graph_list[0].nodes(data=True))[0][1]["attrs"].shape[0]

            self.train_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in train_graphs],
                                               batch_size=batch_size)
            self.val_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in val_graphs],
                                               batch_size=int(batch_size/10))
            self.test_loader = [pyg.utils.from_networkx(g, group_node_attrs=all) for g in test_graphs]



        if extra_features == "cycles":
            self.extra_features = SimpleNodeCycleFeatures()# ExtraFeatures()
            self.features_dim = 4

        self.model = GraphUNet(in_channels=self.x_dim + 1 + self.features_dim, # Currently passing t as a node-level feature
                                 hidden_channels=hidden_dim,
                                 out_channels=self.x_dim,
                                 depth=num_layers,
                                 pool_ratios=0.25).to(self.device)

        # self.model = EdgeCNN(in_channels = self.x_dim + 1 + self.features_dim, # +1 for timesteps, +3 for cycles
        #                  out_channels= self.x_dim,
        #                  hidden_channels=hidden_dim,
        #                  num_layers = num_layers).to(self.device)


        # if ema_scheduler is not None:
        #     self.ema_scheduler = ema_scheduler
        #     self.netG_EMA = copy.deepcopy(self.model).to(self.device)
        #     self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        # else:
        #     self.ema_scheduler = None

        # self.model = GAT(in_channels = self.x_dim + self.features_dim + 1, # +1 for timesteps
        #                  out_channels= self.x_dim,
        #                  hidden_channels=hidden_dim,
        #                  num_layers = num_layers).to(self.device)

        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.0001)


        self.diffusion_steps = diffusion_steps
        self.diffusion_steps_sampling = diffusion_steps_sampling
        if vis_fn == "pca":
            self.val_class = ContinuousVectorMetrics()
            self.vis_fn = self.val_class.vis_batch
            self.val_fn = self.val_class.batch_vector_similarity
        elif vis_fn != "colormap":
            self.vis_fn = cube_val_vis

        else:
            self.vis_fn = colormap_vis

        # date_time = str(datetime.datetime.now())
        # print(str(date_time), os.getcwd())
        # date_time = date_time.replace(' ', '_')
        # date_time = date_time.replace(r':', r'_')
        # self.output_dir = output_dir + f"/{date_time}"
        # os.mkdir(self.output_dir)
        # os.chdir(self.output_dir)

        # self.sigmas = self.noise_schedule(diffusion_steps, schedule_type)
        self.prepare_noise_schedule(diffusion_steps = diffusion_steps, min_beta=min_beta, max_beta=max_beta)
        self.prepare_noise_schedule(diffusion_steps=diffusion_steps_sampling, min_beta=min_beta_sampling, max_beta=max_beta_sampling, sampling=True)
        self.get_feature_normalisers()
        if type(nx_graph_list) is str:
            try:
                self.train_loader = self.train_loader.to_datapipe().batch_graphs(batch_size=batch_size)
                self.val_loader = self.val_loader.to_datapipe().batch_graphs(batch_size=2)
            except:
                pass

        quit()
        # print(f"Noise schedule: {self.sigmas}")



    def get_feature_normalisers(self):
        # self.x_dim = self.train_loader[0].x.shape[1]
        feature_sums = torch.zeros(self.x_dim, dtype=torch.double)
        print(f"Found feature dimensions to be {self.x_dim}")

        counter = 0
        for graph in self.train_loader:

            x = graph.x
            counter += x.shape[0]
            local_sums = torch.sum(x, dim=0)
            feature_sums += local_sums

        self.feature_means = (feature_sums / counter).double().to(self.device)



        x0 = None # self.train_loader[0].x
        for graph in tqdm(self.train_loader, leave = False):
            if x0 is None:
                x0 = graph.x
            else:
                x = graph.x
                x0 = torch.cat((x0, x))

        # print(torch.var(x0, dim = 0), torch.max(x0, dim = 0)[0])

        self.feature_vars = torch.mean(x0, dim = 0).to(self.device)


        if torch.sum(torch.abs(self.feature_vars)) < 1e-3:
            self.feature_vars = self.feature_vars / self.feature_vars

        if torch.sum(torch.abs(self.feature_means)) < 1e-3:
            self.feature_means = self.feature_means / self.feature_means

        print(f"Found feature means: {self.feature_means}")
        print(f"Found feature variances: {self.feature_vars}")





    def prepare_noise_schedule(self, diffusion_steps = 400, min_beta = 10 ** -4, max_beta = 0.02, schedule_type = "cosine", sampling = False):
        # if schedule_type == "cosine":
        #     ts = np.linspace(start = 0, stop = np.pi / 2, num = diffusion_steps)
        #     fn = np.sin(ts)

        if not sampling:
            self.betas = torch.linspace(min_beta, max_beta, diffusion_steps).to(self.device)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)
        else:
            self.betas_sampling = torch.linspace(min_beta, max_beta, diffusion_steps).to(self.device)
            self.alphas_sampling = 1 - self.betas
            self.alpha_bars_sampling = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)


        fig = plt.figure(figsize = (8,4))

        plt.plot(list(range(self.diffusion_steps)), self.betas.detach().cpu(), label = "betas")
        plt.plot(list(range(self.diffusion_steps)), self.alphas.detach().cpu(), label = "alphas")
        plt.plot(list(range(self.diffusion_steps)), self.alpha_bars.detach().cpu(), label = "abar")

        plt.legend(shadow=True)
        plt.savefig("Noise_Schedule.png")
        plt.close()
        print(self.alpha_bars)

        # return fn



    def apply_noise(self, x, t, eta = None):
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


    def train_batch(self, batch, t):

        x0 = batch.x.float().to(self.device) # self.apply_noise(batch.x.float().to(self.device), t - 1)
        eta = torch.randn_like(x0).to(self.device)
        t_appended = torch.full((batch.x.shape[0], 1), t).to(self.device)



        noisy_feat = self.apply_noise(x0, t, eta = eta)


        # print(t_appended.shape, noisy_feat.shape)
        noisy_feat = torch.cat((t_appended, noisy_feat), dim = 1)

        # print(t)
        # fig, (ax1, ax2) =  plt.subplots(ncols=2)
        # ax1.imshow(x0.detach().cpu(), vmin = -1, vmax = 1)
        # ax2.imshow(noisy_feat.detach().cpu()[:, 1:], vmin = -1, vmax = 1)
        # plt.show()

        # inp    = self.apply_noise(batch.x.float().to(self.device), t)
        out = self.model(noisy_feat, batch.edge_index.to(self.device))
        loss = self.loss_fn(out, eta)



        return loss

    def train(self, n_epochs, val_every = 25, gif_first = False):
        self.model.train()

        pbar = tqdm(range(n_epochs))

        losses = []
        # val_losses, val_epochs = [], []
        for epoch_number, epoch in enumerate(pbar):

            epoch_loss = 0.0
            pbar_batch = tqdm(self.train_loader, leave=False, colour="#005500")

            for ib, batch in enumerate(pbar_batch):

                t = np.random.randint(self.diffusion_steps)

                x0 = batch.x.float().to(self.device)  # self.apply_noise(batch.x.float().to(self.device), t - 1)

                x0 = ((x0 - self.feature_means) / self.feature_vars).float()

                eta = torch.randn_like(x0).to(self.device)
                # t_appended = torch.full((batch.x.shape[0], 1), t).to(self.device)
                # x_cycles = self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze()

                noisy_feat = self.apply_noise(x0, t, eta=eta)

                # print(t_appended.shape, x_cycles.shape, noisy_feat.shape)
                # print(t_appended.shape, x_cycles.shape, noisy_feat.shape)



                # print(noisy_feat.shape)
                # try:
                noisy_feat = torch.cat((torch.full((batch.x.shape[0], 1), t).to(self.device),
                                        self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze(),
                                        noisy_feat), dim=1)
                out = self.model(noisy_feat, batch.edge_index.to(self.device))
                # except:
                #     print(f"t: {t_appended.shape}"
                #           f"cycles: {x_cycles.shape}"
                #           f"x: {noisy_feat.shape}"
                #           f"batch: {batch.x.shape}")
                loss = self.loss_fn(out, eta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if self.ema_scheduler is not None:
                #     if epoch_number > self.ema_scheduler['ema_start'] and epoch_number % self.ema_scheduler['ema_iter'] == 0:
                #         self.EMA.update_model_average(self.netG_EMA, self.model)

                epoch_loss += loss.item() / batch.num_graphs
                wandb.log({f"Train/Batch-{self.loss_fn}":loss.item() / batch.num_graphs})

            wandb.log({f"Train/{self.loss_fn}":epoch_loss})


            if epoch_number  % val_every == 0 or epoch_number == n_epochs - 1:
                val_loss = self.validation_epoch(gif_first = gif_first,
                                                 epoch_number = epoch_number)
                # val_epochs.append(epoch_number)
                # val_losses.append(val_loss)

            pbar.set_description(f"Epoch: {epoch} Loss: {str(epoch_loss)[:4]} Validation: {str(val_loss)[:4]}")
            # else:
            #     pbar.set_description(f"Epoch: {epoch} Loss: {str(epoch_loss)[:4]}")

                # if ib > 50:
                #     break
            losses.append(epoch_loss)



        # plt.plot(losses, label = "Train Losses")
        # plt.plot(val_epochs, val_losses, label = "Validation Losses (between x)")
        # plt.yscale('log')
        # plt.legend(shadow=True)
        # plt.savefig("Losses.png")

    def validation_epoch(self, gif_first = False, epoch_number = 0):
        if "val_vis" not in os.listdir():
            os.mkdir("val_vis")
        val_loss = 0.0
        mean_val_metric = 0.

        if self.val_class is not None:
            val_metric, val_loss = self.val_class.validation_by_item(self.val_loader, self, epoch_number=epoch_number)
        else:
            for ib_val, val_batch in enumerate(self.val_loader):

                if ib_val == 0:
                    val_batch_loss, x_pred = self.sample_features(val_batch, visualise=f"val_vis/Epoch_{epoch_number}",
                                                                  gif_first=gif_first)
                else:
                    val_batch_loss, x_pred = self.sample_features(val_batch)

                print(x_pred.shape)

                # if self.val_fn is not None:
                #     val_metric = self.val_fn(val_batch, x_pred).item()
                #     mean_val_metric += val_metric / val_batch.num_graphs
                #     # wandb.log({"Validation/Similarity": val_metric})

                val_loss += val_batch_loss.item() / val_batch.num_graphs

        wandb.log({"Similarity": val_metric})
        wandb.log({f"Val-{self.loss_fn}": val_loss})

        # wandb.log({"Validation/Similarity": val_metric})
        # wandb.log({f"Validation/Val-{self.loss_fn}": val_loss})
        self.model.train()

        return val_loss

    def sample_noise_limit(self, x_shape):
        eta = torch.randn(x_shape).to(self.device)

        return eta

    def sample_features(self, batch, visualise = None, gif_first = False):
        self.model.eval()
        x = self.sample_noise_limit(batch.x.shape).to(self.device)
        edge_index = batch.edge_index.to(self.device)
        sampling_pbar = tqdm(reversed(range(self.diffusion_steps_sampling)), leave=False)
        sums = []
        noise_amounts = []
        every_frame = int(self.diffusion_steps / 100)
        if gif_first:
            frames = []
        for t in sampling_pbar:
            # t_appended = torch.full((batch.x.shape[0], 1), t).to(self.device)
            # x_cycles = self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze()
            # print(torch.full((batch.x.shape[0], 1), t).shape,
            #       self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze().shape,
            #       x.shape)
            # try:
            # x_passed = torch.cat((torch.full((batch.x.shape[0], 1), t).to(self.device),
            #                       self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze(),
            #                       x), dim = 1)
            # except:
            #     print(f"t: {t_appended.shape}"
            #           f"cycles: {x_cycles.shape}"
            #           f"x: {x.shape}"
            #           f"batch: {batch.x.shape}")
            # print(x_passed)
            eta_out = self.model(torch.cat((torch.full((batch.x.shape[0], 1), t).to(self.device),
                                  self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze(),
                                  x), dim = 1),
                                 edge_index)
            # del x_passed
            alpha_t = self.alphas_sampling[t]
            alpha_t_bar = self.alpha_bars_sampling[t]
            # print(alpha_t, alpha_t_bar)
            # noisy = a_bar.sqrt() * x + (1 - a_bar).sqrt() * eta

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_out)

            # print(x, x_passed)

            # if t > 0:  #self.diffusion_steps:
            #     z = torch.randn(batch.x.shape).to(self.device)
            #
            #     beta_t = self.betas_sampling[t]
            #     sigma_t = beta_t.sqrt()
            #     # print(sigma_t)
            #     x = x + sigma_t * z
            #
            #     del z


            if visualise:

                noise_amounts.append(torch.mean((eta_out * self.feature_vars) + self.feature_means).detach().cpu())

                sums.append(torch.mean((x * self.feature_vars) + self.feature_means).detach().cpu())
                if gif_first and t % every_frame == 0:
                    frames.append(((x * self.feature_vars) + self.feature_means).detach().float().cpu().numpy())
            # print(x, eta_out, alpha_t, alpha_t_bar)
            # if t % 50 == 0:
            #     print(torch.sum(x), torch.sum(batch.x.detach().cpu()))
            # x = self.model(x, edge_index)

        # x = ((x + 0.5) * self.feature_vars).float()
        x = (x * self.feature_vars) + self.feature_means



        if visualise is not None:
            if gif_first:
                self.vis_fn(batch, x, sums, noise_amounts, visualise, gif_first = frames)
            else:
                self.vis_fn(batch, x, sums, noise_amounts, visualise, gif_first=False)

        # wandb.log({"Noise_Amounts":np.array(noise_amounts), "Mean_X":np.array(sums)})

        loss = self.loss_fn(x, batch.x.to(self.device))



        return loss, x



def colormap_vis(batch, x, sums, noise_amounts, label, gif_first = False):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    ax1.imshow(batch.x.detach().cpu())  # , vmin = 0, vmax = 1)
    ax2.imshow(x.detach().cpu())  # , vmin = 0, vmax = 1)
    ax3.plot(sums, label="x")
    ax3.plot(noise_amounts, label="eta")
    ax3.legend(shadow=True)

    plt.savefig(f"{label}.png")
    plt.close()
    # plt.show()

@hydra.main(version_base='1.1', config_path='configs', config_name="main")
def main(cfg : DictConfig) -> None:
    # print(cfg)
    # print(OmegaConf.to_yaml(cfg))
    # cfg = cfg["dataset"]
    # print(cfg)
    # print(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(cfg)
    # nx_dummy_graphs = [nx.grid_2d_graph(20,
    #                                     20) for _ in range(500)]
    # for nx_g in nx_dummy_graphs:
    #     n_nodes = nx_g.order()
    #     for n in nx_g.nodes:
    #         nx_g.nodes[n]["attrs"] = np.arange(256) / 256


    # cube_graphs = get_cube_dataset(1000, max_graph_size=5)
    # setup_wandb(data_name="qm9")
    if cfg["name"] == "hex-clustering":
        setup_wandb(cfg)
        graphs = get_triangular_dataset(int(cfg["n_samples"]), max_graph_size=cfg["max_size"])


    elif cfg["name"] == "ring-position":
        setup_wandb(cfg)
        graphs = get_ring_dataset(int(cfg["n_samples"]), max_graph_size=cfg["max_size"])


    elif cfg["name"] == "grid-position":
        setup_wandb(cfg)
        graphs = get_cube_dataset(int(cfg["n_samples"]), max_graph_size=cfg["max_size"])


    elif cfg["name"] == "random-clustering":
        setup_wandb(cfg)
        graphs = get_random_dataset(int(cfg["n_samples"]), max_graph_size=cfg["max_size"])


    elif cfg["name"] == "bike-networks":
        setup_wandb(cfg)
        graphs = road_dataset(int(cfg["n_samples"]), max_graph_size=cfg["max_size"])

    elif cfg["name"] == "reddit":
        reddit_graph = download_reddit()
        setup_wandb(cfg)
        graphs = ESWR(reddit_graph, cfg["n_samples"], cfg["max_size"])


    # ring_graphs = get_cube_dataset(1000, max_graph_size=6)
    # setup_wandb(data_name = "cube")

    # DUNet = DiffusionUNet("CLUSTER", 100, batch_size=100, diffusion_steps=1000)
    # DUNet = DiffusionUNet(cube_graphs,
    #                       200,
    #                       batch_size=25,
    #                       diffusion_steps=200,
    #                       vis_fn="cube")

    DUNet = DiffusionUNet(graphs, cfg)




    # fig, axes = plt.subplots(ncols=20)
    # ts = np.linspace(1, DUNet.diffusion_steps - 1, num = 19).astype(int)



    # x = DUNet.test_loader[0].x.to(self.device)
    # x = ((x - DUNet.feature_means) / DUNet.feature_vars).float()
    # axes[0].imshow(x.detach().cpu())
    #
    # for i, ax in enumerate(axes[1:]):
    #     t = ts[i]
    #
    #     x = DUNet.test_loader[0].x.to(self.device)
    #     x = ((x - DUNet.feature_means) / DUNet.feature_vars).float()
    #
    #
    #     noise = DUNet.apply_noise(x, t)
    #     ax.imshow(noise.detach().cpu())
    #
    #
    # # plt.show()
    # plt.savefig("Noise_Examples.png")
    # plt.close()



    DUNet.train(cfg["n_epochs"], val_every=cfg["val_every"])
    # DUNet.sample_features(DUNet.test_loader[0], visualise="Final")

if __name__ == "__main__":
    main()


