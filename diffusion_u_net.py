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
from Diffusion import *
from discriminator import Discriminator
# os.chdir("graph-2-graph")

import sys,os
sys.path.append(os.getcwd())


class DiffusionUNet(torch.nn.Module):
    def __init__(self, nx_graph_list, cfg):
        super(DiffusionUNet, self).__init__()

        batch_size = int(cfg["batch_size"])
        val_prop = float(cfg["val_prop"])
        test_prop = float(cfg["test_prop"])

        hidden_dim = int(cfg["hidden_dim"])
        num_layers = int(cfg["n_layers"])
        extra_features = cfg["extra_features"]


        min_beta, max_beta = 10**float(cfg["min_beta"]), 10**float(cfg["max_beta"])
        min_beta_sampling, max_beta_sampling = 10**float(cfg["min_beta_sampling"]), 10**float(cfg["max_beta_sampling"])

        diffusion_steps = int(cfg["diffusion_steps"])
        diffusion_steps_sampling = int(cfg["diffusion_steps_sampling"])

        self.val_every = int(cfg["val_every"])
        self.vis_every = int(cfg["vis_every"])

        if diffusion_steps_sampling >= diffusion_steps:
            diffusion_steps_sampling = diffusion_steps - 1

        vis_fn = "pca"

        self.val_fn = None
        self.dist_weighting = cfg["distribution_weighting"]
        self.add_noise = bool(cfg["add_noise_in_sampling"])

        if cfg["feature_type"] == "continuous":
            self.feat_type = "cont"
            self.diff_handler = ContinuousDiffusionFunctions()
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif cfg["feature_type"] == "discrete":
            self.feat_type = "disc"
            self.diff_handler = DiscreteDiffusionFunctions()
            self.loss_fn = torch.nn.BCELoss()
            # self.loss_fn = torch.nn.MSELoss(reduction="mean")
            # self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


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
                dataset = QM9(root = datadir)

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

            # print(train_graphs[0], pyg.utils.from_networkx(train_graphs[0]))
            # print(list(nx_graph_list[0].nodes(data=True)))
            self.x_dim = list(nx_graph_list[0].nodes(data=True))[0][1]["attrs"].shape[0]

            self.train_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in train_graphs],
                                               batch_size=batch_size)
            self.val_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in val_graphs],
                                               batch_size=3)
            self.test_loader = [pyg.utils.from_networkx(g, group_node_attrs=all) for g in test_graphs]



        if extra_features == "cycles":
            self.extra_features = SimpleNodeCycleFeatures()# ExtraFeatures()
            self.features_dim = 4


        if cfg["model"] == "unet":
            self.model = GraphUNet(in_channels=self.x_dim + 1 + self.features_dim, # Currently passing t as a node-level feature
                                     hidden_channels=hidden_dim,
                                     out_channels=self.x_dim,
                                     depth=num_layers,
                                     pool_ratios=0.2,
                                     out_sigmoid=self.feat_type == "disc").to(self.device)
        elif cfg["model"] == "gat":
            self.model = GAT(in_channels = self.x_dim + self.features_dim + 1, # +1 for timesteps
                             out_channels= self.x_dim,
                             hidden_channels=hidden_dim,
                             num_layers = num_layers).to(self.device)

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



        # self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.0001)


        self.diffusion_steps = diffusion_steps
        self.diffusion_steps_sampling = diffusion_steps_sampling


        if vis_fn == "pca" and self.feat_type == "cont":
            self.val_class = ContinuousVectorMetrics()
            self.vis_fn = self.val_class.vis_batch
            self.val_fn = self.val_class.batch_vector_similarity
        else:
            self.val_class = None
            self.vis_fn = None
        # elif vis_fn != "colormap":
        #
        #     self.vis_fn = cube_val_vis
        # else:
        #     self.vis_fn = colormap_vis
        if cfg["use_discriminator"]:
            self.discriminator = Discriminator(self.x_dim)
        else:
            self.discriminator = None


        # self.sigmas = self.noise_schedule(diffusion_steps, schedule_type)
        self.alphas, self.alpha_bars, self.betas = self.diff_handler.prepare_noise_schedule(diffusion_steps = diffusion_steps, min_beta=min_beta, max_beta=max_beta)
        self.alphas_sampling, self.alpha_bars_sampling, self.betas_sampling = self.diff_handler.prepare_noise_schedule(diffusion_steps=diffusion_steps_sampling, min_beta=min_beta_sampling, max_beta=max_beta_sampling, sampling=True)

        if cfg["feature_type"] == "continuous":
            self.feature_means, self.feature_vars = self.diff_handler.get_feature_normalisers(self.x_dim, self.train_loader)
        elif cfg["feature_type"] == "discrete":
            self.feature_marginals = self.diff_handler.get_feature_marginals(self.x_dim, self.train_loader)

        if type(nx_graph_list) is str:
            try:
                self.train_loader = self.train_loader.to_datapipe().batch_graphs(batch_size=batch_size)
                self.val_loader = self.val_loader.to_datapipe().batch_graphs(batch_size=2)
            except:
                pass

    def train(self, n_epochs, gif_first = False):
        self.model.train()

        pbar = tqdm(range(n_epochs))

        losses = []
        for epoch_number, epoch in enumerate(pbar):

            epoch_loss = 0.0
            pbar_batch = tqdm(self.train_loader, leave=False, colour="#005500")

            for ib, batch in enumerate(pbar_batch):
                t = np.random.randint(self.diffusion_steps)
                x0 = batch.x.float().to(self.device)  # self.apply_noise(batch.x.float().to(self.device), t - 1)

                if self.feat_type == "cont":
                    x0 = ((x0 - self.feature_means) / self.feature_vars).float()
                eta = torch.randn_like(x0).to(self.device)

                noisy_feat = self.diff_handler.apply_noise(x0, t, eta=eta)
                noisy_feat = torch.cat((torch.full((batch.x.shape[0], 1), t).to(self.device),
                                        self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze(),
                                        noisy_feat), dim=1)
                out = self.model(noisy_feat, batch.edge_index.to(self.device))

                if self.feat_type == "cont":
                    loss = self.loss_fn(out, eta)
                else:
                    # print(out, x0)
                    loss = self.loss_fn(out, x0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() / batch.num_graphs
                wandb.log({f"Train/Batch-{self.loss_fn}":loss.item() / batch.num_graphs})

            wandb.log({f"Train/{self.loss_fn}":epoch_loss})


            if epoch_number  % self.vis_every == 0 or epoch_number == n_epochs - 1:
                val_loss = self.validation_epoch(gif_first = gif_first,
                                                 epoch_number = epoch_number)
            # elif epoch_number % self.val_every == 0:
            #     self.discriminator.epoch(self.val_loader, self)


            # pbar.set_description(f"Epoch: {epoch} Loss: {str(epoch_loss)[:4]} Validation: {str(val_loss)[:4]}")
            losses.append(epoch_loss)


    def validation_epoch(self, gif_first = False, epoch_number = 0):
        if "val_vis" not in os.listdir():
            os.mkdir("val_vis")
        val_loss = 0.0
        mean_val_metric = 0.

        if self.val_class is not None:
            val_metric, val_loss = self.val_class.validation_by_item(self.val_loader,
                                                                     self,
                                                                     discriminator=self.discriminator,
                                                                     epoch_number=epoch_number)
        else:
            for ib_val, val_batch in enumerate(self.val_loader):

                if ib_val == 0:
                    val_batch_loss, x_pred = self.sample_features(val_batch, visualise=f"val_vis/Epoch_{epoch_number}",
                                                                  gif_first=gif_first)
                else:
                    val_batch_loss, x_pred = self.sample_features(val_batch)

                val_loss += val_batch_loss.item() / val_batch.num_graphs

            # val_loss += val_batch_loss.item() / val_batch.num_graphs

        # self.discriminator.epoch(self.val_loader, self)
        # wandb.log({"Similarity": val_metric})
        wandb.log(val_metric)
        wandb.log({f"Val-{self.loss_fn}": val_loss})

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
        # sums = []
        # noise_amounts = []
        # every_frame = int(self.diffusion_steps / 100)
        # if gif_first:
        #     frames = []
        for t in sampling_pbar:
            eta_out = self.model(torch.cat((torch.full((batch.x.shape[0], 1), t).to(self.device),
                                  self.extra_features(pyg.utils.to_dense_adj(batch.edge_index.to(self.device)))[0].squeeze(),
                                  x), dim = 1),
                                 edge_index)

            if self.feat_type == "cont":
                x = self.diff_handler.remove_noise_step(x, eta_out, t, add_noise = self.add_noise)
            if self.feat_type == "disc":
                x = self.diff_handler.remove_noise_step(eta_out, t, add_noise = self.add_noise) # Don't need eta for this

            # if visualise:
            #     if self.feat_type == "cont":
            #         noise_amounts.append(torch.mean((eta_out * self.feature_vars) + self.feature_means).detach().cpu())
            #
            #         sums.append(torch.mean((x * self.feature_vars) + self.feature_means).detach().cpu())
            #     else:
            #         noise_amounts.append(torch.mean(eta_out).detach().cpu())
            #
            #         sums.append(torch.mean(x).detach().cpu())

        if self.feat_type == "cont":
            x = (x * self.feature_vars) + self.feature_means
            x = x.to("cpu")

        # print(x, batch.x.to(self.device))
        node_loss = self.loss_fn(x, batch.x.to(x.dtype).to(self.device))
        if self.feat_type == "cont":
            distribution_loss = self.loss_fn(torch.mean(x, dim=0), torch.mean(batch.x.to(self.device), dim=0))
        elif self.feat_type == "disc":
            distribution_loss = 0. # self.loss_fn(torch.mean(x), torch.mean(batch.x.to(self.device)))
        loss = node_loss + self.dist_weighting * distribution_loss

        wandb.log({f"Node-{self.loss_fn}":node_loss,
                   f"Component-Mean-{self.loss_fn}": distribution_loss})

        return loss, x


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
        if cfg["sampling_method"] == "ESWR":
            graphs = ESWR(reddit_graph, cfg["n_samples"], cfg["max_size"])
        elif cfg["sampling_method"] == "CSWR":
            graphs = CSWR(reddit_graph, cfg["n_samples"], cfg["max_size"])

    elif cfg["name"] == "facebook":
        reddit_graph = download_facebook()
        setup_wandb(cfg)
        if cfg["sampling_method"] == "ESWR":
            graphs = ESWR(reddit_graph, cfg["n_samples"], cfg["max_size"])
        elif cfg["sampling_method"] == "CSWR":
            graphs = CSWR(reddit_graph, cfg["n_samples"], cfg["max_size"])


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



    DUNet.train(cfg["n_epochs"])
    # DUNet.sample_features(DUNet.test_loader[0], visualise="Final")

if __name__ == "__main__":
    main()


