import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx
import imageio
import wandb
import osmnx as ox
from Datasets import *
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from umap import UMAP
from seaborn import kdeplot


class ContinuousVectorMetrics:
    def __init__(self,
                 sim_fn = torch.cosine_similarity,
                 decomp_fn = UMAP):
        try:
            self.decomp = decomp_fn(n_components=2, n_jobs=6)
        except:
            self.decomp = decomp_fn(n_components=2)
        self.decomp_fitted = False
        self.sim_fn = sim_fn


    def validation_by_item(self, loader, wrapper, epoch_number = 0):
        # model refers to the actual GNN, wrapper to the diffusion construct using it

        x_trues = []
        x_preds = []

        total_loss = 0.
        for ib, batch in enumerate(loader):
            n_graphs = batch.num_graphs
            # print(f"batch has {n_graphs} graphs")
            x_slicing_indices = []
            for ig in range(n_graphs):
                example = batch.get_example(ig).x
                x_trues += [example.detach().cpu()]
                x_slicing_indices += [example.shape[0]]



        # for ib, batch in enumerate(loader):

            # for batch in loader:
            # if ib == 0:
            #     loss, x_pred = wrapper.sample_features(batch,
            #                                            visualise = f"val_vis/Epoch_{epoch_number}",
            #                                             gif_first = False)
            # else:
            loss, x_pred = wrapper.sample_features(batch,
                                                   visualise = None,
                                                   gif_first = False)

            total_loss += loss.item() / batch.num_graphs

            running_count = 0
            for indices in x_slicing_indices:
                # print(x_pred.shape, indices, running_count, running_count + indices)
                x_preds += [x_pred[running_count:running_count+indices, :].detach().cpu()]
                running_count += indices

        # x_trues = torch.cat(x_trues, dim=0)
        # print(x_trues[0], x_preds[0])
        x_trues = torch.cat(x_trues, dim = 0)
        x_preds = torch.cat(x_preds, dim = 0)
        # print(x_preds.shape, x_trues.shape)
        sim = self.vector_similarity(x_trues, x_preds)


        if not self.decomp_fitted:
            self.decomp.fit(x_trues.detach().cpu().numpy())
            self.decomp_fitted = True


        self.vis(x_trues, x_preds)

        return sim, total_loss




    def batch_vector_similarity(self, batch, x):
        # TODO: this might be checking similarity against vectors with loads of zeros
        # NOTE: torch geometric handles batches by preparing one large graph
        return self.vector_similarity(batch.x, x)

    def vector_similarity(self, true, pred):
        sim = self.sim_fn(true.cpu(), pred.cpu(), dim=1)
        return torch.mean(sim)

    # self.vis_fn(batch, x, sums, noise_amounts, visualise, gif_first = frames)
    def vis_batch(self, batch, x, sums, noise_amounts, visualise, gif_first = None):
        if type(batch) != torch.Tensor:
            true_x = batch.x
            pred_x = x
        else:
            true_x = batch

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9,5))

        ax1  = self.vis(true_x, pred_x, ax = ax1)

        ax2.plot(sums, label="x")
        ax2.plot(noise_amounts, label="eta")
        # ax3.set_yscale('log')
        ax2.legend(shadow=True)
        # plt.tight_layout()
        plt.savefig(f"{visualise}.png")
        plt.close()
        if "frame" not in visualise:
            try:
                wandb.log({"Sampling_PNG": wandb.Image(f"{visualise}.png")})
            except:
                pass


    def vis(self, true, pred, label = "projected_vectors", ax = None):
        if not self.decomp_fitted:
            true_projection = self.decomp.fit_transform(true.detach().cpu().numpy())
            self.decomp_fitted = True
        else:
            true_projection = self.decomp.transform(true.detach().cpu().numpy())
        pred_projection = self.decomp.transform(pred.detach().cpu().numpy())


        # print(true_projection.shape, pred_projection.shape)

        if ax is None:
            was_given_ax = False
            fig, ax = plt.subplots(figsize=(8,8))
        else:
            was_given_ax = True

        # print(true)
        # mean, var = np.mean(true.numpy(), axis = 0), np.std(true.numpy(), axis = 0)
        # mean = np.tile(mean, (true.shape[0], 1))
        # var = np.tile(var, (true.shape[0], 1))
        # random = np.random.randn(*true.numpy().shape)
        # # print(random.shape, mean.shape, var.shape)
        # random = random * var
        # random = random + mean
        # random_projection = self.decomp.transform(random)


        kdeplot(x = true_projection[:,0], y =  true_projection[:,1], color = "blue", ax = ax, alpha = 0.5, levels=5)
        kdeplot(x = pred_projection[:,0], y =  pred_projection[:,1], color="red", ax=ax, alpha = 0.5, levels=5)

        ax.scatter(*true_projection.T, label = "true", c = "blue", marker = "x", s = 3, zorder = 1000)
        ax.scatter(*pred_projection.T, label = "pred", c = "red", marker = "x", s = 3, zorder = 1000)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # kdeplot(x = random_projection[:,0], y =  random_projection[:,1], color="black", ax=ax, alpha = 0.25, levels=5)
        # ax.scatter(*random_projection.T, label = "random data", c = "black", marker = "o", s = 3, alpha = 0.5, zorder = 0)

        for line in ax.get_lines():
            line.set_alpha(0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(shadow = True)

        if not was_given_ax:
            plt.savefig(f"{label}.png")
            plt.close()

            try:
                wandb.log({"Projected_Vectors": wandb.Image(f"{label}.png")})
            except:
                pass
        else:
            return ax


    # plt.show()

if __name__ == "__main__":
    reddit_graph = download_reddit()
    graphs = ESWR(reddit_graph, 100, 128)

    pyg_graphs = [pyg.utils.from_networkx(g, group_node_attrs=["attrs"]) for g in graphs]

    mean_similarity = vector_similarity(pyg_graphs[0].x, pyg_graphs[1].x)

    pca_vis(pyg_graphs[0].x, pyg_graphs[1].x)

