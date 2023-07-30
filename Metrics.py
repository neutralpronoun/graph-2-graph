import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.autograd import Function
from typing import Any, List, Optional, Union
import scipy
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

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

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


    def validation_by_item(self, loader, wrapper, discriminator = None, epoch_number = 0):
        # model refers to the actual GNN, wrapper to the diffusion construct using it

        x_trues = []
        x_preds = []
        x_preds_batch = []
        # G = nx.Graph()

        total_loss = 0.
        counter = 0
        for ib, batch in enumerate(tqdm(loader, leave=False)):
            n_graphs = batch.num_graphs

            x_slicing_indices = []
            for ig in range(n_graphs):
                example = batch.get_example(ig).x
                x_trues += [example.detach().cpu()]
                x_slicing_indices += [example.shape[0]]

            loss, x_pred = wrapper.sample_features(batch,
                                                   visualise = None,
                                                   gif_first = False)
            x_preds_batch.append(x_pred.detach().cpu())

            total_loss += loss.item() / batch.num_graphs
            counter += 1

            running_count = 0
            for indices in x_slicing_indices:
                i_x_pred = x_pred[running_count:running_count+indices, :]
                x_preds += [i_x_pred.detach().cpu()]
                running_count += indices

            del x_pred

        if discriminator is not None:
            discriminator.epoch(loader, x_generated=x_preds_batch)

        x_trues = torch.cat(x_trues, dim = 0)
        x_preds = torch.cat(x_preds, dim = 0)
        sim = self.vector_similarity(x_trues, x_preds)
        wass = self.MeanWasserstein(x_trues, x_preds)

        metrics = {"Similarity":sim,
                   "Wasserstein":wass}


        if not self.decomp_fitted:
            self.decomp.fit(x_trues.detach().cpu().numpy())
            self.decomp_fitted = True

        if x_trues.shape[1] > 2:
            self.vis(x_trues, x_preds)
        else:
            self.vis_hist(x_trues, x_preds)

        del x_trues, x_preds

        return metrics, total_loss / counter

    def MeanWasserstein(self, true, pred):
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        # mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        # mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        # print("\n", true.isinf().any(), pred.isinf().any(), true.isnan().any(), pred.isnan().any())
        pred = pred.to(true.dtype)
        # print(true.isinf().any(), pred.isinf().any(), true.isnan().any(), pred.isnan().any(), "\n")
        mean_real = torch.mean(true, dim = 0)
        mean_pred = torch.mean(pred, dim=0).to(mean_real.dtype)

        cov_real = torch.cov(true)
        cov_pred = torch.cov(pred).to(mean_real.dtype)

        # print(cov_pred, cov_real)

        return self.ComputeWasserstein(mean_real, cov_real, mean_pred, cov_pred)

    def ComputeWasserstein(self, mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
        r"""Adjusted version of `Fid Score`_

        The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
        and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

        Args:
            mu1: mean of activations calculated on predicted (x) samples
            sigma1: covariance matrix over activations calculated on predicted (x) samples
            mu2: mean of activations calculated on target (y) samples
            sigma2: covariance matrix over activations calculated on target (y) samples
            eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

        Returns:
            Scalar value of the distance between sets.
        """
        diff = mu1 - mu2
        # print(mu1.isinf().any(), mu2.isinf().any(), mu1.isnan().any(), mu2.isnan().any())
        # print(sigma1.isinf().any(), sigma2.isinf().any(), sigma1.isnan().any(), sigma2.isnan().any())
        # print(sigma1.dtype, sigma2.dtype)
        covmean = sqrtm(sigma1.mm(sigma2))
        # Product might be almost singular
        if not torch.isfinite(covmean).all():
            # rank_zero_info(
            #     f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
            offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
            covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

        tr_covmean = torch.trace(covmean)
        return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

        #
        # cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        # cov_real = cov_real_num / (self.real_features_num_samples - 1)
        # cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        # cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        # return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)




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

        # pos = {}
        # print(G, pred_projection.T.shape)
        # for node in G:
        #     xy = pred_projection.T
        #     pos[node] = [xy[0, node], xy[1, node]]
        #
        # nx.draw_networkx_edges(G, pos = pos, alpha=0.5, ax = ax)

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

        ax.set_title(f"{self.decomp}")

        if not was_given_ax:
            plt.savefig(f"{label}.png")
            plt.close()

            try:
                wandb.log({"Projected_Vectors": wandb.Image(f"{label}.png")})
            except:
                pass
        else:
            return ax

    def vis_hist(self, true, pred, label = "histogram"):
        # true_values = torch.flatten(true).cpu().numpy()
        # pred_values = torch.flatten(pred).cpu().numpy()

        fig, ax = plt.subplots(figsize = (8,6))

        ax.scatter(true.cpu().numpy()[:, 0], true.cpu().numpy()[:, 1], label="True", marker = "+", alpha=0.5)
        ax.scatter(pred.cpu().numpy()[:, 0], pred.cpu().numpy()[:, 1], label="Pred", marker = "x", alpha=0.5)


        # ax.hist(true_values, label = "True values", histtype="step", bins = 30)
        # ax.hist(pred_values, label = "Pred values", histtype="step", bins = 30)

        ax.legend(shadow=True)

        # if not was_given_ax:
        plt.savefig(f"{label}.png")
        plt.close()

        try:
            wandb.log({"Projected_Vectors": wandb.Image(f"{label}.png")})
        except:
            pass
        # else:
        #     return ax


    # plt.show()

if __name__ == "__main__":
    reddit_graph = download_reddit()
    graphs = ESWR(reddit_graph, 100, 128)

    pyg_graphs = [pyg.utils.from_networkx(g, group_node_attrs=["attrs"]) for g in graphs]

    mean_similarity = vector_similarity(pyg_graphs[0].x, pyg_graphs[1].x)

    pca_vis(pyg_graphs[0].x, pyg_graphs[1].x)

