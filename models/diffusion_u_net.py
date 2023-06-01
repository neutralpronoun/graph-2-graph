import torch
import os
import wandb
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from UNet import GraphUNet
print(os.getcwd())
print(os.listdir())
# os.chdir("../")
from CubeLattice import get_cube_dataset, cube_val_vis
# os.chdir("graph-2-graph")


# class DatasetFromNX(pyg.data.InMemoryDataset):
#     def __init__(self, networkx_graphs):
#         super().__init__(transform = None, process = None)
#         self.nx_graphs  = networkx_graphs
#         self.pyg_graphs = [pyg.utils.from_networkx(g, group_node_attrs=all) for g in self.nx_graphs]
#

def setup_wandb(data_name = "cube"):
    kwargs = {'name': f"{data_name}-" + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'Graph-2-Graph',
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'entity':'hierarchical-diffusion'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    wandb.log({"Type":"Training"})

class DiffusionUNet(torch.nn.Module):
    def __init__(self, nx_graph_list, hidden_dim,
                 val_prop = 0.2, test_prop = 0.2, batch_size = 100,
                 min_beta = 10 ** -4, max_beta = 0.002,
                 diffusion_steps = 2000, use_wandb = True,
                 vis_fn = "colormap", output_dir = "outputs"):
        super(DiffusionUNet, self).__init__()

        date_time = str(datetime.datetime.now())
        print(str(date_time), os.getcwd())
        date_time = date_time.replace(' ', '_')
        date_time = date_time.replace(r':', r'_')
        self.output_dir = output_dir + f"/{date_time}"
        os.mkdir(self.output_dir)
        os.chdir(self.output_dir)

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

            self.x_dim = list(nx_graph_list[0].nodes(data=True))[0][1]["attrs"].shape[0]

            self.train_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in train_graphs],
                                               batch_size=batch_size)
            self.val_loader = pyg.loader.DataLoader([pyg.utils.from_networkx(g, group_node_attrs=all) for g in val_graphs],
                                               batch_size=batch_size)
            self.test_loader = [pyg.utils.from_networkx(g, group_node_attrs=all) for g in test_graphs]




        self.model = GraphUNet(in_channels=self.x_dim + 1, # Currently passing t as a node-level feature
                                 hidden_channels=hidden_dim,
                                 out_channels=self.x_dim,
                                 depth=1,
                                 pool_ratios=0.5).to(self.device)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.01)


        self.diffusion_steps = diffusion_steps

        if vis_fn != "colormap":
            self.vis_fn = cube_val_vis
        else:
            self.vis_fn = colormap_vis


        # self.sigmas = self.noise_schedule(diffusion_steps, schedule_type)
        self.prepare_noise_schedule(diffusion_steps = diffusion_steps, min_beta=min_beta, max_beta=max_beta)
        self.get_feature_normalisers()
        if type(nx_graph_list) is str:
            self.train_loader = self.train_loader.to_datapipe().batch_graphs(batch_size=batch_size)
            self.val_loader = self.val_loader.to_datapipe().batch_graphs(batch_size=2)

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

        self.feature_vars = torch.max(x0, dim = 0)[0].to(self.device)
        print(f"Found feature means: {self.feature_means}")
        print(f"Found feature variances: {self.feature_vars}")





    def prepare_noise_schedule(self, diffusion_steps = 400, min_beta = 10 ** -4, max_beta = 0.02, schedule_type = "cosine"):
        # if schedule_type == "cosine":
        #     ts = np.linspace(start = 0, stop = np.pi / 2, num = diffusion_steps)
        #     fn = np.sin(ts)

        self.betas = torch.linspace(min_beta, max_beta, diffusion_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)


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

    def train(self, n_epochs, val_every = 25, gif_first = True):
        self.model.train()

        pbar = tqdm(range(n_epochs))

        losses = []
        val_losses, val_epochs = [], []
        for epoch_number, epoch in enumerate(pbar):

            epoch_loss = 0.0
            pbar_batch = tqdm(self.train_loader, leave=False, colour="#005500")

            for ib, batch in enumerate(pbar_batch):

                t = np.random.randint(self.diffusion_steps)

                x0 = batch.x.float().to(self.device)  # self.apply_noise(batch.x.float().to(self.device), t - 1)

                x0 = ((x0 / self.feature_vars) - 0.5).float()

                eta = torch.randn_like(x0).to(self.device)
                t_appended = torch.full((batch.x.shape[0], 1), t).to(self.device)

                noisy_feat = self.apply_noise(x0, t, eta=eta)
                noisy_feat = torch.cat((t_appended, noisy_feat), dim=1)

                out = self.model(noisy_feat, batch.edge_index.to(self.device))
                loss = self.loss_fn(out, eta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() / batch.num_graphs
                wandb.log({f"Batch-{self.loss_fn}":loss.item() / batch.num_graphs})

            wandb.log({f"{self.loss_fn}":epoch_loss})


            if epoch_number % val_every == 0 or epoch_number == n_epochs - 1:
                if "val_vis" not in os.listdir():
                    os.mkdir("val_vis")
                val_loss = 0.0
                for ib_val, val_batch in enumerate(self.val_loader):
                    if ib_val == 0:
                        val_batch_loss = self.sample_features(val_batch, visualise=f"val_vis/Epoch_{epoch_number}", gif_first=gif_first)
                    else:
                        val_batch_loss = self.sample_features(val_batch)


                    val_loss += val_batch_loss.item() / val_batch.num_graphs
                val_losses.append(val_loss)
                val_epochs.append(epoch_number)
                wandb.log({f"Val-{self.loss_fn}":val_loss})

            pbar.set_description(f"Epoch: {epoch} Loss: {str(epoch_loss)[:4]} Validation: {str(val_loss)[:4]}")
            # else:
            #     pbar.set_description(f"Epoch: {epoch} Loss: {str(epoch_loss)[:4]}")

                # if ib > 50:
                #     break
            losses.append(epoch_loss)



        plt.plot(losses, label = "Train Losses")
        plt.plot(val_epochs, val_losses, label = "Validation Losses (between x)")
        plt.yscale('log')
        plt.legend(shadow=True)
        plt.savefig("Losses.png")


    def sample_noise_limit(self, x_shape):
        eta = torch.randn(x_shape).to(self.device)

        return eta

    def sample_features(self, batch, visualise = None, gif_first = True):
        self.model.eval()
        x = self.sample_noise_limit(batch.x.shape).to(self.device)
        edge_index = batch.edge_index.to(self.device)
        sampling_pbar = tqdm(reversed(range(self.diffusion_steps)), leave=False)
        sums = []
        noise_amounts = []
        every_frame = int(self.diffusion_steps / 100)
        if gif_first:
            frames = []
        for t in sampling_pbar:
            t_appended = torch.full((batch.x.shape[0], 1), t).to(self.device)
            x_passed = torch.cat((t_appended, x), dim = 1)
            # print(x_passed)
            eta_out = self.model(x_passed, edge_index)
            # print(eta_out)
            alpha_t = self.alphas[t]
            alpha_t_bar = self.alpha_bars[t]
            # print(alpha_t, alpha_t_bar)
            # noisy = a_bar.sqrt() * x + (1 - a_bar).sqrt() * eta

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_out)

            # print(x, x_passed)

            if t > 0:  #self.diffusion_steps:
                z = torch.randn(batch.x.shape).to(self.device)

                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z


            if visualise:

                noise_amounts.append(torch.mean((eta_out + 0.5) * self.feature_vars).detach().cpu())

                sums.append(torch.mean((x+0.5) * self.feature_vars).detach().cpu())
                if gif_first and t % every_frame == 0:
                    frames.append(((x+0.5) * self.feature_vars).float())
            # print(x, eta_out, alpha_t, alpha_t_bar)
            # if t % 50 == 0:
            #     print(torch.sum(x), torch.sum(batch.x.detach().cpu()))
            # x = self.model(x, edge_index)

        x = ((x + 0.5) * self.feature_vars).float()
        # x = (x + 1) * self.feature_means



        if visualise is not None:
            if gif_first:
                self.vis_fn(batch, x, sums, noise_amounts, visualise, gif_first = frames)
            else:
                self.vis_fn(batch, x, sums, noise_amounts, visualise, gif_first=False)

        wandb.log({"Noise_Amounts":np.array(noise_amounts), "Mean_X":np.array(sums)})

        loss = self.loss_fn(x, batch.x.to(self.device))

        return loss



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


if __name__ == "__main__":

    # nx_dummy_graphs = [nx.grid_2d_graph(20,
    #                                     20) for _ in range(500)]
    # for nx_g in nx_dummy_graphs:
    #     n_nodes = nx_g.order()
    #     for n in nx_g.nodes:
    #         nx_g.nodes[n]["attrs"] = np.arange(256) / 256


    cube_graphs = get_cube_dataset(1000, max_graph_size=6)
    setup_wandb(data_name="cube")


    # DUNet = DiffusionUNet("CLUSTER", 100, batch_size=100, diffusion_steps=1000)
    DUNet = DiffusionUNet(cube_graphs,
                          100,
                          batch_size=25,
                          diffusion_steps=4000,
                          vis_fn="cube")



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



    DUNet.train(2000, val_every=250)
    DUNet.sample_features(DUNet.test_loader[0], visualise="Final")


