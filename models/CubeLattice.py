"""
Generates cubes (currently only grids...)
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx
import imageio
import wandb

def cube_gif_vis(batch, frames, sums, noise_amounts, label):
    os.mkdir(f"{label}")
    framedir = f"{label}/frames"

    frame_every = int(len(noise_amounts) / len(frames))

    os.mkdir(framedir)
    framenames = []
    for frame_idx, frame in enumerate(tqdm(frames, leave = False)):
        cube_val_vis(batch, frame, sums[:frame_idx*frame_every], noise_amounts[:frame_idx*frame_every], f"{framedir}/{frame_idx}")
        framenames.append(f"{framedir}/{frame_idx}.png")

    with imageio.get_writer(f"{label}.gif", mode="I") as writer:
        for framename in framenames:
            image = imageio.imread(framename)
            writer.append_data(image)

    wandb.log({"Sampling_GIF": wandb.Image(f"{label}.gif")})

def cube_val_vis(batch, x, sums, noise_amounts, label, gif_first = False):
    if gif_first != False:
        cube_gif_vis(batch, gif_first, sums, noise_amounts, label)

    val_graph = batch.to_data_list()[0]
    # print(val_graph.x.shape[0])


    nx_graph_val  = nx.Graph(to_networkx(val_graph, node_attrs=["x"]))
    val_graph.x = x[:val_graph.x.shape[0],:]
    nx_graph_pred = nx.Graph(to_networkx(val_graph, node_attrs=["x"]))
    # print(nx_graph_val.nodes(data=True))
    #
    #
    # pos_val = attributes_to_position(nx_graph_val)
    # pos_pred = attributes_to_position(nx_graph_pred)

    fig = plt.figure(figsize=(12,4))

    if val_graph.x.shape[1] == 2:
        projection = "2d"
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
    else:
        projection = "3d"

        ax1 = fig.add_subplot(131, projection=projection)
        ax2 = fig.add_subplot(132, projection=projection)


    ax3 = fig.add_subplot(133)

    ax1 = cube_vis(nx_graph_val, ax = ax1)
    ax2 = cube_vis(nx_graph_pred, ax=ax2)

    ax3.plot(sums, label="x")
    ax3.plot(noise_amounts, label="eta")
    ax3.set_yscale('log')
    ax3.legend(shadow=True)
    # plt.tight_layout()
    plt.savefig(f"{label}.png")
    plt.close()
    if "frame" not in label:
        wandb.log({"Sampling_PNG": wandb.Image(f"{label}.png")})


def cube_vis(graph, ax = None):
    pos = attributes_to_position(graph)


    node_xyz = np.array([pos[v] for v in sorted(graph)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph.edges()])


    if ax is None:
        fig = plt.figure()
        if node_xyz.shape[1] == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")

    node_colors = [n for n in graph.nodes]

    ax.scatter(*node_xyz.T, s=100, ec="w", c = node_colors, cmap = "viridis")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    if ax is None:
        plt.savefig("Cube_Example.png")
    else:
        return ax

def attributes_to_position(graph):
    positions = {}

    for n in graph.nodes(data=True):
        positions[n[0]] = n[1]["x"]

    return positions

def get_cube(max_length = 6):
    n_in_each_dim = np.random.randint(2, max_length, size = 2)
    max_dim = np.max(n_in_each_dim)


    graph = nx.grid_graph(n_in_each_dim.tolist())
    for node in graph:
        graph.nodes[node]["attrs"] = [n  for n in node] # list(node) / max_dim

    graph = nx.convert_node_labels_to_integers(graph)

    # This block includes the node id (ie position in graph) as an attribute to target
    # for node in graph.nodes:
    #     graph.nodes[node]["attrs"] = [int(node)] + graph.nodes[node]["attrs"]
    # print(graph.nodes(data=True))

    for node in graph.nodes:
        graph.nodes[node]["attrs"] = np.array(graph.nodes[node]["attrs"]).astype(float)

    return graph

def get_cube_dataset(n_graphs, max_graph_size):

    graphs = [get_cube(max_length = max_graph_size) for _ in tqdm(range(n_graphs), leave=False)]
    # graphs = [attributes_to_position(g) for g in graphs]

    return graphs

if __name__ == "__main__":
    print("CubeLattice is Main")
    g = get_cube()
    pos = attributes_to_position(g)

    print(g, pos)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    node_xyz = np.array([pos[v] for v in sorted(g)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])

    ax.scatter(*node_xyz.T, s=100, ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    plt.show()

    # nx.draw(g, pos = pos)