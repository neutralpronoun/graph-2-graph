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
import osmnx as ox
from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler
from Datasets import vis_small_graph, CSWR
from networkx import community as comm

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

    if val_graph.x.shape[1] < 3:
        projection = "2d"
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
    else:
        projection = "3d"

        ax1 = fig.add_subplot(131, projection=projection)
        ax2 = fig.add_subplot(132, projection=projection)


    ax3 = fig.add_subplot(133)

    if val_graph.x.shape[1] != 1:
        ax1 = cube_vis(nx_graph_val, ax = ax1)
        ax2 = cube_vis(nx_graph_pred, ax=ax2)
    else:
        ax1 = graph_vis(nx_graph_val, ax = ax1)
        ax2 = graph_vis(nx_graph_pred, ax = ax2)

    ax3.plot(sums, label="x")
    ax3.plot(noise_amounts, label="eta")
    # ax3.set_yscale('log')
    ax3.legend(shadow=True)
    # plt.tight_layout()
    plt.savefig(f"{label}.png")
    plt.close()
    if "frame" not in label:
        wandb.log({"Sampling_PNG": wandb.Image(f"{label}.png")})


def graph_vis(graph, ax = None):
    pos = nx.spring_layout(graph, seed=42)


    max_dim = 3

    # node_xyz = np.array([pos[v] for v in sorted(graph)])
    # edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph.edges()])
    #
    # if node_xyz.T.shape[1] > max_dim:
    #     node_xyz = np.array([pos[v][:max_dim] for v in sorted(graph)])
    #     edge_xyz = np.array([(pos[u][:max_dim], pos[v][:max_dim]) for u, v in graph.edges()])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    node_colors = [n[1]["x"] for n in graph.nodes(data=True)]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, ax = ax, node_size=50)
    nx.draw_networkx_edges(graph, pos, ax = ax)
    # ax.set_title(f"Mean coord: {np.around(np.mean(node_xyz.T), decimals=3)}")

    # # Plot the edges
    # for vizedge in edge_xyz:
    #     ax.plot(*vizedge.T, color="tab:gray")

    if ax is None:
        plt.savefig("Cube_Example.png")
    else:
        return ax

def cube_vis(graph, ax = None):
    pos = attributes_to_position_dict(graph)


    max_dim = 3

    node_xyz = np.array([pos[v] for v in sorted(graph)])
    # print(node_xyz)
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph.edges()])

    if node_xyz.T.shape[1] > max_dim:
        node_xyz = np.array([pos[v][:max_dim] for v in sorted(graph)])
        edge_xyz = np.array([(pos[u][:max_dim], pos[v][:max_dim]) for u, v in graph.edges()])


    if ax is None:
        fig = plt.figure()
        if node_xyz.shape[1] == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")

    node_colors = [n for n in graph.nodes]

    ax.scatter(*node_xyz.T, s=50, ec="b", c = node_colors, cmap = "viridis")
    ax.set_title(f"Mean coord: {np.around(np.mean(node_xyz.T), decimals=3)}")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    if ax is None:
        plt.savefig("Cube_Example.png")
    else:
        return ax

def attributes_to_position_dict(graph):
    positions = {}

    for n in graph.nodes(data=True):
        try:
            positions[n[0]] = n[1]["x"]
        except:
            positions[n[0]] = n[1]["attrs"]

    return positions

def attributes_to_position_array(graph):
    positions = np.zeros((2, graph.order()))

    for n_idx, n in enumerate(graph.nodes(data=True)):
        positions[0, n_idx] = n[1]["x"]
        positions[1, n_idx] = n[1]["y"]


    return positions

def get_cube(max_length = 6, very_easy = False):
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
        graph.nodes[node]["attrs"] = np.ones(2) if very_easy else np.array(graph.nodes[node]["attrs"]).astype(float)

    return graph

def get_cube_dataset(n_graphs, max_graph_size):

    graphs = [get_cube(max_length = max_graph_size) for _ in tqdm(range(n_graphs), leave=False)]
    # graphs = [attributes_to_position(g) for g in graphs]

    return graphs

def get_triangular(max_length = 6, very_easy = False):
    n_in_each_dim = np.random.randint(2, max_length, size = 2)
    max_dim = np.max(n_in_each_dim)


    graph = nx.triangular_lattice_graph(*n_in_each_dim.tolist())
    # graph = nx.fast_gnp_random_graph(max_length, np.random.random()**2)
    clustering = nx.clustering(graph)
    for node in graph:
        # if graph.degree[node] == 0:
        #     graph.remove_node(node)
        # else:
        graph.nodes[node]["attrs"] = [clustering[node]] # [n  for n in node] # list(node) / max_dim
        # del graph.nodes[node]["pos"]

    graph = nx.convert_node_labels_to_integers(graph)

    # This block includes the node id (ie position in graph) as an attribute to target
    # for node in graph.nodes:
    #     graph.nodes[node]["attrs"] = [int(node)] + graph.nodes[node]["attrs"]
    # print(graph.nodes(data=True))

    for node in graph.nodes:
        graph.nodes[node]["attrs"] = np.ones(2) if very_easy else np.array(graph.nodes[node]["attrs"]).astype(float)

    return graph

def get_triangular_dataset(n_graphs, max_graph_size):

    graphs = [get_triangular(max_length = max_graph_size) for _ in tqdm(range(n_graphs), leave=False)]
    # graphs = [nx.fast_gnp_random_graph(max_graph_size, np.random.random()) for _ in range(n_graphs)]
    # graphs = [attributes_to_position(g) for g in graphs]

    return graphs

def get_random(max_length = 6, very_easy = False):
    # n_in_each_dim = np.random.randint(2, max_length, size = 2)
    # max_dim = np.max(n_in_each_dim)


    # graph = nx.triangular_lattice_graph(*n_in_each_dim.tolist())
    graph = nx.connected_watts_strogatz_graph(max_length, 5, 0.2*np.random.random())
    # clustering = nx.degree(graph)
    for node in graph:
        graph.nodes[node]["attrs"] = [graph.degree[node] / graph.order()] # [n  for n in node] # list(node) / max_dim
        # del graph.nodes[node]["pos"]

    graph = nx.convert_node_labels_to_integers(graph)

    # This block includes the node id (ie position in graph) as an attribute to target
    # for node in graph.nodes:
    #     graph.nodes[node]["attrs"] = [int(node)] + graph.nodes[node]["attrs"]
    # print(graph.nodes(data=True))

    for node in graph.nodes:
        graph.nodes[node]["attrs"] = np.ones(2) if very_easy else np.array(graph.nodes[node]["attrs"]).astype(float)

    return graph

def get_random_dataset(n_graphs, max_graph_size):

    graphs = [get_random(max_length = max_graph_size) for _ in tqdm(range(n_graphs), leave=False)]
    # graphs = [nx.fast_gnp_random_graph(max_graph_size, np.random.random()) for _ in range(n_graphs)]
    # graphs = [attributes_to_position(g) for g in graphs]

    return graphs

def get_ring(max_length = 36):

    num_points = np.random.randint(4, max_length)
    # print(num_points)

    angles = np.linspace(0, 2*np.pi, num=num_points + 1)
    g = nx.Graph()

    xs, ys = np.cos(angles), np.sin(angles)
    node_names = [i for i in range(angles[:-1].shape[0])]

    for i_angle, angle in enumerate(angles[:-1]):
        x, y = xs[i_angle], ys[i_angle]
        g.add_node(i_angle, attrs = np.array([x, y]))

    for i in range(angles.shape[0] - 1):
        try:
            g.add_edge(node_names[i-1], node_names[i])
            g.add_edge(node_names[i], node_names[i+1])
        except:
            pass

    return g

def get_ring_dataset(n_graphs, max_graph_size):

    graphs = [get_ring(max_length = max_graph_size) for _ in tqdm(range(n_graphs), leave=False)]
    # print(graphs[0].nodes(data=True))
    # graphs = [attributes_to_position(g) for g in graphs]

    return graphs

def normalise_coordinates(graph):
    coords = attributes_to_position_array(graph)

    mean = np.mean(coords, axis = 1)
    var  = np.std(coords, axis = 1)

    print(var)

    coords[0,:] = coords[0, :] - mean[0]
    coords[1, :] = coords[1, :] - mean[1]
    coords[0, :] = coords[0,:] / var[0]
    coords[1, :] = coords[1, :] / var[1]

    for n_idx, node in enumerate(graph.nodes(data = True)):
        for key in list(node[1].keys()):
            del graph.nodes(data=True)[node[0]][key]
        graph.nodes[node[0]]["attrs"] = coords[:, n_idx]

    return graph

def remove_edge_attributes(graph):
    for n1, n2, d in list(graph.edges(data=True)):
        graph.remove_edge(n1, n2)
        graph.add_edge(n1, n2)
        # for k in list(d.keys()):
        #     del d[k]

    return graph

def road_dataset(n_graphs, max_graph_size, place = "Bristol, UK"):
    print("started road dataset")
    graph = nx.convert_node_labels_to_integers(nx.Graph(ox.graph_from_place(place)))
    # ox.plot_graph(ox.graph_from_place('Bristol, UK'), node_size=0)
    graphs = []
    n_runs = 5
    max_size=128
    for run in tqdm(range(n_runs), leave=False):
        partition = comm.louvain_communities(graph, resolution=5)

        for part in partition:
            g = graph.subgraph(part)
            if g.order() <= max_size:
                graphs.append(nx.Graph(g))

    # sampler = MetropolisHastingsRandomWalkSampler(number_of_nodes=np.random.randint(max_graph_size))
    #
    #
    #
    # graphs = [nx.convert_node_labels_to_integers(sampler.sample(bristol_network)) for _ in tqdm(range(n_graphs), leave=False)]
    # graphs = [normalise_coordinates(graph) for graph in graphs]
    # graphs = [remove_edge_attributes(graph) for graph in graphs]

    # for node in graphs[0].nodes(data=True):
    #     print(node)
    #
    # # ox.plot_graph(graphs[0])
    #
    # graphs[0] = normalise_coordinates(graphs[0])
    # for node in graphs[0].nodes(data=True):
    #     print(node)

    # pos = attributes_to_position_dict(graphs[0])

    # nx.draw(graphs[0], pos = pos)
    # cube_val_vis(batch, x, sums, noise_amounts, label, gif_first=False)
    fig, axes = plt.subplots(nrows = 8, ncols = 12, figsize=(24,16))

    for ir, row in enumerate(axes):
        for ic, ax in enumerate(row):
            ax = vis_small_graph(graphs[ir*12 + ic], ax = ax)
    plt.tight_layout(h_pad=0, w_pad=0)
    # plt.show()
    plt.savefig("Mum_roads.png", dpi=300)
    # try:
    #     wandb.log({"Train_examples": wandb.Image(f"Train_examples.png")})
    # except:
    #     pass
    # plt.close()

    return graphs




if __name__ == "__main__":
    road_dataset(200, 256)
    # print("CubeLattice is Main")
    # g = get_cube()
    # pos = attributes_to_position(g)
    #
    # print(g, pos)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    #
    # node_xyz = np.array([pos[v] for v in sorted(g)])
    # edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])
    #
    # ax.scatter(*node_xyz.T, s=100, ec="w")
    #
    # # Plot the edges
    # for vizedge in edge_xyz:
    #     ax.plot(*vizedge.T, color="tab:gray")
    #
    # plt.show()

    # nx.draw(g, pos = pos)