"""
Generates cubes (currently only grids...)
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def attributes_to_position(graph):
    positions = {}

    for n in graph.nodes(data=True):
        positions[n[0]] = n[1]["position"]

    return positions

def get_cube(max_length = 6):
    n_in_each_dim = np.random.randint(1, max_length, size = 3)
    max_dim = np.max(n_in_each_dim)


    graph = nx.grid_graph(n_in_each_dim.tolist())
    for node in graph:
        graph.nodes[node]["position"] = np.array(node).astype(float) / max_dim

    graph = nx.convert_node_labels_to_integers(graph)
    print(graph.nodes(data=True))

    return graph

def get_cube_dataset(n_graphs, max_graph_size):

    graphs = [get_cube(max_graph_size) for _ in range(n_graphs)]
    graphs = [attributes_to_position(g) for g in graphs]

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