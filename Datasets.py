import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.utils.convert import to_networkx
import imageio
import wandb
import osmnx as ox
from littleballoffur.exploration_sampling import MetropolisHastingsRandomWalkSampler
from sklearn.preprocessing import OneHotEncoder
# from ToyDatasets import *
import pickle
import zipfile
import wget
from networkx import community as comm


def vis_small_graph(graph, ax = None):
    pos = nx.kamada_kawai_layout(graph)


    # max_dim = 3
    #
    # node_xyz = np.array([pos[v] for v in sorted(graph)])
    # # print(node_xyz)
    # edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph.edges()])
    #
    # if node_xyz.T.shape[1] > max_dim:
    #     node_xyz = np.array([pos[v][:max_dim] for v in sorted(graph)])
    #     edge_xyz = np.array([(pos[u][:max_dim], pos[v][:max_dim]) for u, v in graph.edges()])


    if ax is None:
        fig = plt.figure()
        # if node_xyz.shape[1] == 2:
        ax = fig.add_subplot(111)
        # else:
        #     ax = fig.add_subplot(111, projection="3d")

    node_colors = [n for n in graph.nodes]

    nx.draw_networkx_nodes(graph, pos = pos, node_size=50, edgecolors="b", node_color=node_colors, cmap="viridis", ax = ax)
    nx.draw_networkx_edges(graph, pos = pos, alpha=0.5, ax = ax)

    # ax.scatter(*node_xyz.T, s=50, ec="b", c = node_colors, cmap = "viridis")
    # ax.set_title(f"Mean coord: {np.around(np.mean(node_xyz.T), decimals=3)}")

    # # Plot the edges
    # for vizedge in edge_xyz:
    #     ax.plot(*vizedge.T, color="tab:gray")

    if ax is None:
        plt.savefig("Cube_Example.png")
    else:
        ax.axis('off')
        # ax.axis('off')
        return ax

def vis_big_graph(G, largest_cc=False, label=""):
    if largest_cc:
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        G = CGs[0]

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="sfdp", args='-Gsmoothing')

    fig, (ax) = plt.subplots(ncols=1, figsize=(6, 6))

    nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)
    try:
        nx.draw_networkx_nodes(G, node_size=1, pos=pos, ax=ax,
                               node_color=[node[1]["target"] for node in G.nodes(data=True)])
    except:
        pass

    # ax.set_title(label)
    ax.axis('off')

    plt.tight_layout(h_pad=0, w_pad=0, pad=0)
    plt.savefig(f"{label}.png", dpi=600)

def download_facebook(visualise = False):
    zip_url = "https://snap.stanford.edu/data/facebook_large.zip"
    # embedding_url = "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

    start_dir = os.getcwd()
    for _ in range(3):
        os.chdir('../')
    print(os.getcwd(), os.listdir())
    os.chdir("data")

    if "facebook-graph.npz" in os.listdir():
        with open("facebook-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph




    if "facebook_large" not in os.listdir():
        _ = wget.download(zip_url)
        with zipfile.ZipFile("facebook_large.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove("facebook_large.zip")
    # if "web-redditEmbeddings-subreddits.csv" not in os.listdir():
    #     embedding_data = wget.download(embedding_url)



    os.chdir("facebook_large")

    edgelist = pd.read_csv("musae_facebook_edges.csv")
    # graph = # nx.from_pandas_edgelist(df=edgelist, source="id_1", target="id_2")

    with open("musae_facebook_features.json", "r") as f:
        embeddings = json.load(f)

    all_tokens = set()

    for key in embeddings.keys():
        all_tokens = all_tokens | set(embeddings[key])
    all_tokens = np.array(list(all_tokens)).reshape(-1,1)

    # all_tokens = np.array([all_tokens | set(embeddings[key]) for key in embeddings.keys()][0]).reshape(-1,1)
    print(all_tokens)
    max_token = np.max(all_tokens)

    # encoder = OneHotEncoder()
    # encoder.fit(all_tokens)

    # embedding_df = pd.DataFrame(columns=[i for i in range(max_token)])

    one_hot_embeddings = {}

    for node in embeddings:
        one_hot = np.zeros(max_token + 1)
        one_hot[np.array(embeddings[node])] = 1.
        one_hot_embeddings[node] = one_hot# encoder.transform(np.array(embeddings[node]).reshape(-1,1))
        # print(embedding_df.head())
    # print(embedding_df.head())
    # print(one_hot_embeddings)
    # quit()





    # features =

    # embedding_column_names = ["COMPONENT", *[i for i in range(max_token)]]
    # embeddings = pd.DataFrame(one_hot_embeddings)
    # print(embeddings.head())
    # quit()
    #pd.read_csv("web-redditEmbeddings-subreddits.csv", names=embedding_column_names).transpose()
    # graph_data = pd.read_csv("soc-redditHyperlinks-title.tsv", sep = "\t")


    # graph = nx.from_pandas_edgelist(graph_data, source="SOURCE_SUBREDDIT", target="TARGET_SUBREDDIT")
    # print(graph)

    # CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    # CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    # graph = CGs[0]

    # vis_big_graph(graph, label = "Reddit")



    # embeddings.columns = embeddings.iloc[0]
    # embeddings = embeddings.drop(["COMPONENT"], axis = 0)


    graph = nx.Graph()

    for col in one_hot_embeddings.keys():
        graph.add_node(int(col), attrs=one_hot_embeddings[col].astype(float))
    # print(edgelist)
    sources = edgelist["id_1"].to_numpy()
    targets = edgelist["id_2"].to_numpy()

    # print(list(graph.nodes()))
    #
    # print(sources, targets)

    for i in range(sources.shape[0]):
        graph.add_edge(sources[i], targets[i])

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    # for node in list(graph.nodes(data = True)):
    #     print(node)

    # embedding_subreddits = set(embeddings.columns)
    # node_names = set(graph.nodes())
    #
    # print(embedding_subreddits - node_names)
    # print(node_names - embedding_subreddits)
    #
    #
    # for node in graph.nodes():
    #     graph[node]["attrs"] = embeddings[node]
    #
    # for node in graph.nodes(data=True):
    #     print(node)


    graph = nx.convert_node_labels_to_integers(graph)

    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    if visualise:
        vis_big_graph(graph, label="Reddit")

    with open("reddit-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)
    print(graph)
    # quit()
    return graph

def download_reddit(visualise = False):
    graph_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
    embedding_url = "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

    start_dir = os.getcwd()
    for _ in range(3):
        os.chdir('../')
    print(os.getcwd(), os.listdir())
    os.chdir("data")

    if "reddit-graph.npz" in os.listdir():
        with open("reddit-graph.npz", "rb") as f:
            graph = pickle.load(f)
        os.chdir('../')
        return graph

    if "soc-redditHyperlinks-title.tsv" not in os.listdir():
        graph_data = wget.download(graph_url)
    if "web-redditEmbeddings-subreddits.csv" not in os.listdir():
        embedding_data = wget.download(embedding_url)


    embedding_column_names = ["COMPONENT", *[i for i in range(300)]]
    embeddings = pd.read_csv("web-redditEmbeddings-subreddits.csv", names=embedding_column_names).transpose()
    graph_data = pd.read_csv("soc-redditHyperlinks-title.tsv", sep = "\t")

    embeddings.columns = embeddings.iloc[0]
    embeddings = embeddings.drop(["COMPONENT"], axis = 0)


    graph = nx.Graph()

    for col in embeddings.columns:
        graph.add_node(col, attrs=embeddings[col].to_numpy().astype(float))

    sources = graph_data["SOURCE_SUBREDDIT"].to_numpy()
    targets = graph_data["TARGET_SUBREDDIT"].to_numpy()

    for i in range(sources.shape[0]):
        graph.add_edge(sources[i], targets[i])

    for node in list(graph.nodes(data=True)):
        data = node[1]
        if len(data) == 0:
            graph.remove_node(node[0])

    graph = nx.convert_node_labels_to_integers(graph)
    CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    graph = CGs[0]
    graph = nx.convert_node_labels_to_integers(graph)

    if visualise:
        vis_big_graph(graph, label="Reddit")

    with open("reddit-graph.npz", "wb") as f:
        pickle.dump(graph, f)

    os.chdir(start_dir)

    return graph

def ESWR(graph, n_graphs, size):
    sampler = MetropolisHastingsRandomWalkSampler(number_of_nodes=size)
    graphs = [nx.convert_node_labels_to_integers(sampler.sample(graph)) for _ in tqdm(range(n_graphs), leave=False)]

    fig, axes = plt.subplots(nrows = 5, ncols = 5, figsize=(18,18))

    for ir, row in enumerate(axes):
        for ic, ax in enumerate(row):
            ax = vis_small_graph(graphs[ir*5 + ic], ax = ax)
    # plt.show()
    plt.savefig("Train_examples.png")
    plt.close()
    try:
        wandb.log({"Train_examples": wandb.Image(f"Train_examples.png")})
    except:
        pass

    return graphs

def CSWR(graph, n_runs, max_size, kwargs={"resolution":6}):

    graphs = []

    for run in tqdm(range(n_runs), leave=False):
        partition = comm.louvain_communities(graph, **kwargs)

        for part in partition:
            g = graph.subgraph(part)
            if g.order() <= max_size:
                graphs.append(nx.Graph(g))

    fig, axes = plt.subplots(nrows = 5, ncols = 5, figsize=(18,18))

    for ir, row in enumerate(axes):
        for ic, ax in enumerate(row):
            ax = vis_small_graph(graphs[ir*5 + ic], ax = ax)
    # plt.show()
    plt.savefig("Train_examples.png")
    plt.close()
    try:
        wandb.log({"Train_examples": wandb.Image(f"Train_examples.png")})
    except:
        pass

    return graphs



if __name__ == "__main__":
    reddit_graph = download_reddit()
    graphs = CSWR(reddit_graph, 5)




