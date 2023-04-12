import networkx as nx
from matplotlib import pyplot as plt
import random

from util import (
    display_tree,
    form_grid_graph,
    random_points,
    round_targets_to_graph,
)


def mst_from_pred(G, pred):
    mst = G.copy()
    mst.remove_edges_from(mst.edges())
    for v in pred.keys():
        u = pred[v]
        if u != None:
            mst.add_edge(u, v, weight=G.edges[u, v]["weight"])

    return mst


def random_tree_with_root(G, r):
    in_tree = {node: False for node in G.nodes()}
    pred = {node: None for node in G.nodes()}
    in_tree[r] = True

    for i in G.nodes():
        u = i
        while not in_tree[u]:
            pred[u] = random.choice(list(G[u]))
            u = pred[u]
        u = i
        while not in_tree[u]:
            in_tree[u] = True
            u = pred[u]

    return mst_from_pred(G, pred)


def main():
    target_count = 5
    graphx = 20
    graphy = 20
    scale = 0.05

    s, targets = random_points(target_count)

    G = form_grid_graph(s, targets, graphx, graphy)
    # G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
    # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
    # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

    round_targets_to_graph(G, s, targets)
    targets = [f"target {i}" for i in range(target_count)]
    s = "start"
    nx.set_node_attributes(G, 0, "paths")

    # budget = float("inf")
    budget = nx.minimum_spanning_tree(G).size(weight="weight") * 0.6

    # rescale weights
    for u, v in G.edges:
        G[u][v]["weight"] = G[u][v]["weight"] * scale

    mst = random_tree_with_root(G, s)
    display_tree(G, mst)


if __name__ == "__main__":
    main()
