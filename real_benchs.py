import time
import os
import networkx as nx
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from algo import compute_tree, build_stiener_seed, compute_metric
from util import (
    form_grid_graph,
    round_targets_to_graph,
    form_hex_graph,
    form_triangle_graph,
    bcolors,
    display_graph,
)


def targets_from_file(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        start = lines[1].split(" ")[1:]
        start = (int(start[0][:-1]), int(start[1]))
        lines = lines[2:]
        lines = [line.split(" ")[1:] for line in lines]
        lines = [(int(line[0][:-1]), int(line[1])) for line in lines]
        return start, lines


def factory(file):
    s, targets = targets_from_file(file)

    # G = form_grid_graph(s, targets, graphx, graphy)
    G = form_grid_graph(s, targets, w - 1, w - 1, triangulate=False)
    # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
    # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

    round_targets_to_graph(G, s, targets)
    targets = [f"target {i}" for i in range(t)]
    s = "start"
    nx.set_node_attributes(G, 0, "paths")

    budget = float("inf")
    # budget = nx.minimum_spanning_tree(G).size(weight="weight") * 0.5

    # # rescale weights
    # for u, v in G.edges:
    #     G[u][v]["weight"] = G[u][v]["weight"]

    return G, s, targets, budget
