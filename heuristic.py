import networkx as nx
import random as rd
from math import sqrt


def random_points(n, x_max=100.0, y_max=100.0):
    # 1 start, n targets
    # x coordinates in [0, x_max]
    # y coordinates in [0, y_max]

    # WLOG, s = (0, 0)
    s = (0, 0)

    targets = [(rd.uniform(0, x_max), rd.uniform(0, y_max)) for _ in range(n)]

    return s, targets


def form_grid_graph(s, targets, x_gran, y_gran):
    # gran := total number of vertical and horizontal lines
    #   Maybe specify x and y granularity in future?
    # Add diagonals in one direction if there is not a target
    # Otherwise connect target to nearest grid points

    nodes = targets + [s]
    x_min = min(x for (x, _) in nodes)
    x_max = max(x for (x, _) in nodes)
    y_min = min(y for (_, y) in nodes)
    y_max = max(y for (_, y) in nodes)

    x_dist = (x_max - x_min) / x_gran
    y_dist = (y_max - y_min) / y_gran

    G = nx.grid_2d_graph(x_gran + 1, y_gran + 1)

    # add distances and set positions of non-start / target nodes
    positions = dict()
    for x, y in G.nodes():
        if x < x_gran:
            G[x, y][x + 1, y]["weight"] = x_dist
        if y < y_gran:
            G[x, y][x, y + 1]["weight"] = y_dist

        # set x, y position
        positions[(x, y)] = (x_min + x * x_dist, y_min + y * y_dist)
    nx.set_node_attributes(G, positions, "pos")

    return G


def round_targets_to_graph(G, s, targets):
    # rounds s and target to nearest nodes on graph according to Euclidian dist
    def euclidian_dist(x1, y1, x2, y2):
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # add start node
    min_dist = float("inf")
    closest = None
    for node in G.nodes:
        x2, y2 = G.nodes[node]["pos"]
        if (dist := euclidian_dist(*s, x2, y2)) < min_dist:
            min_dist = dist
            closest = node
    nx.relabel_nodes(G, {closest: "start"}, copy=False)

    # add targets
    for i, (x1, y2) in enumerate(targets):
        min_dist = float("inf")
        closest = None
        for node in G.nodes:
            if node != "start" and "target" not in node:
                x2, y2 = G.nodes[node]["pos"]
                if (dist := euclidian_dist(x1, x2, x2, y2)) < min_dist:
                    min_dist = dist
                    closest = node
        nx.relabel_nodes(G, {closest: f"target {i}"}, copy=False)

    return G
