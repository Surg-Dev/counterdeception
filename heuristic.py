import networkx as nx
import random as rd


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

    for i, t in enumerate(targets):
        G.add_node(f"target {i}", pos=t)
    G.add_node("start", pos=s)

    return G
