import networkx as nx
import random as rd


def random_points(n, x_max=100.0, y_max=100.0):
    # 1 start, n targets
    # x coordinates in [0, x_max]
    # y coordinates in [0, y_max]

    coords = [(rd.uniform(0, x_max), rd.uniform(0, y_max)) for _ in range(n + 1)]
    s = coords[0]
    targets = coords[1:]

    return s, targets


def form_grid_graph(s, targets, gran):
    # gran := total number of vertical and horizontal lines
    #   Maybe specify x and y granularity in future?
    # Add diagonals in one direction if there is not a target
    # Otherwise connect target to nearest grid points

    nodes = targets + [s]
    x_min = min(x for (x, _) in nodes)
    x_max = max(x for (x, _) in nodes)
    y_min = min(y for (_, y) in nodes)
    y_max = max(y for (_, y) in nodes)

    x_gran = (x_max - x_min) / gran
    y_gran = (y_max - y_min) / gran

    G = nx.grid_2d_graph(gran, gran)

    positions = dict()
    # add distances
    for x, y in G.nodes():
        if x < gran - 1:
            G[x, y][x + 1, y]["weight"] = x_gran
        if y < gran - 1:
            G[x, y][x, y + 1]["weight"] = y_gran

        # set x, y position
        positions[(x, y)] = (x_min + x * x_gran, y_min + y * y_gran)

    nx.set_node_attributes(G, positions, "pos")
    return G
