import networkx as nx
import random as rd
from math import sqrt
from matplotlib import pyplot as plt


def display_graph(G, loc=None):
    positions = dict()
    colors = []
    for node in G.nodes():
        if node == "start":
            colors.append("blue")

        elif "target" in node:
            colors.append("red")
        else:
            colors.append("green")

        positions[node] = G.nodes[node]["pos"]

    plt.figure(figsize=(15, 15))
    nx.draw(G, pos=positions, node_color=colors, with_labels=False, node_size=100)
    # if loc != None:
    #     filename = f"{loc}.png"
    #     plt.savefig(filename)
    #     plt.close()
    plt.show()


def display_tree(G, mst, loc=None):
    # Standardized function to display the mst over G
    colors = []
    for node in mst.nodes():
        if node == "start":
            colors.append("blue")

        elif "target" in node:
            colors.append("red")
        else:
            colors.append("green")
    plt.figure(figsize=(15, 15))
    positions = nx.get_node_attributes(G, "pos")
    nx.draw(mst, pos=positions, node_color=colors, with_labels=False, node_size=50)

    if loc != None:
        filename = f"{loc}.png"
        plt.savefig(filename)
        plt.close()
    # plt.show()


def random_points(n, x_max=100.0, y_max=100.0):
    # 1 start, n targets
    # x coordinates in [0, x_max]
    # y coordinates in [0, y_max]

    targets = [(rd.uniform(0, x_max), rd.uniform(0, y_max)) for _ in range(n)]

    # start should be somewhere around border of box formed by targets
    tx_min = min(x for (x, _) in targets)
    tx_max = max(x for (x, _) in targets)
    ty_min = min(y for (_, y) in targets)
    ty_max = max(y for (_, y) in targets)

    side = rd.randint(0, 3)
    if side == 0:  # left
        s = (tx_min, rd.uniform(ty_min, ty_max))
    if side == 1:  # top
        s = (rd.uniform(tx_min, tx_max), ty_max)
    if side == 2:  # right
        s = (tx_max, rd.uniform(ty_min, ty_max))
    if side == 3:  # bottom
        s = (rd.uniform(tx_min, tx_max), ty_min)

    return s, targets


def random_graph(n, ts, ef=3):
    V = n
    E = int(ef * n)
    G = nx.gnm_random_graph(V, E)

    while not nx.is_connected(G):
        G = nx.gnm_random_graph(V, E)

    for (u, v, w) in G.edges(data=True):
        w["weight"] = rd.randint(1, 500)
    nx.set_node_attributes(G, 0, "paths")

    targets = rd.sample(G.nodes, ts + 1)
    s = targets[0]
    targets = targets[1:]

    return G, s, targets


def triangulate_grid_graph(G):
    # If this isn't a grid graph then assume undefined behavior

    original_nodes = [(x, y) for (x, y) in G.nodes()]
    for x, y in original_nodes:
        if (x + 1, y) in G and (x, y + 1) in G:
            x_pos = (G.nodes[x, y]["pos"][0] + G.nodes[x + 1, y]["pos"][0]) / 2
            y_pos = (G.nodes[x, y]["pos"][1] + G.nodes[x, y + 1]["pos"][1]) / 2

            G.add_node((x + 0.5, y + 0.5), pos=(x_pos, y_pos))

            x_dist = G[x, y][x + 1, y]["weight"] / 2
            y_dist = G[x, y][x, y + 1]["weight"] / 2
            weight = pow(x_dist**2 + y_dist**2, 0.5)
            G.add_edge((x, y), (x + 0.5, y + 0.5), weight=weight)
            G.add_edge((x + 1, y), (x + 0.5, y + 0.5), weight=weight)
            G.add_edge((x, y + 1), (x + 0.5, y + 0.5), weight=weight)
            G.add_edge((x + 1, y + 1), (x + 0.5, y + 0.5), weight=weight)

    return G


def form_grid_graph(s, targets, x_gran, y_gran, triangulate=True):
    # TODO: Able to set x_weights and y_weights
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

    if triangulate:
        return triangulate_grid_graph(G)
    else:
        return G


def form_triangle_graph(s, targets, x_gran, y_gran, weight):
    G = nx.triangular_lattice_graph(x_gran, y_gran)

    for u, v in G.edges():
        G[u][v]["weight"] = weight

    # Compute rescaling for node positions
    nodes = targets + [s]
    x_min = min(x for (x, _) in nodes)
    x_max = max(x for (x, _) in nodes)
    y_min = min(y for (_, y) in nodes)
    y_max = max(y for (_, y) in nodes)

    Gx_min = min(G.nodes[x, y]["pos"][0] for (x, y) in G.nodes())
    Gx_max = max(G.nodes[x, y]["pos"][0] for (x, y) in G.nodes())
    Gy_min = min(G.nodes[x, y]["pos"][1] for (x, y) in G.nodes())
    Gy_max = max(G.nodes[x, y]["pos"][1] for (x, y) in G.nodes())

    x_scale = (x_max - x_min) / (Gx_max - Gx_min)
    y_scale = (y_max - y_min) / (Gy_max - Gy_min)

    positions = dict()
    for x, y in G.nodes():
        curr_x, curr_y = G.nodes[x, y]["pos"]
        new_x = (curr_x - Gx_min) / (Gx_max - Gx_min) * (x_max - x_min) + x_min
        new_y = (curr_y - Gy_min) / (Gy_max - Gy_min) * (y_max - y_min) + y_min
        positions[(x, y)] = (new_x, new_y)
    nx.set_node_attributes(G, positions, "pos")

    return G


def form_hex_graph(s, targets, x_gran, y_gran, weight):
    G = nx.hexagonal_lattice_graph(x_gran, y_gran)

    for u, v in G.edges():
        G[u][v]["weight"] = weight

    # Compute rescaling for node positions
    nodes = targets + [s]
    x_min = min(x for (x, _) in nodes)
    x_max = max(x for (x, _) in nodes)
    y_min = min(y for (_, y) in nodes)
    y_max = max(y for (_, y) in nodes)

    Gx_min = min(G.nodes[x, y]["pos"][0] for (x, y) in G.nodes())
    Gx_max = max(G.nodes[x, y]["pos"][0] for (x, y) in G.nodes())
    Gy_min = min(G.nodes[x, y]["pos"][1] for (x, y) in G.nodes())
    Gy_max = max(G.nodes[x, y]["pos"][1] for (x, y) in G.nodes())

    x_scale = (x_max - x_min) / (Gx_max - Gx_min)
    y_scale = (y_max - y_min) / (Gy_max - Gy_min)

    positions = dict()
    for x, y in G.nodes():
        curr_x, curr_y = G.nodes[x, y]["pos"]
        new_x = (curr_x - Gx_min) / (Gx_max - Gx_min) * (x_max - x_min) + x_min
        new_y = (curr_y - Gy_min) / (Gy_max - Gy_min) * (y_max - y_min) + y_min
        positions[(x, y)] = (new_x, new_y)
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
    for i, (x1, y1) in enumerate(targets):
        min_dist = float("inf")
        closest = None
        for node in G.nodes:
            if node != "start" and "target" not in node:
                x2, y2 = G.nodes[node]["pos"]
                if (dist := euclidian_dist(x1, y1, x2, y2)) < min_dist:
                    min_dist = dist
                    closest = node
        nx.relabel_nodes(G, {closest: f"target {i}"}, copy=False)

    return G
