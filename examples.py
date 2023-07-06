from util import (
    random_points,
    form_grid_graph,
    display_graph,
    display_tree,
    round_targets_to_graph,
)
import networkx as nx
from matplotlib import pyplot as plt

# constants for consistency
START_COLOR = "blue"
TARGET_COLOR = "red"
HIGHLIGHT_COLOR = "purple"
NODE_COLOR = "green"
HIGHLIGHT_SIZE = 500
NODE_SIZE = 350
EDGE_WEIGHT = 5


def shortest_path(loc=None):
    # example showing that agents do not always take shortest path

    # form graph
    s = (2, 0)
    targets = [(2, 3), (3, 4)]
    G = nx.grid_2d_graph(5, 5)
    positions = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, positions, "pos")
    nx.relabel_nodes(G, {s: "start"}, copy=False)
    for i, tar in enumerate(targets):
        nx.relabel_nodes(G, {tar: f"target {i}"}, copy=False)

    # specify edges to keep
    t = G.to_directed()
    black_edges = {
        ((2, 1), (2, 2)),
        ((2, 2), "target 0"),
        ("start", (2, 1)),
    }
    blue_edges = {
        ("start", (3, 0)),
        ((3, 0), (3, 1)),
        ((3, 1), (3, 2)),
        ((3, 2), (3, 3)),
        ((3, 3), "target 0"),
    }
    blue_dotted_edges = {
        ((3, 3), "target 1"),
    }
    keep = black_edges | blue_edges | blue_dotted_edges
    for edge in list(t.edges()):
        if edge not in keep:
            t.remove_edge(*edge)

    # label nodes
    labels = dict()
    for node in t.nodes():
        if "start" not in node and "target" not in node:
            labels[node] = ""
        else:
            labels[node] = node

    # set node attributes
    colors = []
    sizes = []
    for node in t.nodes():
        if node == "start":
            colors.append(START_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        elif "target" in node:
            colors.append(TARGET_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        else:
            colors.append(NODE_COLOR)
            sizes.append(NODE_SIZE)

    # set edge attributes
    weights = []
    edge_colors = []
    styles = []
    for e in t.edges():
        weights.append(EDGE_WEIGHT)
        if e in black_edges:
            edge_colors.append("black")
            styles.append("solid")
        elif e in blue_edges:
            edge_colors.append("blue")
            styles.append("solid")
        elif e in blue_dotted_edges:
            edge_colors.append("blue")
            styles.append("dashed")
        else:
            edge_colors.append("white")
            styles.append("solid")

    # start drawing
    plt.figure(figsize=(9.5, 9.5))
    positions = nx.get_node_attributes(G, "pos")
    nx.draw(
        t,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        width=weights,
        arrowsize=80,
        arrowstyle="->",
        edge_color=edge_colors,
        style=styles,
    )

    # move labels
    for node in positions:
        x, y = positions[node]
        if "target" in node:
            positions[node] = (x, y + 0.30)
        else:
            positions[node] = (x + 0.35, y + 0.30)

    # finish drawing
    nx.draw_networkx_labels(G, positions, labels, font_size=35)
    if loc != None:
        print(f"saving {loc}")
        plt.savefig(loc)
        plt.close()
    else:
        plt.show()


def last_deceptive_point(loc=None):
    # example demonsrating the concept of a last deceptive point

    # form graph
    s = (2, 0)
    targets = [(1, 3), (3, 3)]
    G = nx.grid_2d_graph(5, 5)
    positions = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, positions, "pos")
    nx.relabel_nodes(G, {s: "start"}, copy=False)
    for i, tar in enumerate(targets):
        nx.relabel_nodes(G, {tar: f"target {i}"}, copy=False)

    # specify edges to keep
    t = G.to_directed()
    blue_edges = {
        ("start", (2, 1)),
        ((2, 1), (2, 2)),
        ((2, 2), (2, 3)),
        ((2, 3), "target 0"),
    }
    blue_dotted_edges = {
        ((2, 3), "target 1"),
    }
    keep = blue_edges | blue_dotted_edges
    for edge in list(t.edges()):
        if edge not in keep:
            t.remove_edge(*edge)

    # label nodes
    labels = dict()
    for node in t.nodes():
        if "start" not in node and "target" not in node:
            labels[node] = ""
        else:
            labels[node] = node

    # set node attributes
    colors = []
    sizes = []
    for node in t.nodes():
        if node == "start":
            colors.append(START_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        elif "target" in node:
            colors.append(TARGET_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        elif node == (2, 3):
            colors.append(HIGHLIGHT_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        else:
            colors.append(NODE_COLOR)
            sizes.append(NODE_SIZE)

    # set edge attributes
    weights = []
    edge_colors = []
    styles = []
    for e in t.edges():
        weights.append(EDGE_WEIGHT)
        if e in blue_edges:
            edge_colors.append("blue")
            styles.append("solid")
        elif e in blue_dotted_edges:
            edge_colors.append("blue")
            styles.append("dashed")
        else:
            edge_colors.append("white")
            styles.append("solid")

    # start drawing
    plt.figure(figsize=(9.5, 9.5))
    positions = nx.get_node_attributes(G, "pos")
    nx.draw(
        t,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        width=weights,
        arrowsize=80,
        arrowstyle="->",
        edge_color=edge_colors,
        style=styles,
    )

    # move labels
    for node in positions:
        x, y = positions[node]
        if "target" in node:
            positions[node] = (x, y + 0.30)
        else:
            positions[node] = (x, y - 0.20)

    # finish drawing
    nx.draw_networkx_labels(G, positions, labels, font_size=35)
    if loc != None:
        print(f"saving {loc}")
        plt.savefig(loc)
        plt.close()
    else:
        plt.show()


def unique_distance(loc=None):
    # example demonsrating the concept of unique distance

    # form graph
    s = (2, 0)
    targets = [(0, 3), (2, 4), (4, 4)]
    G = nx.grid_2d_graph(5, 5)
    positions = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, positions, "pos")
    nx.relabel_nodes(G, {s: "start"}, copy=False)
    for i, tar in enumerate(targets):
        nx.relabel_nodes(G, {tar: f"target {i}"}, copy=False)

    # specify edges to keep
    t = G.to_directed()
    black_edges = {
        ((2, 3), "target 1"),
        ((2, 3), (1, 3)),
        ((1, 3), "target 0"),
        ((3, 2), (4, 2)),
        ((4, 2), (4, 3)),
        ((4, 3), "target 2"),
    }
    blue_edges = {
        ("start", (2, 1)),
        ((2, 1), (3, 1)),
        ((3, 1), (3, 2)),
        ((3, 2), (3, 3)),
        ((3, 3), (2, 3)),
    }
    keep = blue_edges | black_edges
    for edge in list(t.edges()):
        if edge not in keep:
            t.remove_edge(*edge)

    # label nodes
    labels = dict()
    for node in t.nodes():
        if "start" not in node and "target" not in node:
            labels[node] = ""
        else:
            labels[node] = node

    # set node attributes
    colors = []
    sizes = []
    for node in t.nodes():
        if node == "start":
            colors.append(START_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        elif "target" in node:
            colors.append(TARGET_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        else:
            colors.append(NODE_COLOR)
            sizes.append(NODE_SIZE)

    # set edge attributes
    weights = []
    edge_colors = []
    styles = []
    for e in t.edges():
        weights.append(5)
        if e in blue_edges:
            edge_colors.append("blue")
            styles.append("solid")
        elif e in black_edges:
            edge_colors.append("black")
            styles.append("solid")
        else:
            edge_colors.append("white")
            styles.append("solid")

    # start drawing
    plt.figure(figsize=(11, 11))
    positions = nx.get_node_attributes(G, "pos")
    nx.draw(
        t,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        width=weights,
        arrowsize=80,
        arrowstyle="->",
        edge_color=edge_colors,
        style=styles,
    )

    # move labels
    for node in positions:
        x, y = positions[node]
        if "target" in node:
            positions[node] = (x, y + 0.30)
        else:
            positions[node] = (x, y - 0.20)

    # finish drawing
    nx.draw_networkx_labels(G, positions, labels, font_size=35)
    if loc != None:
        print(f"saving {loc}")
        plt.savefig(loc)
        plt.close()
    else:
        plt.show()


def cycle(loc=None):
    # example demonsrating why cycles are bad for counterdeception

    # form graph
    s = (0, 0)
    targets = [(2, 4), (2, 2)]
    G = nx.grid_2d_graph(5, 5)
    positions = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, positions, "pos")
    nx.relabel_nodes(G, {s: "start"}, copy=False)
    for i, tar in enumerate(targets):
        nx.relabel_nodes(G, {tar: f"target {i}"}, copy=False)

    # specify edges to keep
    t = G.to_directed()
    keep = {
        ("start", (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (0, 4)),
        ((0, 4), (1, 4)),
        ((1, 4), "target 0"),
        ("target 0", (3, 4)),
        ((3, 4), (4, 4)),
        ((4, 4), (4, 3)),
        ((4, 3), (4, 2)),
        ((4, 2), (3, 2)),
        ((3, 2), "target 1"),
        ("target 1", (1, 2)),
        ((1, 2), (0, 2)),
    }
    for edge in list(t.edges()):
        if edge not in keep:
            t.remove_edge(*edge)

    # label nodes
    labels = dict()
    for node in t.nodes():
        if "start" not in node and "target" not in node:
            labels[node] = ""
        else:
            labels[node] = node

    # set node attributes
    colors = []
    sizes = []
    for node in t.nodes():
        if node == "start":
            colors.append(START_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        elif "target" in node:
            colors.append(TARGET_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        else:
            colors.append(NODE_COLOR)
            sizes.append(NODE_SIZE)

    # set edge attributes
    weights = []
    edge_colors = []
    styles = []
    for e in t.edges():
        weights.append(5)
        if e in keep:
            edge_colors.append("blue")
            styles.append("solid")
        else:
            edge_colors.append("white")
            styles.append("solid")

    # start drawing
    plt.figure(figsize=(9, 9))
    positions = nx.get_node_attributes(G, "pos")
    nx.draw(
        t,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        width=weights,
        arrowsize=80,
        arrowstyle="->",
        edge_color=edge_colors,
        style=styles,
    )

    # move labels
    for node in positions:
        x, y = positions[node]
        if "target" in node:
            positions[node] = (x, y + 0.30)
        else:
            positions[node] = (x, y - 0.20)

    # finish drawing
    nx.draw_networkx_labels(G, positions, labels, font_size=35)
    if loc != None:
        print(f"saving {loc}")
        plt.savefig(loc)
        plt.close()
    else:
        plt.show()


def dag(loc=None):
    # example demonsrating why dags can always be reduced to trees

    # form graph
    s = (1, 0)
    targets = [(2, 4)]
    G = nx.grid_2d_graph(5, 5)
    positions = {(x, y): (x, y) for x, y in G.nodes()}
    nx.set_node_attributes(G, positions, "pos")
    nx.relabel_nodes(G, {s: "start"}, copy=False)
    for i, tar in enumerate(targets):
        nx.relabel_nodes(G, {tar: f"target {i}"}, copy=False)

    # specify edges to keep
    t = G.to_directed()
    blue_edges = {
        ("start", (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
        ((1, 3), (1, 4)),
        ((1, 4), "target 0"),
    }
    dotted_edges = {
        ("start", (2, 0)),
        ((2, 0), (3, 0)),
        ((3, 0), (3, 1)),
        ((3, 1), (3, 2)),
        ((3, 2), (3, 3)),
        ((3, 3), (2, 3)),
        ((2, 3), "target 0"),
    }
    keep = blue_edges | dotted_edges
    for edge in list(t.edges()):
        if edge not in keep:
            t.remove_edge(*edge)

    # label nodes
    labels = dict()
    for node in t.nodes():
        if "start" not in node and "target" not in node:
            labels[node] = ""
        else:
            labels[node] = node

    # set node attributes
    colors = []
    sizes = []
    for node in t.nodes():
        if node == "start":
            colors.append(START_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        elif "target" in node:
            colors.append(TARGET_COLOR)
            sizes.append(HIGHLIGHT_SIZE)
        else:
            colors.append(NODE_COLOR)
            sizes.append(NODE_SIZE)

    # set edge attributes
    weights = []
    edge_colors = []
    styles = []
    for e in t.edges():
        weights.append(5)
        if e in blue_edges:
            edge_colors.append("blue")
            styles.append("solid")
        elif e in dotted_edges:
            edge_colors.append("blue")
            styles.append("dashed")
        else:
            edge_colors.append("white")
            styles.append("solid")

    # start drawing
    plt.figure(figsize=(9, 9))
    positions = nx.get_node_attributes(G, "pos")
    nx.draw(
        t,
        pos=positions,
        node_color=colors,
        node_size=sizes,
        width=weights,
        arrowsize=80,
        arrowstyle="->",
        edge_color=edge_colors,
        style=styles,
    )

    # move labels
    for node in positions:
        x, y = positions[node]
        if "target" in node:
            positions[node] = (x, y + 0.30)
        else:
            positions[node] = (x, y - 0.20)

    # finish drawing
    nx.draw_networkx_labels(G, positions, labels, font_size=35)
    if loc != None:
        print(f"saving {loc}")
        plt.savefig(loc)
        plt.close()
    else:
        plt.show()


def main():
    filename = "examples/shortest.png"
    shortest_path(loc=filename)

    filename = "examples/last_deceptive.png"
    last_deceptive_point(loc=filename)

    filename = "examples/unique_distance.png"
    unique_distance(loc=filename)

    filename = "examples/cycle.png"
    cycle(loc=filename)

    filename = "examples/dag.png"
    dag(loc=filename)


if __name__ == "__main__":
    main()
