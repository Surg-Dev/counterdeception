from util import (
    random_points,
    form_grid_graph,
    form_triangle_graph,
    round_targets_to_graph,
    form_hex_graph,
)
from matplotlib import pyplot as plt
import networkx as nx

n = 5
s, targets = random_points(n)

G = form_triangle_graph(s, targets, n**2, n**2, 0.5)
# G = form_hex_graph(s, targets, n**2, n**2, 0.5)
# G = form_grid_graph(s, targets, n**2, n**2, triangulate=True)

G = round_targets_to_graph(G, s, targets)
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
plt.show()
