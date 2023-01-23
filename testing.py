# idk what's going on in main.py
# We honestly need to refactor and comment alot of the code
# This is JUST for testing random stuff, expect nothing here to stay

from heuristic import random_points, form_grid_graph
from matplotlib import pyplot as plt
import networkx as nx

n = 5
s, targets = random_points(n)
G = form_grid_graph(s, targets, 3 * n, 1 * n)

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
nx.draw(G, pos=positions, node_color=colors, with_labels=False, node_size=200)
plt.show()
