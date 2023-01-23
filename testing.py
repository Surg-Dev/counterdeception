# idk what's going on in main.py
# We honestly need to refactor and comment alot of the code
# This is JUST for testing random stuff, expect nothing here to stay

from heuristic import random_points, form_grid_graph
from matplotlib import pyplot as plt
import networkx as nx

s, targets = random_points(5)
G = form_grid_graph(s, targets, 10)

positions = dict()
for x, y in G.nodes():
    positions[(x, y)] = G.nodes[x, y]["pos"]

plt.figure(figsize=(15, 15))
nx.draw(G, pos=positions, with_labels=True, node_size=800)
plt.show()
