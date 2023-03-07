import pickle
import time
import networkx as nx
from matplotlib import pyplot as plt

from util import (
    display_graph,
    form_grid_graph,
    form_hex_graph,
    form_triangle_graph,
    random_points,
    round_targets_to_graph,
    bcolors,
)

# # Initial Parameters
# target_count = 5
# graphx = 100
# graphy = 100
#
# s, targets = random_points(target_count)
# G = form_grid_graph(s, targets, graphx, graphy)
# # G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
# # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
# # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)
# round_targets_to_graph(G, s, targets)
#
# display_graph(G)
#
# targets = [f"target {i}" for i in range(target_count)]
# s = "start"
# nx.set_node_attributes(G, 0, "paths")
#
#
# def factory():
#     return G, s, targets
#
#
# improvement, before, after = benchmark_single(factory, float("inf"))
#
# if improvement:
#     print("Made Improvements")
#     if before == 0.0:
#         print("    Now Unforced")
#     else:
#         print(f"    Before: {before}")
#         print(f"    After:  {after}")
# else:
#     print("No Improvements")
#     if after == 0.0:
#         print("    Still Forced")

# G = round_targets_to_graph(G, s, targets)
# positions = dict()
# colors = []
# for node in G.nodes():
#     if node == "start":
#         colors.append("blue")
#
#     elif "target" in node:
#         colors.append("red")
#     else:
#         colors.append("green")
#
#     positions[node] = G.nodes[node]["pos"]
#
# plt.figure(figsize=(15, 15))
# nx.draw(G, pos=positions, node_color=colors, with_labels=False, node_size=100)
# plt.show()

print("stay")
i = 0
for j in range(100):
    # print(f"{bcolors.OKBLUE}Computing{'.'*(i)}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}{bcolors.LOADING[i]}{bcolors.ENDC}")
    time.sleep(0.25)
    print(bcolors.CLEAR_LAST_LINE)
    i = (i + 1) % 4
print("appear")
