import networkx as nx
from matplotlib import pyplot as plt


def display_tree(G, mst):
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
    plt.show()
