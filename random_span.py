import networkx as nx
import random


def mst_from_pred(G, pred):
    mst = G.copy()
    mst.remove_edges_from(mst.edges())
    for v in pred.keys():
        u = pred[v]
        if u != None:
            mst.add_edge(u, v, weight=G.edges[u, v]["weight"])

    return mst


def random_tree_with_root(G, r):
    # Algorithm from https://dl.acm.org/doi/10.1145/237814.237880
    in_tree = {node: False for node in G.nodes()}
    pred = {node: None for node in G.nodes()}
    in_tree[r] = True

    for i in G.nodes():
        u = i
        while not in_tree[u]:
            pred[u] = random.choice(list(G[u]))
            u = pred[u]
        u = i
        while not in_tree[u]:
            in_tree[u] = True
            u = pred[u]

    return mst_from_pred(G, pred)
