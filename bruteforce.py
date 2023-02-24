import networkx as nx

from util import random_graph

V = 5
tcount = 2
G, s, targets = random_graph(V, tcount)
COST = nx.minimum_spanning_tree(G, weight="weight").size(weight="weight") * 1.5

print("start: ", s, "targets: ", targets)
best = None
bestc = float("inf")
bestm = 0
for t in nx.SpanningTreeIterator(G, weight="weight", minimum=True):
    # Get Predecessors
    pred = nx.dfs_predecessors(t, s)

    # Determine Paths Counts
    for v in targets:
        while v != s:
            t.nodes[v]["paths"] += 1
            v = pred[v]
    t.nodes[s]["paths"] = tcount

    # Remove Nodes with no paths
    remove = []
    for v in t.nodes():
        if t.nodes[v]["paths"] == 0:
            remove.append(v)

    for v in remove:
        t.remove_node(v)

    c = t.size(weight="weight")
    if c >= COST:
        continue

    # Determine Counterdeception metric
    metric = float("inf")
    forced = False

    for v in targets:
        cur = v
        curdist = 0
        while cur != s and t.nodes[cur]["paths"] == 1:
            if t.nodes[cur]["paths"] == 1:
                curdist += t.edges[cur, pred[cur]]["weight"]
            cur = pred[cur]
        if curdist == 0:
            forced = True
        else:
            metric = min(metric, curdist)

    if forced:
        metric = 0.0
    else:
        if metric > bestm:
            best = t
            bestc = c
            bestm = metric
        elif metric == bestm:
            if c < bestc:
                best = t
                bestc = c
                bestm = metric

    print(t, "weight: ", c, "/", COST, "metric: ", metric, "forced: ", forced)

print("Best Tree: ", best, "weight: ", bestc, "/", COST, "metric: ", bestm)
