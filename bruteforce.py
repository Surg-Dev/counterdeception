import networkx as nx
import random as rd
import sys
import copy

def random_graph(n, ts, ef=3):
    V = n
    E = int(ef*n)
    G = nx.gnm_random_graph(V,E)

    while (not nx.is_connected(G)):
        G = nx.gnm_random_graph(V,E)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = rd.randint(1,500)
    nx.set_node_attributes(G, 0, 'paths')

    targets = rd.sample(G.nodes, ts+1)
    s = targets[0]
    targets = targets[1:]

    return G, s, targets
V = 5
tcount = 2
G, s, targets = random_graph(V, tcount)
COST = nx.minimum_spanning_tree(G, weight='weight').size(weight='weight')*1.5

print ("start: ", s, "targets: ",targets)
best = None
bestc = sys.maxsize
bestm = 0
for t in nx.SpanningTreeIterator(G, weight='weight', minimum=True):
    # Get Predecessors
    pred = nx.dfs_predecessors(t, s)

    # Determine Paths Counts
    for v in targets:
        while v != s:
            t.nodes[v]['paths'] += 1
            v = pred[v]
    t.nodes[s]['paths'] = tcount

    # Remove Nodes with no paths
    remove = []
    for v in t.nodes():
        if t.nodes[v]['paths'] == 0:
            remove.append(v)
    
    for v in remove:
        t.remove_node(v)

    c = t.size(weight='weight')    
    if c>=COST:
        continue

    # Determine Counterdeception metric
    metric = sys.maxsize
    forced = False

    for v in targets:
        cur = v
        curdist = 0
        while cur != s and t.nodes[cur]['paths'] == 1:
            if t.nodes[cur]['paths'] == 1:
                curdist += t.edges[cur, pred[cur]]['weight']
            cur = pred[cur]
        if curdist == 0:
            forced = True
        else:
            metric = min(metric, curdist)
    
    if forced:
        metric = 0
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

    
    print(t, "weight: ", c,"/",COST, "metric: ", metric, "forced: ", forced)

print("Best Tree: ", best, "weight: ", bestc,"/",COST, "metric: ", bestm)
