import networkx as nx
import random as rd

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
V = 4
tcount = 2
G, s, targets = random_graph(V, tcount)
COST = nx.minimum_spanning_tree(G, weight='weight').size(weight='weight')*1.5

print (targets)
for t in nx.SpanningTreeIterator(G, weight='weight', minimum=True):
    rem = False
    # TODO: remove nodes that are not in the path to any target faster than this.
    while not rem:
        succ = nx.dfs_successors(t, s)
        print(succ)
        toRemove = []
        for v in t.nodes():
            if v not in succ:
                if v not in targets:
                    toRemove.append(v)
                    rem = True
        for v in toRemove:
            t.remove_node(v)
        if rem == True:
            rem = False
        else:
            rem = True

    c = t.size(weight='weight')
    if c>=COST:
        continue

    # TODO: Measure unique paths to targets.
    print(t, "weight: ", c,"/",COST)
