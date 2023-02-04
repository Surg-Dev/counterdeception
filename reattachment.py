import heuristic
import networkx as nx
import sys
from matplotlib import pyplot as plt

# Build the graph and targets
def build_graph(target_count, gridx, gridy, x_max=100, y_max=100):
    s, targets = heuristic.random_points(target_count)
    G = heuristic.form_grid_graph(s, targets, 10, 10)
    heuristic.round_targets_to_graph(G, s, targets)
    targets = [f'target {i}' for i in range(target_count)]
    s = "start"
    nx.set_node_attributes(G, 0, "paths")
    return G, s, targets

def computeSSSP(G, targets):
    target_paths = dict()
    # Compute the SSSP for each target
    for target in targets:
        target_paths[target] = nx.single_source_dijkstra(G, target)
    return target_paths

def buildStienerSeed(G, s ,targets):
    # Build the seed MST and trim it.
    mst = nx.minimum_spanning_tree(G)
    pred = nx.dfs_predecessors(mst, s)

    # Mark paths from targets towards the source.
    for v in targets:
        while v != s:
            mst.nodes[v]["paths"] += 1
            v = pred[v]
    mst.nodes[s]["paths"] = len(targets)

    # Remove nodes past targets with no further targets.
    remove = []
    for v in mst.nodes():
        if mst.nodes[v]["paths"] == 0:
            remove.append(v)

    for v in remove:
        mst.remove_node(v)
    
    return mst

def computeMetric(mst, s, targets):
    pred = nx.dfs_predecessors(mst, s)
    # Determine Counterdeception metric, mark if the tree has a forced path.
    forced = False
    metric = mst.size(weight="weight")
    # min_target = None
    target_metrics = []


    for v in targets:
        cur = v
        curdist = 0
        while cur != s and mst.nodes[cur]["paths"] == 1:
            # print(cur)
            # print("see node:", mst.nodes[cur])
            if mst.nodes[cur]["paths"] == 1:
                curdist += mst.edges[cur, pred[cur]]["weight"]
            cur = pred[cur]
        if curdist == 0:
            forced = True
        else:
            target_metrics.append((curdist, v))
            if  metric < curdist:
                metric = curdist

    
    return forced, metric, target_metrics

#Initial Parameters
count = 3
graphx = 10
graphy = 10

# Set up graph, seed tree, and metric values.
G, s, targets = build_graph(count, graphx, graphy)
target_paths = computeSSSP(G, targets)
mst = buildStienerSeed(G, s, targets)
forced, metric, target_list = computeMetric(mst, s, targets)
budget = mst.size(weight="weight") * 2
print("FORCED: ", forced)


# print(nx.single_source_dijkstra(G, s))

# Find the path of the minimum metric, and reattach the target somewhere else.

old_metric = metric+1
curcost = mst.size(weight="weight")
print(mst.nodes())
while old_metric != metric or forced:
    target_list = sorted(target_list)
    # Pick the minimum target
    for dist, v in target_list:
        pred = nx.dfs_predecessors(mst, s)

        mstprime = mst.copy()
        updated = False

        # Remove it from the tree
        toremove = []
        cur = v
        # Remove the path supporting the target from the tree
        while cur != s and mst.nodes[cur]["paths"] == 1:
            toremove.append(cur)
            cur = pred[cur]
        for node in toremove:
            mstprime.remove_node(node)
        
        # Compute Dijkstras
        dists, dijpath = nx.single_source_dijkstra(G, v)
        # print(mstprime.nodes())
        # print("trying target", v)
        for potential in mstprime.nodes():
            print("trying node", potential)
            path = dijpath[potential]
            sb = False
            # Check if the path crosses any nodes in the tree
            for x in range(len(path)-1):
                if x in mst.nodes():
                    sb = True
            if sb:
                continue
            mstcheck = mstprime.copy()
            # Add node and path to the tree
            for i in range(len(path)-1):
                mstcheck.add_node(path[i], paths=1, pos=G.nodes[path[i]]["pos"])
                if path[i+1] not in mstcheck.nodes():
                    mstcheck.add_node(path[i+1], paths=1, pos=G.nodes[path[i]]["pos"])
                mstcheck.add_edge(path[i], path[i+1], weight=G.edges[path[i], path[i+1]]["weight"])
            print("here")
            forcedp, metricp, target_listp = computeMetric(mstcheck, s, targets)
            if forcedp == False and forced == True or metricp > metric:
                if (mstcheck.size(weight="weight") < budget):
                    print("update!")
                    print("old metric ", metric, "new metric ", metricp)
                    print(mstcheck.nodes())
                    mst = mstcheck
                    forced = forcedp
                    old_metric = metric
                    metric = metricp
                    target_list = target_listp
                    updated = True
                    break
        if updated:
            break

print(mst.nodes())
            
            


    





# For each node currently on the tree:
    # Path to it from target to tree via dijkstra's
    # Make sure it doesn't cross any current nodes on the tree
    # Add the nodes back to the tree
    # Recompute Metric to see if it improves

# If it does, repeat from start

# If it doesn't, try the next target
