from util import display_tree
import networkx as nx


# This precomputes SSSP for each target, however we can subsitute with any "heuristic"
# The heuristic must be a function that ouputs a dictionary of dictionaries of paths that cover G.
def compute_SSSP(G, targets):
    # Compute the SSSP for each target using Dijkstra's
    target_paths = dict()
    for target in targets:
        target_paths[target] = nx.single_source_dijkstra_path(G, target)
    return target_paths


def compute_Astar(G, tree, s, t):
    nx.set_edge_attributes(G, 0.0, "a_star")

    # Compute the A* heuristic for each edge
    # print(s, t)
    # print(tree.nodes())
    # print("HERE")
    # print(G.nodes())
    # print("Init dists")
    for u, v in G.edges():
        # print (u, v)
        # Compute midpoint between u,v
        mid = (
            (G.nodes[u]["pos"][0] + G.nodes[v]["pos"][0]) / 2,
            (G.nodes[u]["pos"][1] + G.nodes[v]["pos"][1]) / 2,
        )

        # Compute distance from midpoint to target
        dist = (
            (mid[0] - G.nodes[t]["pos"][0]) ** 2 + (mid[1] - G.nodes[t]["pos"][1]) ** 2
        ) ** 0.5

        # Set the heuristic to the distance
        G[u][v]["a_star"] = dist

    filtered = [x for x in G.nodes() if (x not in tree.nodes())]
    filtered.append(s)
    filtered.append(t)

    H = G.subgraph(filtered)

    # print("Done dists")
    # neighbors = list(G.neighbors(s))
    # for neighbor in neighbors:
    #     G[s][neighbor]['a_star'] = 0

    # neighbors = list(G.neighbors(t))
    # for neighbor in neighbors:
    #     G[t][neighbor]['a_star'] = 0

    # for n in tree.nodes():
    #     # Get all neighbors of n
    #     neighbors = list(G.neighbors(n))
    #     # Set edge weights to infinity
    #     for neighbor in neighbors:
    #         G[n][neighbor]['a_star'] = float('inf')

    # Make sure s and t are traversable

    # Run Dijkstra's
    try:
        path = nx.shortest_path(H, s, t, weight="a_star")
        length = nx.shortest_path_length(H, s, t, weight="a_star")
    except nx.NetworkXNoPath:
        path = []
        length = float("inf")
    # print("Done Dijkstra's")

    return path, length


def mark_paths(tree, s, targets):
    pred = nx.dfs_predecessors(tree, s)
    nx.set_node_attributes(tree, 0, "paths")

    # Mark paths from targets towards the source.
    for v in targets:
        while v != s:
            tree.nodes[v]["paths"] += 1
            v = pred[v]
    tree.nodes[s]["paths"] = len(targets)

    # Pass predecessor information back to the caller.
    return pred


# Builds a seed MST and trims it to remove nodes with no paths to targets.
# O(V + E) time.
def build_stiener_seed(G, s, targets):
    # Build the seed MST and trim it.
    # mst = nx.minimum_spanning_tree(G)

    # Compute the maximum spanning tree
    mst = nx.maximum_spanning_tree(G)

    # Mark paths from targets towards the source.
    pred = mark_paths(mst, s, targets)

    # Remove nodes past targets with no further targets.
    remove = []
    for v in mst.nodes():
        if mst.nodes[v]["paths"] == 0:
            remove.append(v)

    for v in remove:
        mst.remove_node(v)

    return mst, pred


def compute_metric(mst, s, targets, pred=None):
    # Determine Counterdeception metric, mark if the tree has a forced path.
    # TODO: Define forced path

    if pred is None:
        pred = nx.dfs_predecessors(mst, s)

    forced = False
    # Placeholder value, metric should never match this.
    metric = mst.size(weight="weight")
    target_metrics = []

    # For each target, add up the distance between nodes
    # Repeat until we reach the source or a node with multiple paths
    for v in targets:
        cur = v
        curdist = 0
        while cur != s and mst.nodes[cur]["paths"] == 1:
            curdist += mst.edges[cur, pred[cur]]["weight"]
            cur = pred[cur]
        if curdist == 0:
            forced = True
            continue
        cur = pred[v]
        while cur != s:
            if cur in targets:
                curdist = -curdist
                break
            cur = pred[cur]
        target_metrics.append((curdist, v))

    # Sort the target metrics by distance, ascending, and pick the first one.
    target_metrics = sorted(target_metrics)
    metric = target_metrics[0][0]

    return forced, metric, target_metrics


# TODO: Run a pathfinding algorithm to find a greedy path to reattach to if it runs into a blocked path via the heuristic.


def reattachment(
    G, s, targets, budget, mst, forced, metric, target_list, pred, target_paths
):
    # Pick a target starting with the minimum contribution to the metric distance
    for c, t in enumerate(target_list):
        orig_metric_v, v = t
        print(orig_metric_v)
        # Make a copy of the MST to remove the target and corresponding path from.
        mstprime = mst.copy()
        updated = False

        # Remove the target and its path from the tree.
        cur = v
        while cur != s and mstprime.nodes[cur]["paths"] == 1:
            mstprime.remove_node(cur)
            cur = pred[cur]

        # Get the path for the target from the precomputed SSSP
        dijpath = target_paths[v]

        best_tree = {
            "tree": mst,
            "forced": forced,
            "metric": metric,
            "target_list": target_list,
            "pred": pred,
        }
        print("beginning reattachment")
        count = 0

        # For each node on the remaining tree:
        for potential in mstprime.nodes():
            # Skip reattaching to a target.
            if potential in targets:
                continue
            # Retrieve the pred shortest path
            path = dijpath[potential]
            dist_path = -1
            # Check if the path crosses any nodes in the tree
            for x in path[:-1]:
                if x in mstprime.nodes():
                    # Run A* from target to potential to find a path that doesn't cross the tree
                    path, dist_path = compute_Astar(G, mstprime, v, potential)
                    break

            if dist_path == float("inf"):
                continue

            # Make a new tree to reattach the target to.
            mstcheck = mstprime.copy()
            # Add nodes and edges to the tree.
            for i in range(len(path) - 1):
                mstcheck.add_node(path[i], paths=1, pos=G.nodes[path[i]]["pos"])
                if path[i + 1] not in mstcheck.nodes():
                    mstcheck.add_node(path[i + 1], paths=1, pos=G.nodes[path[i]]["pos"])
                mstcheck.add_edge(
                    path[i],
                    path[i + 1],
                    weight=G.edges[path[i], path[i + 1]]["weight"],
                )
            # Compute the new predecessor list and metric on the tree.
            predcheck = mark_paths(mstcheck, s, targets)
            forcedp, metricp, target_listp = compute_metric(
                mstcheck, s, targets, predcheck
            )

            # If the metric is negative, we are reattaching to a forced branch. Skip this reattachment.
            trynext = False
            new_metric_v = 0
            for m, y in target_listp:
                if y == v:
                    if m < 0:
                        trynext = True
                        break
                    new_metric_v = m
                    break
            if trynext:
                continue

            # heurmetric = target_listp[c][0]

            # If the tree either removes forced paths or improves the metric w/o adding forced paths,
            # *and* the tree is under the budget, update the tree and corresponding values.
            # print(target_listp)
            # print ("Old summed metric: ", sum(i for i,j in target_list))
            # print ("New summed metric: ", sum(i for i,j in target_listp))

            # TODO: Test if instead of taking the metric improvment for the specific target as the third condition
            # Try taking the difference of each target's metric as a sum and see if it's positive (net gain across all targets)
            if (
                (forcedp == False and best_tree["forced"] == True)
                or (metricp > best_tree["metric"] and forcedp == best_tree["forced"])
                or (orig_metric_v < 0 and new_metric_v > 0)
                or (
                    forcedp == False
                    and best_tree["forced"] == False
                    and new_metric_v > orig_metric_v
                )
            ):
                # or (forcedp == False and best_tree["forced"] == False and heurmetric > best_tree["target_list"][c][0] and metricp >= best_tree["metric"]):
                # or (sum(abs(i) for i,j in target_listp) > sum(abs(i) for i,j in best_tree["target_list"])):
                # or (forcedp == False and best_tree["forced"] == False and sum(i for i,j in target_listp) > sum(i for i,j in best_tree["target_list"]))
                if mstcheck.size(weight="weight") < budget:
                    best_tree = {
                        "tree": mstcheck,
                        "forced": forcedp,
                        "metric": metricp,
                        "target_list": target_listp,
                        "pred": predcheck,
                    }
                    count += 1
                    orig_metric_v = new_metric_v
                    updated = True

        print("reattached ", count, " times")
        # Don't try to reattach any other targets if we updated the tree.
        # The same or earlier targets may be reattached multiple times.
        # Note that if we change "reattaching the minimum target", this condition may need to change
        if best_tree["tree"] != mst:
            # print("improved!")
            return (
                best_tree["tree"],
                best_tree["forced"],
                best_tree["metric"],
                best_tree["target_list"],
                best_tree["pred"],
                updated,
            )

    return mst, forced, metric, target_list, pred, updated


def reattachment_approximation(
    G, s, targets, budget, mst, forced, metric, target_list, pred, loc=None
):
    # print before
    if loc != None:
        curr_loc = f"{loc}/{0}"
        display_tree(G, mst, loc=curr_loc)

    # Precompute Dijkstra's from each target to all other nodes in the graph
    target_paths = compute_SSSP(G, targets)

    old_metric = float("inf")
    updated = True

    count = 1
    mult = 1  # control how often we save an image
    # Continue until we find no local improvement
    while updated:
        if count % mult == 0:
            curr_loc = f"{loc}/{count}" if loc != None else None
            display_tree(G, mst, loc=curr_loc)
        count += 1
        # print(f"{forced = }")

        old_metric = metric
        mst, forced, metric, target_list, pred, updated = reattachment(
            G, s, targets, budget, mst, forced, metric, target_list, pred, target_paths
        )
        print("update!")

    # print after
    if loc != None:
        curr_loc = f"{loc}/{count+1}"
        display_tree(G, mst, loc=curr_loc)

    return mst, pred


def compute_tree(G, s, targets, budget, loc=None):
    # Initialize A* property on every edge in graph
    nx.set_edge_attributes(G, 0.0, "a_star")

    # Build the seed MST and trim it.
    mst, pred = build_stiener_seed(G, s, targets)

    # Get original characteristics of the tree
    forced, metric, target_list = compute_metric(mst, s, targets, pred)
    mstbench = mst.copy()
    originalsize = mst.size(weight="weight")
    metricbench = metric

    # print("FORCED: ", forced)
    # display_tree(G, mst)

    mst, pred = reattachment_approximation(
        G, s, targets, budget, mst, forced, metric, target_list, pred, loc=loc
    )

    # print(f"budget: {budget}")
    # print(
    #     f"original mst: {mstbench} original metric: {metricbench} original size: {originalsize}, forced: {forced}"
    # )
    # print(
    #     f"final tree: {mst} final metric: {metric} final size: {mst.size(weight='weight')} forced: {forced}"
    # )

    # display_tree(G, mst)

    return mst, pred
