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
    mst = nx.minimum_spanning_tree(G)

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


def reattachment(
    G, s, targets, budget, mst, forced, metric, target_list, pred, target_paths
):
    # Pick a target starting with the minimum contribution to the metric distance
    for _, v in target_list:
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

        # For each node on the remaining tree:
        for potential in mstprime.nodes():
            # Retrieve the pred shortest path
            path = dijpath[potential]
            # Check if the path crosses any nodes in the tree
            sb = False
            for x in path[:-1]:
                if x in mstprime.nodes():
                    sb = True
            if sb:
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
            # If the tree either removes forced paths or improves the metric w/o adding forced paths,
            # *and* the tree is under the budget, update the tree and corresponding values.
            if (forcedp == False and best_tree["forced"] == True) or (
                metricp > best_tree["metric"] and forcedp == best_tree["forced"]
            ):
                if mstcheck.size(weight="weight") < budget:
                    best_tree = {
                        "tree": mstcheck,
                        "forced": forcedp,
                        "metric": metricp,
                        "target_list": target_listp,
                        "pred": predcheck,
                    }
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
            )

    return mst, forced, metric, target_list, pred


def reattachment_approximation(
    G, s, targets, budget, mst, forced, metric, target_list, pred
):
    # Precompute Dijkstra's from each target to all other nodes in the graph
    target_paths = compute_SSSP(G, targets)

    old_metric = float("inf")
    # Continue until we find no local improvement
    while old_metric != metric:
        old_metric = metric
        mst, forced, metric, target_list, pred = reattachment(
            G, s, targets, budget, mst, forced, metric, target_list, pred, target_paths
        )
    return mst, pred


def compute_tree(G, s, targets, budget):
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
        G, s, targets, budget, mst, forced, metric, target_list, pred
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
