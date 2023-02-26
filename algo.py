from math import isclose

import networkx as nx

from util import display_tree


def brute_force(G, s, targets, budget, loc=None):
    best_tree = None
    best_m = float("inf")
    count = 0
    for t in nx.SpanningTreeIterator(G, weight="weight", minimum=True):
        if count % 1000 == 0:
            print(f"    {count} trees processed")
            curr_size = t.size(weight="weight")
            print(f"        {curr_size = }")
            print(f"        {budget = }")
        # Get Predecessors
        pred = nx.dfs_predecessors(t, s)

        # Determine Paths Counts
        for v in targets:
            while v != s:
                t.nodes[v]["paths"] += 1
                v = pred[v]
        t.nodes[s]["paths"] = len(targets)

        # Remove Nodes with no paths
        remove = []
        for v in t.nodes():
            if t.nodes[v]["paths"] == 0:
                remove.append(v)

        for v in remove:
            t.remove_node(v)

        c = t.size(weight="weight")
        if c > budget:
            # iterator goes in increasing size order
            # So once we hit past budget, there is no point
            curr_loc = f"{loc}/brute_force" if loc != None else None
            display_tree(G, mst, loc=curr_loc)
            return best_tree, metric

        # Determine Counterdeception metric
        forced, metric, target_list = compute_metric(t, s, targets)
        if metric < best_m:
            best_tree = t
            best_m = metric

        count += 1

    # Also return here as a fail safe (for example for infinite budget)
    curr_loc = f"{loc}/brute_force" if loc != None else None
    display_tree(G, mst, loc=curr_loc)
    return best_tree, metric


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


# Builds a seed tree and trims it to remove nodes with no paths to targets.
# O(V + E) time.
def build_stiener_seed(G, s, targets, minimum=True):
    # Build the seed MST and trim it.

    if minimum:
        mst = nx.minimum_spanning_tree(G)
    else:
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
    def is_better_tuple(old, new):
        # (forced, min_metric, sum_metric, cost, potential_vert)
        #
        # forced = 0 if not forced, 1 if forced
        # min_metric = the change of minimum metric from the initial tree on this reattachment cycle for this target
        # sum_metric = the sum of metric for each target.
        #   TODO: Maybe something other than sum?
        # cost = the cost of the tree (mainly used for a tie breaker for the previous values)
        # potential_vert = vertex we are reattaching too (used for reference)
        #
        # Returns true iff new is better than old in some measure

        # 0 := not forced, 1 := forced
        if old[0] == 0 and new[0] == 1:
            return False
        if old[0] == 1 and new[0] == 0:
            return True
        # If we get here, forcing hasn't changed

        # See if minimum metric improved
        if old[1] > new[1]:
            return False
        if old[1] < new[1]:
            return True
        # If we get here, minimum metric is same

        # See if sum of metrics improved
        if old[2] > new[2]:
            return False
        if old[2] < new[2]:
            return True
        # If we get here, sum of metrics is same

        # See if cost improved
        #   Floats are weird so we first check if they are close
        if not isclose(new[3], old[3]):
            if old[3] < new[3]:
                return False
            if old[3] > new[3]:
                return True
        # If we get here, cost is same

        # if no improvement, retain old
        return False

    best_tree = {
        "tree": mst,
        "forced": forced,
        "metric": metric,
        "target_list": target_list,
        "pred": pred,
    }
    best_tuple = (
        1 if forced else 0,
        metric,
        sum(met for (met, v) in target_list),
        mst.size(weight="weight"),
        None,
    )
    updated = False
    start_tuple = best_tuple

    # Pick a target starting with the minimum contribution to the metric distance
    for c, t in enumerate(target_list):
        orig_metric_v, v = t
        print(f"trying to reattach {v} with metric {orig_metric_v}")
        # Make a copy of the MST to remove the target and corresponding path from.
        mstprime = mst.copy()

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
        count = 0

        best_seen_metric = best_tree["metric"]
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

            # form new tuple

            curr_tuple = (
                1 if forcedp else 0,
                metricp,
                sum(met for (met, v) in target_listp),
                mstcheck.size(weight="weight"),
                None,
            )

            # TODO: Test if instead of taking the metric improvment for the specific target as the third condition
            # Try taking the difference of each target's metric as a sum and see if it's positive (net gain across all targets)
            # if (
            #     ((
            #         forcedp == False and best_tree["forced"] == True
            #     )  # Tree is no longer forced
            #     or (
            #         metricp > best_tree["metric"] and forcedp == best_tree["forced"]
            #     )  # Improved metric, may or may not be forced still
            #     or (orig_metric_v < 0 and new_metric_v > 0)
            #     or (
            #         forcedp == False
            #         and best_tree["forced"] == False
            #         and new_metric_v > orig_metric_v
            #     ))
            #     and mstcheck.size(weight="weight") < budget
            # ):
            if is_better_tuple(best_tuple, curr_tuple):
                # or (forcedp == False and best_tree["forced"] == False and heurmetric > best_tree["target_list"][c][0] and metricp >= best_tree["metric"]):
                # or (sum(abs(i) for i,j in target_listp) > sum(abs(i) for i,j in best_tree["target_list"])):
                # or (forcedp == False and best_tree["forced"] == False and sum(i for i,j in target_listp) > sum(i for i,j in best_tree["target_list"]))
                # we only want to take improvements
                # if not best_tree["metric"] > metricp:
                #     print((forcedp == False and best_tree["forced"] == True))
                #     print((metricp > best_tree["metric"] and forcedp == best_tree["forced"]))
                #     print("\t", forcedp)
                #     print((orig_metric_v < 0 and new_metric_v > 0))
                #     print(forcedp == False and best_tree["forced"] == False and new_metric_v > orig_metric_v)
                #     assert False

                # Updating since we saw something better
                best_seen_metric = max(best_seen_metric, metricp)
                best_tree = {
                    "tree": mstcheck,
                    "forced": forcedp,
                    "metric": metricp,
                    "target_list": target_listp,
                    "pred": predcheck,
                }
                updated = True
                best_tuple = curr_tuple
                # Don't understand what this is doing
                orig_metric_v = new_metric_v
                count += 1

        if not updated:
            print(f"    Made no reattachments for {v}")
        else:
            # Don't try to reattach any other targets if we updated the tree.
            # The same or earlier targets may be reattached multiple times.
            # Note that if we change "reattaching the minimum target", this condition may need to change

            print(f"    reattached {count} times")
            if start_tuple[0] == 1 and best_tuple[0] == 0:
                print("    Unforced tree")
            if start_tuple[1] < best_tuple[1]:
                print("    Increased minimum metric")
            if start_tuple[2] < best_tuple[2]:
                print("    Increased sum of metrics")
            if (
                not isclose(start_tuple[3], best_tuple[3])
                and start_tuple[3] > best_tuple[3]
            ):
                print("    Found cheaper tree")
                print(f"    starting cost = {start_tuple[3]}")
                print(f"    ending cost   = {best_tuple[3]}")

            if best_tree["metric"] != best_seen_metric:
                print("        !!  saw better metric, didn't take it  !!")
                # TODO: Figure out when and why this happens and if it's what we expect
                # assert False

            return (
                best_tree["tree"],
                best_tree["forced"],
                best_tree["metric"],
                best_tree["target_list"],
                best_tree["pred"],
                updated,
            )

    print("Made no updates")
    assert not updated
    return mst, forced, metric, target_list, pred, updated


def reattachment_approximation(
    G, s, targets, budget, mst, forced, metric, target_list, pred, loc=None
):
    # Precompute Dijkstra's from each target to all other nodes in the graph
    target_paths = compute_SSSP(G, targets)

    old_metric = float("inf")
    updated = True

    count = 0
    mult = 1  # control how often we save an image
    # Continue until we find no local improvement
    while updated:
        if count % mult == 0:
            curr_loc = f"{loc}/{count}" if loc != None else None
            display_tree(G, mst, loc=curr_loc)
        print()
        count += 1
        print(f"round {count}")

        old_metric = metric
        mst, forced, metric, target_list, pred, updated = reattachment(
            G, s, targets, budget, mst, forced, metric, target_list, pred, target_paths
        )

    # # print after
    # if loc != None:
    #     curr_loc = f"{loc}/{count}"
    #     display_tree(G, mst, loc=curr_loc)

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
