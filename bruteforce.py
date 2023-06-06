import networkx as nx
from networkx.algorithms.tree.mst import SpanningTreeIterator
from algo import compute_metric, mark_paths
import pickle

from util import random_graph, display_tree, display_graph, random_points, form_grid_graph, round_targets_to_graph, bcolors

def bruteforce(G, s, targets, budget):
    best = float('-inf')
    best_tree = None
    count = 0
    spacer = 100
    for t in SpanningTreeIterator(G):
        if count % spacer == 0:
            print(f"Num Trees: {bcolors.OKBLUE}{count}{bcolors.ENDC}")
            print(f"Metric: {bcolors.OKBLUE}{best}{bcolors.ENDC}")
        count += 1

        # trim tree
        # Mark paths from targets towards the source.
        pred = mark_paths(t, s, targets)

        # Remove nodes past targets with no further targets.
        remove = []
        for v in t.nodes():
            if t.nodes[v]["paths"] == 0:
                remove.append(v)

        for v in remove:
            t.remove_node(v)

        # janky early return
        if t.size(weight="weight") > budget:
            print(bcolors.CLEAR_LAST_LINE)
            return best_tree

        # compute metric
        forced, metric, target_metrics = compute_metric(t, s, targets, pred=pred)

        metric = metric if not forced else 0.0
        if metric > best:
            best = metric
            best_tree = t

        if count % spacer == spacer - 1:
            print(f"{bcolors.CLEAR_LAST_LINE}")
            print(f"{bcolors.CLEAR_LAST_LINE}")

    # if we don't early return, we've explored everything
    return best_tree

def main():
    # Initial Parameters
    target_count = 2
    graphx = 2
    graphy = 2

    def factory():
        s, targets = random_points(target_count)

        # G = form_grid_graph(s, targets, graphx, graphy)
        G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
        # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
        # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

        round_targets_to_graph(G, s, targets)
        targets = [f"target {i}" for i in range(target_count)]
        s = "start"
        nx.set_node_attributes(G, 0, "paths")

        budget = float("inf")
        # budget = nx.minimum_spanning_tree(G).size(weight="weight") * 0.5

        # rescale weights
        for u, v in G.edges:
            G[u][v]["weight"] = G[u][v]["weight"]

        return G, s, targets, budget

    prefix = "results/brute/"
    for i in range(3):
        G, s, targets, budget = factory()
        best_tree = bruteforce(G, s, targets, budget)

        pickle.dump(G, open(f"{prefix}G_{i + 1}.pickle", 'wb'))
        pickle.dump(best_tree, open(f"{prefix}best_tree_{i + 1}.pickle", 'wb'))
        print()

if __name__ == "__main__":
    main()
