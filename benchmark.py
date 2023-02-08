import networkx as nx
from matplotlib import pyplot as plt
from algo import compute_tree, compute_metric, build_stiener_seed
from util import build_graph


def benchmark_single(G, s, targets, budget):
    # compute a single improvement

    # get before
    mst, pred = build_stiener_seed(G, s, targets)
    forced, metric, target_list = compute_metric(mst, s, targets, pred)
    before = metric if not forced else 0.0

    # get after
    mst, pred = compute_tree(G, s, targets, budget)
    forced, metric, target_list = compute_metric(mst, s, targets, pred)
    after = metric if not forced else 0.0

    improvement = True if after > before else False
    return improvement, before, after


def benchmark_many(n, count, graphx, graphy):
    both_forced = 0
    now_unforced = 0
    unimproved = 0
    improved = 0

    for _ in range(n):

        # Set up graph, seed tree, and metric values.
        G, s, targets = build_graph(count, graphx, graphy)
        budget = build_stiener_seed(G, s, targets)[0].size(weight="weight") * 5
        improvement, before, after = benchmark_single(G, s, targets, budget)

        if improvement:
            improved += 1
            if before == 0.0:
                now_unforced += 1
        else:
            unimproved += 1
            if after == 0.0:
                both_forced += 1

    print(f"Number of graphs = {n}")
    print(f"{both_forced     = }")
    print(f"{now_unforced    = }")
    print(f"{unimproved      = }")
    print(f"{improved        = }")


def main():
    # Initial Parameters
    count = 10
    graphx = 50
    graphy = 50
    benchmark_many(5, count, graphx, graphy)


if __name__ == "__main__":
    main()
