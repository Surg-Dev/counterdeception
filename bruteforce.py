import networkx as nx
from networkx.algorithms.tree.mst import SpanningTreeIterator
from algo import compute_metric, mark_paths, compute_tree, build_stiener_seed
import pickle
import time
import os

from util import (
    random_graph,
    display_tree,
    display_graph,
    random_points,
    form_grid_graph,
    round_targets_to_graph,
    bcolors,
)

# number n you put here = a(n)
#  Number of spanning trees in n x n grid
# https://oeis.org/A007341
num_span = [
    1,
    4,
    192,
    100352,
    557568000,
    32565539635200,
    19872369301840986112,
    126231322912498539682594816,
    8326627661691818545121844900397056,
    5694319004079097795957215725765328371712000,
    40325021721404118513276859513497679249183623593590784,
]


def bruteforce(G, s, targets, budget):
    best = float("-inf")
    best_tree = None
    count = 0
    spacer = 10
    for t in SpanningTreeIterator(G):
        if count % spacer == 0:
            print(f"Num Trees: ~{bcolors.OKBLUE}{count}{bcolors.ENDC}")
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
            return best_tree, best

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
    return best_tree, best


def generate_bruteforce_graphs(factory, count, prefix=None):
    # We need to store some information for later

    total_time = 0.0
    for i in range(count):
        os.makedirs(f"{prefix}/{i + 1}")
        G, s, targets, budget = factory()
        # display_graph(G)

        start = time.perf_counter()
        best_tree, best = bruteforce(G, s, targets, budget)
        end = time.perf_counter()

        if prefix != None:
            pickle.dump(G, open(f"{prefix}/{i + 1}/G.pickle", "wb"))
            pickle.dump(best_tree, open(f"{prefix}/{i + 1}/best_tree.pickle", "wb"))
            ellapsed = end - start
            total_time += ellapsed
            info = {
                "s": s,
                "targets": targets,
                "budget": budget,
                "time": ellapsed,
                "metric": best,
            }
            for k, v in info.items():
                print(k, v)
            print()
            pickle.dump(info, open(f"{prefix}/{i + 1}/info.pickle", "wb"))
        else:
            display_tree(G, best_tree)
        print()

    if prefix != None:
        stats = {
            "count": count,
            "total_time": ellapsed,
            "avg_time": ellapsed / count,
        }
        for k, v in stats.items():
            print(k, v)
        print()
        pickle.dump(stats, open(f"{prefix}/stats.pickle", "wb"))


def sample_algo(prefix, count, rounds):
    for i in range(count):
        G = pickle.load(open(f"{prefix}/{i + 1}/G.pickle", "rb"))

        info = pickle.load(open(f"{prefix}/{i + 1}/info.pickle", "rb"))
        s = info["s"]
        targets = info["targets"]
        budget = info["budget"]

        best_algo_metric = float("-inf")
        best_algo_tree = None
        for _ in range(rounds):
            res, pred, _ = compute_tree(G, s, targets, budget)
            forced, metric, _ = compute_metric(res, s, targets, pred)
            metric = 0.0 if forced else metric
            if metric > best_algo_metric:
                best_algo_metric = metric
                best_algo_tree = res

        pickle.dump(best_algo_tree, open(f"{prefix}/{i + 1}/best_algo.pickle", "wb"))
        algo_res = {
            "metric": best_algo_metric,
        }
        pickle.dump(algo_res, open(f"{prefix}/{i + 1}/algo_res.pickle", "wb"))


def sample_rand(prefix, count, rounds):
    spacer = 50
    for i in range(count):
        if i % spacer == spacer - 1:
            print(f"{bcolors.CLEAR_LAST_LINE}")
        if i % spacer == 0:
            print(f"Num Samples: ~{bcolors.OKBLUE}{i}{bcolors.ENDC}")
        G = pickle.load(open(f"{prefix}/{i + 1}/G.pickle", "rb"))

        info = pickle.load(open(f"{prefix}/{i + 1}/info.pickle", "rb"))
        s = info["s"]
        targets = info["targets"]
        budget = info["budget"]

        best_rand_metric = float("-inf")
        best_rand_tree = None
        for _ in range(rounds):
            size = float("inf")
            while size > budget or size == float("inf"):  # TODO: Add failsafe here
                res, pred = build_stiener_seed(G, s, targets, minimum=None)
                size = res.size(weight="weight")
            forced, metric, _ = compute_metric(res, s, targets, pred)
            metric = metric if not forced else 0.0
            if metric > best_rand_metric:
                best_rand_metric = metric
                best_rand_tree = res

        pickle.dump(best_rand_tree, open(f"{prefix}/{i + 1}/best_rand.pickle", "wb"))
        rand_res = {
            "metric": best_rand_metric,
        }
        pickle.dump(rand_res, open(f"{prefix}/{i + 1}/rand_res.pickle", "wb"))


def time_bruteforce(n, factory, count, samples):
    total_time = 0.0
    spacer = 10
    for i in range(count):
        print(
            f"Sampling Bruteforce on Graph {i + 1} / {count} of size {n + 1} x {n + 1} nodes"
        )
        num_trees = 0
        G, s, targets, budget = factory(n)
        start = time.perf_counter()
        while num_trees < samples:
            for t in SpanningTreeIterator(G):
                if num_trees % spacer == 0:
                    print(f"Num Trees: ~{bcolors.OKBLUE}{num_trees}{bcolors.ENDC}")

                if num_trees >= samples:
                    break
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

                # compute metric
                forced, metric, target_metrics = compute_metric(
                    t, s, targets, pred=pred
                )

                if num_trees % spacer == spacer - 1:
                    print(f"{bcolors.CLEAR_LAST_LINE}")
                num_trees += 1
        end = time.perf_counter()

        total_time += end - start
        print()

    avg_time = total_time / count
    return avg_time


def time_random(n, factory, count, samples):
    total_time = 0.0
    spacer = 50
    for i in range(count):
        print(
            f"Sampling Random on Graph {i + 1} / {count} of size {n + 1} x {n + 1} nodes"
        )
        num_trees = 0
        G, s, targets, budget = factory(n)
        start = time.perf_counter()
        for num_trees in range(samples):
            if num_trees % spacer == 0:
                print(f"Num Trees: ~{bcolors.OKBLUE}{num_trees}{bcolors.ENDC}")

            size = float("inf")
            while size > budget or size == float("inf"):  # TODO: Add failsafe here
                res, pred = build_stiener_seed(G, s, targets, minimum=None)
                size = res.size(weight="weight")
            forced, metric, _ = compute_metric(res, s, targets, pred)

            if num_trees % spacer == spacer - 1:
                print(f"{bcolors.CLEAR_LAST_LINE}")
            num_trees += 1

        end = time.perf_counter()
        total_time += end - start
        print()

    avg_time = total_time / count
    return avg_time


def main():
    pass

    # TODO: Remove all of this dead code into proper benchmarks in results.py

    # # Goal: Show that random benchmarking does not scale well
    # #   Plot n by # of spanning trees in n x n graph / # number of random trees we can generate in that time

    # # number of graphs to generate
    # count = 2
    # # number of random graphs to bruteforce
    # samples = 10000

    # # graph parameters
    # target_count = 2
    # n_min, n_max = 3, 6

    # def factory(n):
    #     s, targets = random_points(target_count)

    #     G = form_grid_graph(s, targets, n, n, triangulate=False)

    #     round_targets_to_graph(G, s, targets)
    #     targets = [f"target {i}" for i in range(target_count)]
    #     s = "start"
    #     nx.set_node_attributes(G, 0, "paths")

    #     budget = float("inf")

    #     # rescale weights
    #     for u, v in G.edges:
    #         G[u][v]["weight"] = G[u][v]["weight"]

    #     return G, s, targets, budget

    # brute_times = [
    #     time_bruteforce(n, factory, count, samples) / samples
    #     for n in range(n_min, n_max + 1)
    # ]
    # extrapolated_bruteforce = [
    #     brute_times[i] * a[n_min + i] for i in range(len(brute_times))
    # ]

    # rand_times = [
    #     time_random(n, factory, count, samples) / samples
    #     for n in range(n_min, n_max + 1)
    # ]
    # extrapolated_random = [rand_times[i] * a[n_min + i] for i in range(len(rand_times))]

    # print("Extrapolated Times")
    # for i, (extrap_time, rand_time) in enumerate(
    #     zip(extrapolated_bruteforce, extrapolated_random)
    # ):
    #     print(f"    {n_min + i + 1} x {n_min + i + 1} nodes")
    #     print(f"        Number of Graphs to Iterate Through: {a[n_min + i]}")
    #     print(f"        Bruteforce Time: {extrap_time}s")
    #     print(f"        Bruteforce Time: {extrap_time / 3600}h")
    #     print(f"        Random Time:     {rand_time}s")
    #     print(f"        Random Time:     {rand_time / 3600}h")
    #     print()

    # # TODO: Put all of this into individual functions

    # ##################################
    # # GENERATE GRAPHS AND BRUTEFORCE #
    # ##################################

    # loc = "results/brute"
    # n = 1

    # for i in range(n):
    #     if os.path.exists(f"{loc}/{i + 1}/"):
    #         print("Remove files and rerun bruteforce")
    #         return

    # # Initial Parameters
    # target_count = 2
    # graphx = graphy = 3
    # print(f"Total Number of Trees: {bcolors.FAIL}{a[graphx]}{bcolors.ENDC}")

    # def factory():
    #     s, targets = random_points(target_count)

    #     # G = form_grid_graph(s, targets, graphx, graphy)
    #     G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
    #     # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
    #     # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

    #     round_targets_to_graph(G, s, targets)
    #     targets = [f"target {i}" for i in range(target_count)]
    #     s = "start"
    #     nx.set_node_attributes(G, 0, "paths")

    #     budget = float("inf")
    #     # budget = nx.minimum_spanning_tree(G).size(weight="weight") * 0.5

    #     # rescale weights
    #     for u, v in G.edges:
    #         G[u][v]["weight"] = G[u][v]["weight"]

    #     return G, s, targets, budget

    # generate_bruteforce_graphs(factory, n, prefix=loc)

    # ########################################
    # # GET TIMINGS FOR ALGORITHM AND RANDOM #
    # ########################################

    # stats = pickle.load(open(f"{loc}/stats.pickle", "rb"))

    # # budget in seconds
    # allotted = stats["avg_time"]
    # count = stats["count"]

    # samples = 10000
    # spacer = 100

    # algo_time = 0.0
    # for i in range(samples):
    #     if i % spacer == 0:
    #         print(f"Num Sampless: ~{bcolors.OKBLUE}{i}{bcolors.ENDC}")
    #     G, s, targets, budget = factory()

    #     start_time = time.perf_counter()
    #     res, _, _ = compute_tree(G, s, targets, budget)
    #     forced, metric, _ = compute_metric(res, s, targets)
    #     end_time = time.perf_counter()

    #     algo_time += end_time - start_time
    #     if i % spacer == spacer - 1:
    #         print(f"{bcolors.CLEAR_LAST_LINE}")
    # algo_time /= samples

    # rounds = int(allotted / algo_time)

    # stats["algo_time"] = algo_time
    # stats["algo_rounds"] = rounds

    # sample_algo(loc, count, rounds)

    # print("Doing Random Sampling")
    # rand_time = 0.0
    # for i in range(samples):
    #     if i % spacer == spacer - 1:
    #         print(f"{bcolors.CLEAR_LAST_LINE}")
    #     if i % spacer == 0:
    #         print(f"Num Samples: ~{bcolors.OKBLUE}{i}{bcolors.ENDC}")
    #     G, s, targets, budget = factory()

    #     start_time = time.perf_counter()
    #     size = float("inf")
    #     while size > budget or size == float("inf"):
    #         rst, pred = build_stiener_seed(G, s, targets, minimum=None)
    #         size = rst.size(weight="weight")
    #     forced, metric, _ = compute_metric(rst, s, targets, pred=pred)
    #     end_time = time.perf_counter()

    #     rand_time += end_time - start_time
    # rand_time /= samples

    # rounds = int(allotted / rand_time)

    # stats["rand_time"] = rand_time
    # stats["rand_rounds"] = rounds

    # sample_rand(loc, count, rounds)

    # pickle.dump(stats, open(f"{loc}/stats.pickle", "wb"))

    # ###################
    # # GET FINAL STATS #
    # ###################

    # stats = pickle.load(open(f"{loc}/stats.pickle", "rb"))
    # for k, v in stats.items():
    #     print(f"{k}: {v}")
    # print()

    # for i in range(count):
    #     if os.path.exists(f"{loc}/{i + 1}/info.pickle"):
    #         brute_res = pickle.load(open(f"{loc}/{i + 1}/info.pickle", "rb"))
    #         print(f"    Bruteforce Metric: {brute_res['metric']}")

    #     if os.path.exists(f"{loc}/{i + 1}/algo_res.pickle"):
    #         algo_res = pickle.load(open(f"{loc}/{i + 1}/algo_res.pickle", "rb"))
    #         print(f"    Algorithm Metric:  {algo_res['metric']}")

    #     if os.path.exists(f"{loc}/{i + 1}/rand_res.pickle"):
    #         rand_res = pickle.load(open(f"{loc}/{i + 1}/rand_res.pickle", "rb"))
    #         print(f"    Random Metric:     {rand_res['metric']}")

    #     print()


if __name__ == "__main__":
    main()
