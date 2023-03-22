import os
import time

import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt

from algo import brute_force, build_stiener_seed, compute_metric, compute_tree
from util import *


def random_bench(n, G, s, targets, budget, loc=None):
    # Build n random spanning trees over G, compute metric, take max

    best = float("-inf")
    for i in range(n):
        print(
            f"Generating Random Spanning Tree {bcolors.OKGREEN}{i + 1}/{n}{bcolors.ENDC}"
        )
        size = float("inf")
        while size > budget:  # TODO: Add failsafe here
            rst, pred = build_stiener_seed(G, s, targets, minimum=False)
            size = rst.size(weight="weight")
        forced, metric, _ = compute_metric(rst, s, targets)
        res = metric if not forced else 0.0
        best = max(best, res)
        print(bcolors.CLEAR_LAST_LINE)

    return best


def benchmark(n, factory, loc=None, brute=False):
    # To make life a little easier, this function will require you to pass a function
    # which takes no arguments and returns a graph G, start s, targets, and budget
    # So make your own factory method
    #
    # compute many improvements

    metric_before_min = []
    metric_after_min = []
    metric_before_max = []
    metric_after_max = []
    metric_brute = []

    minimum_stats = {
        "both_forced": 0,
        "now_unforced": 0,
        "unimproved": 0,
        "improved": 0,
    }
    maximum_stats = {
        "both_forced": 0,
        "now_unforced": 0,
        "unimproved": 0,
        "improved": 0,
    }

    heur_times_min = []
    heur_times_max = []
    brute_times = []

    for i in range(n):
        print(f"Starting benchmarks {i} / {n - 1}")
        # Set up graph, seed tree, and file locations.
        G, s, targets, budget = factory()

        # Do benchmark on minimum spanning tree
        curr_loc = f"{loc}/{i}_min" if loc != None else None

        # get before for minimum
        mst, pred = build_stiener_seed(G, s, targets)
        forced, metric, target_list = compute_metric(mst, s, targets, pred)
        before = metric if not forced else 0.0
        metric_before_min.append(before)

        # get after for minimum
        start_time = time.perf_counter()
        mst, pred, rounds = compute_tree(G, s, targets, budget, loc=curr_loc)
        end_time = time.perf_counter()
        heur_times_min.append(end_time - start_time)
        forced, metric, target_list = compute_metric(mst, s, targets, pred)
        after = metric if not forced else 0.0
        metric_after_min.append(after)

        improvement = True if after > before else False
        if improvement:
            minimum_stats["improved"] += 1
            print("Made Improvements")
            if before == 0.0:
                minimum_stats["now_unforced"] += 1
                print("    Now Unforced")
            else:
                print(f"    Before: {before}")
                print(f"    After:  {after}")
        else:
            minimum_stats["unimproved"] += 1
            print("No Improvements")
            if after == 0.0:
                minimum_stats["both_forced"] += 1
                print("    Still Forced")
        print(f"Benchmark {i} / {n - 1} took {end_time} seconds")
        print(f"Took {rounds} rounds")
        print()

        if brute:
            print("Starting brute force computation")
            start_time = time.perf_counter()
            best_tree, metric = brute_force(G, s, targets, budget, loc=loc)
            end_time = time.perf_counter()
            brute_times.append(end_time - start_time)

    # stat computation galore
    print(f"Number of graphs = {n}")
    print()
    print("Minimum Spanning Seed Tree:")
    print("    both_forced  =", minimum_stats["both_forced"])
    print("    now_unforced =", minimum_stats["now_unforced"])
    print("    unimproved   =", minimum_stats["unimproved"])
    print("    improved     =", minimum_stats["improved"])
    print(f"Longest Minimum Heuristic Run  = {max(heur_times_min)} seconds")
    print(f"Shortest Minimum Heuristic Run = {min(heur_times_min)} seconds")
    print(f"Average Minimum Heuristic Run  = {sum(heur_times_min) / n} seconds")
    print()
    if brute:
        print(f"longest brute calc     = {max(brute_times)}")
    print()
    # TODO: Smarter Stats


def create_heatmap(min_width, max_width, target_min, target_max, loc=None):
    if loc is not None:
        avgs = [[0.0 for _ in range(max_width + 1)] for _ in range(target_max + 1)]
        with open(f"{loc}/heatmap.txt", "r") as f:
            for line in f:
                if "width" not in line:
                    parts = line.split()
                    width = int(parts[0])
                    target_count = int(parts[1])
                    avg = float(parts[2])
                    avgs[target_count][width] = avg

        sns.set()
        sns.heatmap(avgs)
        filename = f"{loc}/heatmap.png"
        print(f"saving {filename}")
        plt.savefig(filename)
        plt.close()

def heatmap(min_width, max_width, target_min, target_max, rounds, loc=None):
    # Create a heatmap of average times of running the algorithm
    # on various size graphs between min_width and max_width
    # ranges are inclusive
    # For now only triangulated grid graph
    # Saves progress to textfile in case of hang

    with open(f"{loc}/heatmap.txt", "w") as f:
        f.write("width, target, time,      average_number_of_rounds\n")
        avgs = [[0.0 for _ in range(max_width + 1)] for _ in range(target_max + 1)]
        for width in range(min_width, max_width + 1):
            for target_count in range(target_min, target_max + 1):

                def factory():
                    s, targets = random_points(target_count)

                    G = form_grid_graph(s, targets, width, width)
                    # G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
                    # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
                    # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

                    round_targets_to_graph(G, s, targets)
                    targets = [f"target {i}" for i in range(target_count)]
                    s = "start"
                    nx.set_node_attributes(G, 0, "paths")

                    # budget = float("inf")
                    budget = nx.minimum_spanning_tree(G).size(weight="weight") * 1.5

                    return G, s, targets, budget

                total_time = 0.0
                total_rounds = 0
                for round in range(rounds):
                    print(f"Round {round}: target count = {target_count}, {width = }")
                    G, s, targets, budget = factory()
                    start_time = time.perf_counter()
                    mst, pred, count = compute_tree(G, s, targets, budget, loc=None)
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                    total_rounds += count
                avgs[target_count][width] = total_time / rounds

                print(
                    f"{rounds} rounds with target count = {target_count}, {width = } took {total_time} seconds"
                )
                print(f"    or   {total_time / 60} minutes")
                print(f"and {total_rounds} rounds")
                print(f"")

                width_s = f"{width:<6}"
                target_s = f"{target_count:<7}"
                time_s = f"{avgs[target_count][width]:.5f}"
                time_s = f"{time_s:<10}"
                rounds_s = f"{total_rounds / rounds:.5f}"
                f.write(
                    f"{width_s} {target_s} {time_s} {rounds_s}\n"
                )

    create_heatmap(min_width, max_width, target_min, target_max, loc=loc)


def main():
    # Initial Parameters
    target_count = 3
    graphx = 10
    graphy = 10

    def factory():
        s, targets = random_points(target_count)

        G = form_grid_graph(s, targets, graphx, graphy)
        # G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
        # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
        # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

        round_targets_to_graph(G, s, targets)
        targets = [f"target {i}" for i in range(target_count)]
        s = "start"
        nx.set_node_attributes(G, 0, "paths")

        # budget = float("inf")
        budget = nx.minimum_spanning_tree(G).size(weight="weight") * 1.5

        return G, s, targets, budget

    ##################
    # MASS BENCHMARK #
    ##################

    # brute = False  # WARNING: This is really really slow

    # bench_count = 1
    # if os.path.exists("images/current/*"):
    #     print("Remove files and rerun benchmark")
    #     return

    # for i in range(bench_count):
    #     os.makedirs(f"images/current/{i}_min")
    #     if brute:
    #         os.makedirs(f"images/current/{i}_brute")

    # benchmark(bench_count, factory, loc="images/current", brute=brute)

    #####################
    # HEATMAP BENCHMARK #
    #####################

    if not os.path.exists("images/heatmap/"):
        os.makedirs("images/heatmap/")

    min_width = 5
    max_width = 7
    target_min = 2
    target_max = 3
    rounds = 1

    heatmap(min_width, max_width, target_min, target_max, rounds, loc="images/heatmap")

    ####################
    # RANDOM BENCHMARK #
    ####################
    # benches = 3
    # num_rand = 0
    # rand_res = []
    # rand_times = []
    # algo_res = []
    # algo_times = []
    # for _ in range(benches):
    #     G, s, targets, budget = factory()
    #     for u, v, dat in G.edges(data=True):
    #         assert "weight" in dat

    #     start_time = time.perf_counter()
    #     rand_res.append(random_bench(num_rand, G, s, targets, budget))
    #     end_time = time.perf_counter()
    #     rand_times.append(end_time - start_time)

    #     start_time = time.perf_counter()
    #     mst, pred, rounds = compute_tree(G, s, targets, budget, loc=None)
    #     end_time = time.perf_counter()
    #     algo_times.append(end_time - start_time)
    #     forced, metric, target_list = compute_metric(mst, s, targets, pred)
    #     algo_res.append(metric if not forced else 0.0)

    # alg_beat = 0
    # for rand, alg in zip(rand_res, algo_res):
    #     if alg >= rand:
    #         alg_beat += 1

    # print(f"Algorithm beat random spanning trees {alg_beat}/{benches} times")
    # print(f"Average Random Spanning Tree Run = {sum(rand_times) / benches} seconds")
    # print(f"Average Algorithm Run            = {sum(algo_times) / benches} seconds")


if __name__ == "__main__":
    main()
