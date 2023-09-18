import time
import imageio
import os

import signal
import networkx as nx
from math import ceil
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from algo import compute_tree, build_stiener_seed, compute_metric
from util import (
    random_points,
    form_grid_graph,
    round_targets_to_graph,
    form_hex_graph,
    form_triangle_graph,
    bcolors,
    display_graph,
    display_tree,
)
from bruteforce import generate_bruteforce_graphs, num_span


def count_iterations_bench(wl, wh, tl, th, num_graphs, loc):
    # Count the average number of iterations done by Reattachment on graphs
    #   of various sizes and targets
    #
    # wl, wh: The low and high range of the number of nodes on
    #   the sides of the square grid graphs
    # tl, th: The low and high range of number of targets
    # num_graphs: The number of graphs to randomly generate
    # loc: The location to save res.txt, the output
    with open(f"{loc}/res.txt", "w") as f:
        for w in range(wl, wh + 1):
            for t in range(tl, th + 1):
                avg = 0

                def factory():
                    s, targets = random_points(t)

                    G = form_grid_graph(s, targets, w - 1, w - 1, triangulate=False)

                    round_targets_to_graph(G, s, targets)
                    targets = [f"target {i}" for i in range(t)]
                    s = "start"
                    nx.set_node_attributes(G, 0, "paths")

                    budget = float("inf")

                    return G, s, targets, budget

                for _ in range(num_graphs):
                    G, s, targets, budget = factory()
                    _, _, rounds = compute_tree(G, s, targets, budget, minimum=True)
                    avg += rounds

                avg /= num_graphs
                f.write(f"{w} x {w} / {t} = {avg}\n")
                print(f"{w} x {w} / {t} = {avg}")


def bruteforce_bench(num_graphs, size, target_count, loc):
    # Generate graphs and bruteforce the most counterdeceptive graphs on these
    #
    # num_graphs: The number of square grid graphs to generate
    # size: The number of nodes on the sides of the square grid graphs
    # target_count: The number of targets to place in the graph
    # loc: The location to save the outputs

    for i in range(num_graphs):
        if os.path.exists(f"{loc}/{i + 1}/"):
            print("Remove files and rerun bruteforce")
            return

    # Initial Parameters
    graphx = graphy = size - 1
    print(f"Total Number of Trees: {bcolors.FAIL}{num_span[graphx]}{bcolors.ENDC}")

    def factory():
        s, targets = random_points(target_count)

        G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)

        round_targets_to_graph(G, s, targets)
        targets = [f"target {i}" for i in range(target_count)]
        s = "start"
        nx.set_node_attributes(G, 0, "paths")

        budget = float("inf")

        return G, s, targets, budget

    generate_bruteforce_graphs(factory, num_graphs, prefix=loc)


def brute_comparison(loc, brute_loc, num_graphs, random_samples):
    # Run reattachment multiple times
    #
    # loc: The location to save the outputs
    # brute_loc: The location of the bruteforced square grid graphs
    # num_graphs: The number of square grid graphs to use
    # random_samples: The number of times to run Reattachment with
    #   random spanning seed tree

    budget = float("inf")

    mst_avg = 0
    rand_avg = 0

    for i in range(num_graphs):
        # get relevant info
        G_f = open(f"{brute_loc}/{i + 1}/G.pickle", "rb")
        G = pickle.load(G_f)
        best_tree_f = open(f"{brute_loc}/{i + 1}/best_tree.pickle", "rb")
        best_tree = pickle.load(best_tree_f)
        info_f = open(f"{brute_loc}/{i + 1}/info.pickle", "rb")
        info = pickle.load(info_f)
        best_metric = info["metric"]
        s = info["s"]
        targets = info["targets"]
        G_f.close()
        best_tree_f.close()
        info_f.close()

        # run reattachment with mst seed
        output, pred, rounds = compute_tree(G, s, targets, budget, minimum=True)
        forced, metric, _ = compute_metric(output, s, targets)
        mst_res = metric if not forced else 0.0

        # get average metric using random trees
        avg_res = 0.0
        for k in range(random_samples):
            output, pred, rounds = compute_tree(G, s, targets, budget, minimum=None)
            # TODO: figure out something better for when output == None
            if output != None:
                forced, metric, _ = compute_metric(output, s, targets)
                res = metric if not forced else 0.0
                avg_res += res
            else:
                avg_res += 0
        avg_res /= random_samples

        mst_avg += mst_res / best_metric * 100
        rand_avg += avg_res / best_metric * 100

    # make graphs
    fig, ax = plt.subplots()
    mst_avg /= num_graphs
    rand_avg /= num_graphs
    vals = [round(mst_avg, 2), round(rand_avg, 2)]
    trees = ["MST Seed", "Rand Seed"]
    ax.bar(trees, vals)
    ax.bar_label(ax.containers[0], label_type="edge")
    ax.set_ylabel("Avg. Metric / Optimal Metric")
    ax.legend()

    filename = f"{loc}/results.png"
    plt.savefig(filename)
    plt.show()
    plt.close()


def compare_seed_trees_diff_targets(rounds, random_samples, size, target_counts, loc):
    # Now compare the performance of different seed trees on larger graphs
    #
    # rounds: The number of rounds to take averages over
    # random_samples: The number of times to run Reattachment with
    #   random spanning seed tree
    # size: The number of nodes on the sides of the square grid graphs
    # target_counts: List of the number of targets to place in the graph
    # loc: The location to save the outputs

    mst_res = []
    rand_res = []
    equal_res = []

    # Get data for each target count
    for target_count in target_counts:
        mst_better = 0
        rand_better = 0
        both_equal = 0
        for _ in range(rounds):

            def factory():
                s, targets = random_points(target_count)

                G = form_grid_graph(s, targets, size - 1, size - 1, triangulate=false)

                round_targets_to_graph(G, s, targets)
                targets = [f"target {i}" for i in range(target_count)]
                s = "start"
                nx.set_node_attributes(G, 0, "paths")

                budget = float("inf")

                return G, s, targets, budget

            mst, avg = compare_seed_trees(factory, random_samples)
            if mst > avg:
                mst_better += 1
            elif avg > mst:
                rand_better += 1
            else:
                both_equal += 1
        mst_res.append(mst_better)
        rand_res.append(rand_better)
        equal_res.append(both_equal)

    with open(f"{loc}/{graph_size + 1}x{graph_size + 1}_data.txt", "w") as f:
        f.write(f"Graph Size = {graph_size + 1}x{graph_size + 1}\n")
        for i, target_count in enumerate(target_counts):
            f.write(f"Target Count: {target_count}\n")
            f.write(f"    MST Seed Better  = {mst_res[i]}")
            f.write(f"    Rand Seed Better = {rand_res[i]}")
            f.write(f"    Both Equal       = {equal_res[i]}")


def sprint_bench(size, target_count, num_graphs, t_low, t_high, loc):
    # Compare randomly generating trees vs Reattachment in a sprint race
    #
    # size: The number of nodes on the sides of the triangulated square grid graphs
    # target_counts: List of the number of targets to place in the graph
    # num_graphs: The number of triangulated square grid graphs to use
    # t_low, t_high: low and high range of sprint times
    # loc: The location to save the outputs

    target_count = 15
    graph_size = 24
    loc = "results/sprint"
    num_graphs = 50
    t_low, t_high = 30, 270

    def factory():
        s, targets = random_points(target_count)

        G = form_grid_graph(s, targets, graph_size - 1, graph_size - 1)

        round_targets_to_graph(G, s, targets)
        targets = [f"target {i}" for i in range(target_count)]
        s = "start"
        nx.set_node_attributes(G, 0, "paths")

        budget = float("inf")

        return G, s, targets, budget

    f = open(f"{loc}/res.txt", "w")
    for t in [t_low, t_high]:
        both_forced = 0
        algo_better = 0
        rand_better = 0
        avg_rand = 0
        avg_algo = 0
        for i in range(num_graphs):
            print(f"Graph {i + 1} / {num_graphs}")
            rand_res, algo_res, num_rand, num_algo = single_sprint_benchmark(factory, t)
            if algo_res == rand_res == 0.0:
                both_forced += 1
            elif algo_res > rand_res:
                algo_better += 1
            else:
                rand_better += 1
            avg_rand += num_rand
            avg_algo += num_algo
            print(f"    {both_forced = }")
            print(f"    {algo_better = }")
            print(f"    {rand_better = }\n")
        avg_rand /= num_graphs
        avg_algo /= num_graphs
        f.write(f"Timespan = {t}s\n")
        f.write(f"    {both_forced = }\n")
        f.write(f"    {algo_better = }\n")
        f.write(f"    {rand_better = }\n")
        f.write(f"    {avg_rand    = }\n")
        f.write(f"    {avg_algo    = }\n\n")
    f.close()


def main():
    # TODO: MAKE A PASS MATCHING REFERENCE NUMBERS

    # ####################
    # # COUNT ITERATIONS #
    # ####################
    # # Generate Table 6, the average number of iterations for
    # #   graphs of different sizes and targets
    # wl, wh = 5, 13
    # tl, th = 2, 10
    # num_graphs = 200
    # loc = "results/count"
    # count_iterations_bench(wl, wh, tl, th, num_graphs, loc)

    # ##################################
    # # GENERATE GRAPHS AND BRUTEFORCE #
    # ##################################
    # # Bruteforce the graphs used for Figure 7
    # num_graphs = 10
    # size = 4
    # target_count = 2
    # loc = "results/brute"
    # bruteforce_bench(num_graphs, size, target_count, loc)
    # ############################################################
    # # COMPARE RANDOM SEED TREE VS MST SEED TREE , SMALL GRAPHS #
    # ############################################################
    # # Run reattachment using different seed trees
    # loc = "results/brute_comparison"
    # brute_loc = "final_results/results/brute"
    # num_graphs = 10
    # random_samples = 250
    # brute_comparison(loc, brute_loc, num_graphs, random_samples)

    ############################################################
    # COMPARE RANDOM SEED TREE VS MST SEED TREE , LARGE GRAPHS #
    ############################################################
    # Table 8 comparison of seed trees on larger graphs
    loc = "results/seed_comparison"
    rounds = 250
    random_samples = 25
    target_counts = [2, 4, 7, 10]
    graph_sizes = [8, 11, 13]
    for graph_size in graph_sizes:
        compare_seed_trees_diff_targets(
            rounds, random_samples, graph_size, target_counts, loc
        )

    ####################
    # SPRINT BENCHMARK #
    ####################
    # Table 9 Sprint Benchmark
    target_count = 15
    graph_size = 24
    loc = "results/sprint"
    num_graphs = 50
    t_low, t_high = 30, 270
    sprint_bench(size, target_count, num_graphs, t_low, t_high, loc)


if __name__ == "__main__":
    main()
