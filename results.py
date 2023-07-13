import time
import imageio
import os

import networkx as nx
from math import ceil
import pickle
import matplotlib.pyplot as plt
import numpy as np
from algo import compute_tree, build_stiener_seed, compute_metric
from util import (
    random_points,
    form_grid_graph,
    round_targets_to_graph,
    form_hex_graph,
    form_triangle_graph,
    bcolors,
)
from benchmark import (
    random_bench,
    mixed_benchmark,
    read_mixed_benchmark,
    create_mixed_graphs,
    test_budget,
)
from bruteforce import generate_bruteforce_graphs, num_span


def determine_budget(
    factory,
    num_graphs,
    budget_mult_low,
    budget_mult_high,
    gran,
    random_samples,
    loc=None,
):
    # run algorithm with MST seed + random spanning trees on various budgets
    # compare to best trees on some graphs

    graphs = []
    trees = []
    starts = []
    targets = []
    metrics = []
    if loc != None:
        txt = open(f"{loc}/data.txt", "w")

    results = []
    for i in range(num_graphs):
        G, s, tars, _ = factory()
        print(f"Testing graph {i + 1}")
        mst, _ = build_stiener_seed(G, s, tars, minimum=True)
        size = mst.size(weight="weight")
        budget_low = size * budget_mult_low
        budget_high = size * budget_mult_high
        interval = (budget_high - budget_low) / (gran - 1)

        if loc != None:
            txt.write(f"Graph {i + 1}:\n")

        mst_results = []
        avg_results = []
        for j in range(gran):
            print(f"    Testing gran {j + 1}/{gran}")
            budget = budget_low + j * interval

            # test minimum spanning tree seed
            output, pred, rounds = compute_tree(G, s, tars, budget, minimum=True)
            forced, metric, _ = compute_metric(output, s, tars)
            mst_res = metric if not forced else 0.0
            mst_results.append(round(mst_res, 2))

            # get average metric using random trees
            avg_res = 0.0
            for k in range(random_samples):
                print(f"        Testing random sample {k + 1}/{random_samples}")
                output, pred, rounds = compute_tree(
                    G, s, tars, budget, loc=None, minimum=None
                )
                # TODO: figure out something better for when output == None
                if output != None:
                    forced, metric, _ = compute_metric(output, s, tars)
                    res = metric if not forced else 0.0
                    avg_res += res
                else:
                    avg_res += 0
            avg_res /= random_samples
            avg_results.append(round(avg_res, 2))

            if loc != None:
                txt.write(f"    Budget: {budget}\n")
                txt.write(f"    Metric w/ MST seed:  {mst_res}\n")
                txt.write(f"    Metric w/ Rand seed: {avg_res}\n")
                txt.write("\n")

        results.append((mst_results, avg_results))
        if loc != None:
            txt.write("\n")

    if loc != None:
        txt.close()

    # generate graphs and stats and such
    if loc != None:
        for i in range(num_graphs):
            # TODO: come up with better x-axis labels
            interval = (budget_mult_high - budget_mult_low) / gran
            x_axis_labels = [
                f"{round(budget_mult_low + (j + 1) * interval, 2)}" for j in range(gran)
            ]
            mst_results, avg_results = results[i]
            data = {
                "MST Seed": mst_results,
                "Rand Seed": avg_results,
            }

            x = np.arange(len(x_axis_labels))
            width = 0.15
            multiplier = 0

            fig, ax = plt.subplots()
            fig.set_figwidth(25)

            for attribute, measurement in data.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                # ax.bar_label(rects, padding=3)
                multiplier += 1

            ax.set_ylabel("Metric")
            ax.set_xlabel(
                "Budget Increment \n Budget = (Budget Increment) * MST_Weight"
            )
            ax.set_title(f"Graph {i + 1}")
            ax.set_xticks(x + width, x_axis_labels)
            ax.legend(loc="upper left", ncols=3)
            fig.subplots_adjust(bottom=0.25, top=0.75)
            filename = f"{loc}/metrics_{i + 1}.png"
            plt.savefig(filename)
            # plt.show()
            plt.close()


def brute_comparison(loc, brute_loc, num_graphs, random_samples):
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
    vals = [100, round(mst_avg, 2), round(rand_avg, 2)]
    trees = ["Best Tree", "MST Seed", "Rand Seed"]
    ax.bar(trees, vals)
    ax.bar_label(ax.containers[0], label_type="edge")
    ax.set_ylabel("% Diff to Best Metric")
    ax.legend()

    filename = f"{loc}/results.png"
    plt.savefig(filename)
    plt.show()
    plt.close()


def main():
    # ##################################
    # # GENERATE GRAPHS AND BRUTEFORCE #
    # ##################################

    # loc = "results/brute"
    # n = 10

    # for i in range(n):
    #     if os.path.exists(f"{loc}/{i + 1}/"):
    #         print("Remove files and rerun bruteforce")
    #         return

    # # Initial Parameters
    # target_count = 2
    # graphx = graphy = 3
    # print(f"Total Number of Trees: {bcolors.FAIL}{num_span[graphx]}{bcolors.ENDC}")

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

    #     # # rescale weights
    #     # for u, v in G.edges:
    #     #     G[u][v]["weight"] = G[u][v]["weight"]

    #     return G, s, targets, budget

    # generate_bruteforce_graphs(factory, n, prefix=loc)

    #############################################
    # BENCHMARK REATTACHMENT AGAINST BRUTEFORCE #
    #############################################

    loc = "results/brute_comparison"
    brute_loc = "results/brute"
    num_graphs = 10
    random_samples = 100
    brute_comparison(loc, brute_loc, num_graphs, random_samples)

    # ###############################
    # # DETERMINE BUDGET MULTIPLIER #
    # ###############################

    # # Initial Parameters
    # target_count = 2
    # graphx = graphy = 4

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

    #     # # rescale weights
    #     # for u, v in G.edges:
    #     #     G[u][v]["weight"] = G[u][v]["weight"]

    #     return G, s, targets, budget

    # determine_budget(factory, 10, 1, 3, 60, 25, loc="results/budget")

    # # Initial Parameters
    # target_count = 6
    # graphx = 20
    # graphy = 20

    # def factory():
    #     s, targets = random_points(target_count)

    #     G = form_grid_graph(s, targets, graphx, graphy)
    #     # G = form_grid_graph(s, targets, graphx, graphy, triangulate=False)
    #     # G = form_hex_graph(s, targets, graphx, graphy, 1.0)
    #     # G = form_triangle_graph(s, targets, graphx, graphy, 1.0)

    #     round_targets_to_graph(G, s, targets)
    #     targets = [f"target {i}" for i in range(target_count)]
    #     s = "start"
    #     nx.set_node_attributes(G, 0, "paths")

    #     # budget = float("inf")
    #     budget = nx.minimum_spanning_tree(G).size(weight="weight") * 0.5

    #     # rescale weights
    #     for u, v in G.edges:
    #         G[u][v]["weight"] = G[u][v]["weight"]

    #     return G, s, targets, budget

    ####################
    # BUDGET BENCHMARK #
    ####################

    # G, s, targets, _ = factory()
    # mst, _ = build_stiener_seed(G, s, targets, minimum=True)
    # cost = mst.size(weight="weight")
    # budget_low = 0.9 * cost
    # budget_high = 3.0 * cost
    # n = 30
    # loc = "results"
    # test_budget(G, s, targets, budget_low, budget_high, n, loc=loc)

    # # make gif
    # frames = []
    # for i in range(n):
    #     if os.path.exists(f"results/{i + 1}.png"):
    #         image = imageio.v2.imread(f"results/{i + 1}.png")
    #         frames.append(image)
    # imageio.mimsave(f"{loc}/budgets.gif", frames, duration=300)

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

    # if not os.path.exists("images/heatmap/"):
    #     os.makedirs("images/heatmap/")

    # min_width = 5
    # max_width = 7
    # target_min = 2
    # target_max = 3
    # rounds = 1

    # heatmap(min_width, max_width, target_min, target_max, rounds, loc="images/heatmap")

    ####################
    # RANDOM BENCHMARK #
    ####################

    # benches = 5
    # num_rand = 0  # 0 means use number of rounds
    # rand_res = []
    # rand_attempts = []
    # rand_times = []
    # algo_res = []
    # algo_times = []
    # algo_rounds = []
    # for i in range(benches):
    #     G, s, targets, budget = factory()

    #     if not os.path.exists(f"images/{i}/"):
    #         os.makedirs(f"images/{i}/")

    #     start_time = time.perf_counter()
    #     mst, pred, rounds = compute_tree(G, s, targets, budget, loc=f"images/{i}")
    #     end_time = time.perf_counter()
    #     algo_times.append(end_time - start_time)
    #     forced, metric, target_list = compute_metric(mst, s, targets, pred)
    #     algo_res.append(metric if not forced else 0.0)

    #     # The number of rounds generated by heuristic algorithm
    #     #   is the number of random trees generated
    #     num_trees = rounds if num_rand == 0 else num_rand
    #     start_time = time.perf_counter()
    #     rand_metric, attempts = random_bench(
    #         num_trees, G, s, targets, budget, loc=f"images/{i}/random"
    #     )
    #     rand_res.append(rand_metric)
    #     rand_attempts.append(attempts)
    #     end_time = time.perf_counter()
    #     rand_times.append(end_time - start_time)
    #     algo_rounds.append(rounds)

    # alg_beat = 0
    # rand_beat_alg = []
    # alg_beat_rand = []
    # alg_forced = 0
    # rand_forced = 0
    # both_forced = 0
    # for rand, alg in zip(rand_res, algo_res):
    #     if alg > rand:
    #         alg_beat += 1
    #     if rand > 0 and alg > 0:
    #         if rand >= alg:
    #             rand_beat_alg.append((rand - alg) / alg * 100)
    #         elif alg >= rand:
    #             alg_beat_rand.append((alg - rand) / rand * 100)
    #     else:
    #         if rand == alg == 0:
    #             both_forced += 1
    #         if rand == 0:
    #             rand_forced += 1
    #         if alg == 0:
    #             alg_forced += 1

    # print(f"Algorithm beat random spanning trees {alg_beat}/{benches} times")
    # if len(alg_beat_rand) > 0:
    #     print(
    #         f"    Algorithm was on average {sum(alg_beat_rand) / len(alg_beat_rand)}% better"
    #     )
    # print(f"Algorithm produced {alg_forced} forced trees")
    # print(
    #     f"Algorithm produced {int(sum(algo_rounds) / len(algo_rounds))} trees on average"
    # )
    # print(f"Random spanning trees beat algorithm {benches - alg_beat}/{benches} times")
    # if len(rand_beat_alg) > 0:
    #     print(
    #         f"    Random spanning trees was on average {sum(rand_beat_alg) / len(rand_beat_alg)}% better"
    #     )
    # print(f"Random spanning trees produced {rand_forced} forced trees")
    # print(f"Average Random Spanning Tree Run = {sum(rand_times) / benches} seconds")
    # print(f"Average Algorithm Run            = {sum(algo_times) / benches} seconds")

    # print(f"Both algorithms produced forced trees {both_forced} times")

    # # compute ratio of avg algo time / avg rand time
    # algo_weight = ceil((sum(algo_times) / benches) / (sum(rand_times) / benches))
    # print(f"{algo_weight = }")

    # ###################
    # # MIXED BENCHMARK #
    # ###################

    # # Compute weights
    # samples = 20

    # rand_time = 0.0
    # algo_time = 0.0
    # for _ in range(samples):
    #     G, s, targets, budget = factory()
    #     start_time = time.perf_counter()
    #     rand_metric, attempts = random_bench(1, G, s, targets, budget)
    #     end_time = time.perf_counter()
    #     rand_time += end_time - start_time

    #     start_time = time.perf_counter()
    #     _, _, _ = compute_tree(G, s, targets, budget)
    #     end_time = time.perf_counter()
    #     algo_time += end_time - start_time
    # rand_time /= samples
    # algo_time /= samples
    # algo_weight = ceil(algo_time / rand_time)

    # print(f"Algo Weight = {algo_weight}")

    # end = 10
    # total = algo_weight * end
    # n = 20
    # loc = "images/mixed"
    # mixed_benchmark(total, algo_weight, n, 0, end, factory, loc=loc, jump=1)
    # rand_res, algo_res = read_mixed_benchmark(loc)
    # create_mixed_graphs(rand_res, algo_res, loc=loc)


if __name__ == "__main__":
    main()
