###################################################
########### DEPRECIATED, A* SHENANIGANS ###########
###################################################

import queue
import random as rd

import networkx as nx

from util import random_graph


def maximize_minimum_reaction_time(G, s, t):
    h = closure_heuristic(G, s, t)
    q = queue.PriorityQueue()
    came_from = dict()
    cost_so_far = dict()
    q.put((0, s))
    came_from[s] = s
    cost_so_far[s] = 0
    while not q.empty():
        _, n = q.get()
        if n == t:
            break
        for m in G.neighbors(n):
            new_cost = cost_so_far[n] + G[n][m]["weight"]
            if m not in cost_so_far or new_cost < cost_so_far[m]:
                cost_so_far[m] = new_cost
                priority = new_cost * h(m)
                q.put((priority, m))
                came_from[m] = n
    return came_from, cost_so_far


def closure_heuristic(G, s, t):
    paths = nx.single_source_dijkstra_path_length(G, s)
    path_counts = nx.get_node_attributes(G, "paths")

    def heuristic(n):
        p = nx.shortest_path_length(G, n, t)
        if path_counts[n] == 0:
            return 1
        print("something new")
        return p / (paths[n] + p)

    return heuristic


def main():
    G, s, targets = random_graph(40, 5)
    h = closure_heuristic(G, s, targets)
    costs = {}
    for t in targets:
        path, cost = maximize_minimum_reaction_time(G, s, t)
        ind = t
        while ind != s:
            G.nodes[ind]["paths"] += 1
            ind = path[ind]
        costs[t] = cost[t]
    true_costs = nx.single_source_dijkstra_path_length(G, s)
    for t in targets:
        print(f"Heuristic Cost: {costs[t]} True Cost: {true_costs[t]}")


if __name__ == "__main__":
    main()
