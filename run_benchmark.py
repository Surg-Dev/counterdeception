from benchmark_funcs import *


# main func
def main():
    # Initial Parameters
    target_count = 5
    graphx = 10
    graphy = 10

    # Define Graph Factory
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

        budget = float("inf")
        # budget = nx.minimum_spanning_tree(G).size(weight="weight") * 0.5

        # rescale weights
        for u, v in G.edges:
            G[u][v]["weight"] = G[u][v]["weight"]

        return G, s, targets, budget
    
    # Run Benchmark
    print(sprint_benchmark(factory))



if __name__ == '__main__':
    main()