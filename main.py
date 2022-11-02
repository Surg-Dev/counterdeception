import networkx as nx
import random as rd

def maximize_minimum_reaction_time(G, s, t, h):
    # Some A* implementation
    pass

def closure_heuristic(G, s, t):
    paths = nx.single_source_dijkstra_path_length(G, s)
    def heuristic(n):
        if n['paths'] == 0:
            return 1.0
        p = nx.shortest_path_length(G, n, t)
        return p/(p+paths[n])
    return heuristic

def random_graph(n, ts, ef=2.5):
    V = n
    E = int(ef*n)
    G = nx.gnm_random_graph(V,E)

    while (not nx.is_connected(G)):
        G = nx.gnm_random_graph(V,E)
    
    nx.set_edge_attributes(G, rd.randint(1,500), 'weight')
    nx.set_node_attributes(G, 0, 'paths')

    targets = rd.sample(G.nodes, ts+1)
    s = targets[0]
    targets = targets[1:]

    return G, s, targets

def main():
    G, s, targets = random_graph(40, 5)
    h = closure_heuristic(G, s, targets)
    #maximize_minimum_reaction_time(G, s, targets, h)

if __name__ == "__main__":
    main()