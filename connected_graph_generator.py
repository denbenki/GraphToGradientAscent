from itertools import combinations, groupby
import networkx as nx
import random

def gnp_random_connected_graph(n, p, seed=None): #TODO: Give credit
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    if seed is not None:
        assert isinstance(seed, int), "Random seed must be an integer"

    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        seed += 1  # Increase seed
        random.seed(seed)

        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            seed += 1  # Increase seed
            random.seed(seed)
            if random.random() < p:
                G.add_edge(*e)
    return G