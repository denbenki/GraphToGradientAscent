import matplotlib.pyplot as plt
import networkx as nx

"""A function which creates a 3D representation of a graph."""
#TODO: (Possibly) make the function work without networkx

def plot_threed_graph(positions, graph):

    assert isinstance(positions, dict)
    assert isinstance(graph, nx.classes.graph.Graph)

    c1 = []
    c2 = []
    c3 = []
    for nodes in positions.values():
        c1.append(nodes[0])
        c2.append(nodes[1])
        c3.append(nodes[2])

    ax = plt.axes(projection='3d')


    # The below plots the connections between the nodes, that is, the edges. We gather the indices of the two connected
    # nodes, gather from these the xyz-coordinates and use this to plot the edge.
    for i, j in enumerate(graph.edges()):
        x_p = [positions[j[0]][0], positions[j[1]][0]]
        y_p = [positions[j[0]][1], positions[j[1]][1]]
        z_p = [positions[j[0]][2], positions[j[1]][2]]
        ax.plot(x_p, y_p, z_p, c='black', alpha=0.5)
    ax.scatter3D(c1, c2, c3)
    for _i in range(0, len(c1)):
        ax.text(c1[_i], c2[_i], c3[_i], s=str(_i))

    plt.show()

