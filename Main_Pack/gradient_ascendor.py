from .ascend_gradient import optimize_positions
from .plot_3D_graph import plot_threed_graph

#TODO: Update this for variable loss functions. Right now it's hard coded. And I should get a hold of where I'm going

def graph_position_ascendor(graph, learning_rate = 0.01): #Just setting the skeleton

    """
    A function which, given a graph, minimises a loss function given 3D
    coordinates assigned to the graph as argument. ##Do this description better.

    Parameters:
    ______________
    graph: A networkx graph

    learning_rate: Step size for moving along the gradient

    """

    positions = optimize_positions(graph, learning_rate)
    plot_threed_graph(positions, graph)  # TODO: Fix so this parses