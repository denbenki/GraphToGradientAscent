import tensorflow as tf
from .ascend_gradient import optimize_positions
from .plot_3D_graph import plot_threed_graph

#TODO: Update this for variable loss functions. Right now it's hard coded. And I should get a hold of where I'm going

def graph_position_ascendor(graph, learning_rate = 0.01):

    """
    A function which, given a graph, minimises a loss function given 3D
    coordinates assigned to the graph as argument. ##Do this description better.

    Parameters:
    ______________
    graph: A networkx graph

    learning_rate: Step size for moving along the gradient

    """

    # This is going to be a little ugly, but I'll try to improve the syntax later
    positions_var_list = optimize_positions(graph, learning_rate)
    positions_list = []

    i = 0
    while i < len(positions_var_list):
        # Check the optimized positions and add them to a list of lists: sub-list  = coordinates of nodes

        a = positions_var_list[i].numpy()
        b = positions_var_list[i+1].numpy()
        c = positions_var_list[i+2].numpy()
        positions_list.append([a, b, c])

        i += 3

    positions_dict = {j: positions_list[j] for j in range(0, len(positions_list))}  # Turn it in to a dict
    plot_threed_graph(positions_dict, graph)