import tensorflow as tf
import numpy
import networkx as nx
import random

from .ascend_gradient_backup import optimize_positions


#TODO: Update this for variable loss functions. Right now it's hard coded. And I should get a hold of where I'm going

def graph_position_ascendor(graph, learning_rate=0.1):

    """
    A function which, given a graph, minimises a loss function given 3D
    coordinates assigned to the graph as argument. ##Do this description better.

    Parameters:
    ______________
    graph: A networkx graph

    learning_rate: Step size for moving along the gradient

    """

    # Let's start with getting a way of creating the variables necessary.
    adj_mat = tf.constant(numpy.asarray(nx.to_numpy_matrix(graph, dtype=numpy.float64)))  # adjacency matrix of graph; current edge weight = 1. TODO: CHeck if this is a good idea
    nu_nodes = tf.size(adj_mat[0])

    # Initialisation of positions. Should be seeded. (Problem: not a variable? Let's mekk around; let's try doing matrix of variables NOTE: Might be demanding in terms of time).
    pos_mat = tf.Variable(tf.random.uniform((nu_nodes, 3), dtype=numpy.float64))

    # Runs a function for optimizing the positions.

    positions = optimize_positions(pos_mat, adj_mat, nu_nodes, learning_rate)

    chopped_positions = []
    i = 0
    while i < len(positions):
        element = []
        for j in range(0, 3):
            element.append(positions[i + j])
        chopped_positions.append(element)
        i += 3

    positions_dict = {j: chopped_positions[j] for j in range(0, len(chopped_positions))}  # Turn it in to a dict (Linear = bad?)
    return positions_dict