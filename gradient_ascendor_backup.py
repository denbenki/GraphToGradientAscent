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
    adj_mat = tf.constant(numpy.asarray(nx.to_numpy_matrix(graph, dtype= numpy.float32)))  # adjacency matrix of graph; current edge weight = 1. TODO: CHeck if this is a good idea
    nu_nodes = tf.size(adj_mat[0])

    # Initialisation of positions. Should be seeded. (Problem: not a variable? Let's mekk around; let's try doing matrix of variables).
    pos_mat = tf.random.uniform((nu_nodes, 3))

    # Runs a function for optimizing the positions.

    positions = optimize_positions(pos_mat, adj_mat, nu_nodes, learning_rate)


    positions_dict = {j: positions[j] for j in range(0, len(positions))}  # Turn it in to a dict
    return positions_dict