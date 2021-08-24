import numpy as np
import tensorflow as tf
import networkx as nx
import numpy
import random

#TODO: Give both an option for what algorithm to use when optimising AND what loss function to use.

# The below is out-commented in order to try precalculating the gradients.

# Hard-coding a function for testing the algorithm.
def loss_function_pos_mat(pos_mat, adj_mat, nu_nodes):
    # Manipulating the arrays to create symmetric, square distance array

    expd = tf.expand_dims(pos_mat, 2)  # Give "depth"
    tiled = tf.tile(expd, [1, 1, nu_nodes])
    trans = tf.transpose(pos_mat)
    sq_dists = tf.reduce_sum(tf.math.squared_difference(trans, tiled), 1)

    # Creating log frequency reaction rate array (proportional)
    reac_mat = tf.exp(tf.negative(sq_dists))
    sum_reac = tf.reduce_sum(reac_mat)
    log_freq_reac = tf.math.log(tf.divide(reac_mat, sum_reac))

    # Matrix to be summed over to create likelihood
    like_mat = tf.math.multiply(adj_mat, log_freq_reac)
    return tf.negative(tf.reduce_sum(like_mat))

def optimize_positions(pos_mat, adj_mat, nu_nodes, learning_rate):

    """
    :param graph: A networkx graph
    :param learning_rate: Step size
    :return: positions minimizing a hard coded loss function.
    """
    # Defining the optimiser and loss function

    optim = tf.keras.optimizers.Adam(learning_rate)  # Arg = learning rate
    loss_function = lambda: loss_function_pos_mat(pos_mat, adj_mat, nu_nodes)

    # Running the optimization. Each call for the "minimize" method is one step.
    for i in range(0, 2000):
        optim.minimize(loss_function, [pos_mat])
        print(-loss_function_pos_mat(pos_mat, adj_mat, nu_nodes))

    positions = tf.reshape(pos_mat, [tf.size(pos_mat)])

    return positions