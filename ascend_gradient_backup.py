import numpy as np
import tensorflow as tf
import networkx as nx
import numpy
import random

#TODO: Give both an option for what algorithm to use when optimising AND what loss function to use.

# The below is out-commented in order to try precalculating the gradients.

# Hard-coding a function for testing the algorithm.
def loss_function_var_list(var_list, con_tup_list):


    # Creates a dictionary of exponential functions of the distances and adds these together

    exp_neg_dist2 = {}
    sum_w = 0

    for i in range(0, len(var_list)):  # Is this the fastest way? Run some testing
        for j in range(0, i):
            exp_neg_dist2[(j, i)] = (tf.exp(tf.negative((tf.tensordot(tf.math.subtract(var_list[i], var_list[j]),
                                              tf.math.subtract(var_list[i], var_list[j]), 1)))))
            sum_w += 2 * exp_neg_dist2[(j, i)]  # Since we are only moving through the lower triangle: Add twice

    # Time to create the likelihood-value (negative, since we are minimizing)

    likelihood_neg = 0
    for e in con_tup_list:
        likelihood_neg -= (2 * tf.math.log(exp_neg_dist2[e] / sum_w))  # TODO: Add counts of barcodesz
    return likelihood_neg

def optimize_positions(graph, learning_rate):

    """
    :param graph: A networkx graph
    :param learning_rate: Step size
    :return: positions minimizing a hard coded loss function.
    """

    # Let's start with getting a way of creating the variables necessary.
    G = numpy.asarray(nx.to_numpy_matrix(graph))  # To handle the graph smoothly
    con_tup_list = graph.edges  # To handle the connections

    # Initialisation of positions. When testing, we'll find a way to seed this from outside the function.
    var_list = []
    seed = 0
    for i in range(0, G.shape[0]):
        coordinates = []  # 3D-coordinates

        for j in range(0, 3):
            random.seed(seed)
            coordinates.append(tf.Variable(5 * random.random()))  # Initializing the variables.
            seed += 1
        var_list.append(coordinates)  # Add to the array

    # Defining the optimiser and loss function

    optim = tf.keras.optimizers.Adam(learning_rate)  # Arg = learning rate
    loss_function = lambda: loss_function_var_list(var_list, con_tup_list)

    # Running the optimization. Each call for the "minimize" method is one step.
    for i in range(0, 200):
        optim.minimize(loss_function, var_list)
        print(-loss_function_var_list(var_list, con_tup_list))


    return var_list