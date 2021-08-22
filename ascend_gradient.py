import numpy as np
import tensorflow as tf
import networkx as nx
import numpy
import random

#TODO: Give both an option for what algorithm to use when optimising AND what loss function to use.

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
            coordinates.append(tf.Variable(random.random()))  # Initializing the variables. Note the scaling factor
            seed += 1

        var_list.append(coordinates)  # Add to the array

    # Defining the optimiser and summing the edge weights

    optim = tf.keras.optimizers.SGD(learning_rate)

    sum_n = 0
    n_dict = {}

    # Sum together the weights (TODO: enable weighting)
    for e in con_tup_list:
        a = 1  # graph[e[0]][e[1]]["weight"]. That is: here we can set the weight when necessary
        sum_n += 2 * a

        # Symmetric dictionary
        n_dict[e] = a
        n_dict[e[1], e[0]] = a

    # Running the descent (in trying to get it to work, I'll create a linear var list)

    no_nodes = len(var_list)
    dist_dict = {}
    w_dict = {}

    lin_var_list = []
    for element in var_list:
        for element_ in element:
            lin_var_list.append(element_)

    # Adding the exceptions

    for i_3 in range(0, no_nodes):
        w_dict[(i_3, i_3)] = tf.constant(0.0)
        dist_dict[(i_3, i_3, 0)], dist_dict[(i_3, i_3, 1)], dist_dict[(i_3, i_3, 2)] \
            = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)

    for i in range(0, 200):  # From 0 to number of iterations
        # Initialising sum
        sum_w = 0.0

        # Sum together the "reaction rates" and calculate all distances
        for i_1 in range(0, no_nodes):
            for j_1 in range(0, i_1):
                b = tf.math.subtract(var_list[i_1], var_list[j_1])
                c = tf.exp(tf.negative(tf.tensordot(b, b, 1)))
                sum_w += 2 * c
                # Saving the reaction rate and the distance symmetrically

                dist_dict[(i_1, j_1, 0)], dist_dict[(i_1, j_1, 1)], dist_dict[(i_1, j_1, 2)] = b[0], b[1], b[2]

                # NB: Difference in polarity
                dist_dict[(j_1, i_1, 0)], dist_dict[(j_1, i_1, 1)], dist_dict[(j_1, i_1, 2)] = -b[0], -b[1], -b[2]

                w_dict[(i_1, j_1)], w_dict[(j_1, i_1)] = c, c  # Saving the reaction rates

        # Saving the quotient [How do I avoid this turning into inf???]
        quot = sum_n/sum_w

        # Defining the Grads (Start thinking about optimisation right away, there's probably a lot of things to improve)
        grad_list = []
        for i_2 in range(0, no_nodes):  # TODO: Might change the iternames
            for coord in range(0, 3):
                grad = 0.0
                for j_2 in range(0, no_nodes):
                        grad += -2 * n_dict.get((i_2, j_2), 0) * dist_dict[(i_2, j_2, coord)] + 2 * \
                                quot * dist_dict[(i_2, j_2, coord)] * w_dict[(i_2, j_2)]
                grad_list.append(tf.constant(grad))

        optim.apply_gradients(zip(grad_list, lin_var_list))  # Follows the gradients
        used_in_debugg = np.dot(grad_list, grad_list)
        print(used_in_debugg)


    return var_list