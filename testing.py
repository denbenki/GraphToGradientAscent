import time

import Main_Pack
import random
import networkx as nx
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

from Main_Pack import ascend_gradient
from Main_Pack import connected_graph_generator
from Main_Pack import gradient_ascendor
from Main_Pack import gradient_ascendor_backup
from Main_Pack import plot_3D_graph

# node_time_list = []
# node_amount_list = []
# max_amount_of_nodes = 50
# for i in range(1, max_amount_of_nodes + 1):

i = 5

graph = connected_graph_generator.gnp_random_connected_graph(i, 1/i, 10)

# The dude below is the self-processed gradients

# a0 = time.time()
# pos_test_dict = gradient_ascendor.graph_position_ascendor(graph)
# b0 = time.time()
# print(b0-a0)
#
# # Testing the results of doing the gradients myself on the loss function. This is gon' get a little ugly
# def loss_function_var_list(var_list, con_tup_list):
#
#
#     # Creates a dictionary of exponential functions of the distances and adds these together
#
#     exp_neg_dist2 = {}
#     sum_w = 0
#
#     for i in range(0, len(var_list)):  # Is this the fastest way? Run some testing
#         for j in range(0, i):
#             exp_neg_dist2[(j, i)] = (tf.exp(tf.negative((tf.tensordot(tf.math.subtract(var_list[i], var_list[j]),
#                                               tf.math.subtract(var_list[i], var_list[j]), 1)))))
#             sum_w += 2 * exp_neg_dist2[(j, i)]  # Since we are only moving through the lower triangle: Add twice
#
#     # Time to create the likelihood-value (negative, since we are minimizing)
#
#     likelihood_neg = 0
#     for e in con_tup_list:
#         likelihood_neg -= (2 * tf.math.log(exp_neg_dist2[e] / sum_w))  # TODO: Add counts of barcodesz
#     return likelihood_neg
#
# # Creating a list to run the loss function
#
# variable_list = []
#
# for position in pos_test_dict.values():
#     var_list_sub = []
#     for coord in position:
#         var_list_sub.append(tf.Variable(coord))
#
#     variable_list.append(var_list_sub)
#
# print(-loss_function_var_list(variable_list, graph.edges))

a0 = time.time()
gradient_ascendor_backup.graph_position_ascendor(graph)
b0 = time.time()
print(b0-a0)

# node_time_list.append(b0-a0)
# node_amount_list.append(i)

# plt.plot(node_amount_list, node_time_list)
# plt.show()