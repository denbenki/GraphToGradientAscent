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

i = 10

graph = connected_graph_generator.gnp_random_connected_graph(i, 1/i, 10)

a0 = time.time()
gradient_ascendor.graph_position_ascendor(graph)
b0 = time.time()
print(b0-a0)

a0 = time.time()
gradient_ascendor_backup.graph_position_ascendor(graph)
b0 = time.time()
print(b0-a0)

# node_time_list.append(b0-a0)
# node_amount_list.append(i)

# plt.plot(node_amount_list, node_time_list)
# plt.show()