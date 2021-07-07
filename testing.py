import Main_Pack
import random

from Main_Pack import ascend_gradient
from Main_Pack import connected_graph_generator
from Main_Pack import gradient_ascendor
from Main_Pack import plot_3D_graph

graph = connected_graph_generator.gnp_random_connected_graph(100, 0.01, 10)
gradient_ascendor.graph_position_ascendor(graph)