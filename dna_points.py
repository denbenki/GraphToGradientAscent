# Inspired from https://github.com/vlavorini/Numpy-VS-Tensorflow and https://towardsdatascience.com/numpy-vs-tensorflow-speed-on-matrix-calculations-9cbff6b3ce04
# His goals are different, but as a part of it he also wants to calculate the squared distances using only matrices and no for loops

# Also look at https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
# I think they do it even smarter


import tensorflow as tf
import numpy as np

# Things that can be precalculated before the iterations
n_nodes = 5
zero_diag = tf.zeros(n_nodes)
edges = np.ones((n_nodes, n_nodes), dtype=np.float32) # All edges equal weight here
sum_edges = tf.reduce_sum(edges)

# Make the nodes as a 3-by-n matrix
ptf = tf.Variable(tf.random.uniform((n_nodes, 3)))

# Some matrix/vector manipulation to calculate the squared differences
expd = tf.expand_dims(ptf,2)
tiled = tf.tile(expd, [1,1,tf.shape(ptf)[0]])
trans = tf.transpose(ptf)
sq_dists = tf.reduce_sum(tf.math.squared_difference(trans, tiled), 1)

# Also calculate just the distances
dists = tf.sqrt(sq_dists)

# Reaction rates
reaction_rates = tf.exp(tf.negative(sq_dists))
# Set self-reaction to 0
reaction_rates_filtered = tf.linalg.set_diag(reaction_rates, zero_diag)
# Sum them all and scale
sum_rate = tf.reduce_sum(reaction_rates_filtered)
scaling_factor = tf.divide(sum_edges, sum_rate)
relative_reaction_rates = tf.multiply(reaction_rates_filtered, scaling_factor)

# Now get the relative contributions to the likelihood
sum1 = tf.multiply(edges, dists)
sum2 = tf.multiply(relative_reaction_rates, dists)
# And the likelihood itself, as a vector, giving the loss per node
likelihood1 = tf.reduce_sum(sum2 - sum1, 0)
