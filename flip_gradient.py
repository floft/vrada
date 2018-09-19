"""
Gradient reversal layer

Taken from: https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

class FlipGradientBuilder(object):
    """
    Usage:
        from flip_gradient import flip_gradient

        # Flip the gradient of y w.r.t. x and scale by l (defaults to 1.0)
        y = flip_gradient(x, l)
    """
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()