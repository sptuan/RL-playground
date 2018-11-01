"""
104中我们来定义一个层
"""

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size,out_size,activation_function = None)
    Weight = tf.Variable(tf.random_normal([[in_size,out_size]]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.natmul(inputs, Weight) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else
        outputs = activation_function(Wx_plus_b)

    return outputs


x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
perdition = add_layer(l1, 10, 1, activation_function = None)


