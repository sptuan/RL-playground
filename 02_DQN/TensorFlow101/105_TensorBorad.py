"""
105 TensorBoard Practice

"""

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope("layer"):
        # weights
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")

        # biases
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")

        # Wx_plus_b
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 使用input图层包含起来
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_in")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_in")

# 建层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# 误差计算，两者平方取平均
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

# 训练参数
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 开始初始化
init = tf.global_variables_initializer()

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

# 开始训练
for i in range(10000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # 输出误差
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
