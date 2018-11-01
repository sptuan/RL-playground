"""
TensorFlow101
之前使用过Darknet和Caffe，TF还是第一次碰，尤其是最近TF官方支持了Raspbian，我先来学习这个框架的基本操作。
"""

import tensorflow as tf
import numpy as np

# 创建一组数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### 创建结构 ###
# W是个矩阵形式
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)    # 激活init非常重要

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
