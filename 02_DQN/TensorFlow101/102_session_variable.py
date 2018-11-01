"""
本次主要体验tf对session和变量的处理思想
"""

import tensorflow as tf

state = tf.Variable(0,name='counter')

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # tf中一定记得，变量要初始化！这点新上手容易忘
    for _ in range(100):
        sess.run(update)
        print(sess.run(state))
