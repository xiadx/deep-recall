#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf


x = tf.convert_to_tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=tf.float32)

w = tf.Variable(tf.zeros([3, 2]))
b = tf.Variable(tf.zeros([2]))

pred = tf.matmul(x, w) + b
label = tf.convert_to_tensor([
    [1, 0],
    [0, 1]
], dtype=tf.int32)

loss = tf.losses.sigmoid_cross_entropy(label, pred)

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(loss))
