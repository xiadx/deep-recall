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
    0,
    1
], dtype=tf.int32)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
regularizer = tf.contrib.layers.l2_regularizer(0.1)
loss += tf.contrib.layers.apply_regularization(regularizer)

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(loss))
