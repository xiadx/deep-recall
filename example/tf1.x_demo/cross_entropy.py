#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import sys
import tensorflow as tf


x = tf.constant([
    [1.0, 2.0, 3.0],
    [-1.0, -2.0,  -3.0]
])

w = tf.Variable(tf.zeros([3, 2]))
b = tf.Variable(tf.zeros([2]))
pred = tf.matmul(x, w) + b
l1 = tf.constant([
    [0, 1],
    [1, 0]
])
l2 = tf.constant([
    1,
    0
])

l1 = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=l1)
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=l2)

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(l1))
    print(sess.run(l2))
