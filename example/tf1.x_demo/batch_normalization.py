#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf


x = tf.convert_to_tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=tf.float32)

bn = tf.layers.batch_normalization(x)

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(bn))
