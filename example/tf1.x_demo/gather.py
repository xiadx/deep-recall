#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf

x = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])

z = tf.gather(x, [1, 0], axis=1)

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(z))
