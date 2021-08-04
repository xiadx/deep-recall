#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf


x = [tf.constant([['a', 'b'], ['c', 'd']]), tf.constant([['e', 'f'], ['g', 'h']])]
z = tf.string_join(x, ',')

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(z))
