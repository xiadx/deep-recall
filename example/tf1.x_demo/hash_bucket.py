#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf


x = tf.constant([
    ['abc', 'def', 'ghi'],
    ['xyz', 'uvw', 'rst']
])

z = tf.string_to_hash_bucket_strong(x, 10, [0, 0])

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(z))
