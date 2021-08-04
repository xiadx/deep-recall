#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf

print(tf.get_variable_scope().name)
print(tf.get_variable_scope().reuse)

with tf.variable_scope("foo", reuse=tf.AUTO_REUSE) as foo_scope:
    print(foo_scope.name)
    print(foo_scope.reuse)
    v1 = tf.get_variable("v", [1])
    print(v1)
    v2 = tf.get_variable("v", [1])
    print(v2)
    print(v1 is v2)

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
