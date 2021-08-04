#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf


labels = tf.convert_to_tensor([1.0, 0.0, 1.0], dtype=tf.float32)
predicts = tf.convert_to_tensor([10.0, -20.0, 20.0], dtype=tf.float32)
loss = tf.losses.log_loss(labels, tf.nn.sigmoid(predicts))

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(loss))
