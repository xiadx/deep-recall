#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf


tf.enable_eager_execution()

x = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])

z = tf.reshape(x, [-1, 6])

tf.print('z:', z)

print(z)
