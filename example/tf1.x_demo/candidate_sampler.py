#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import tensorflow as tf

true_classes = tf.convert_to_tensor([
    [1, 2, 3, 4]
], dtype=tf.int64)

sampled_candidates, true_expected_count, sampled_expected_count = \
    tf.nn.uniform_candidate_sampler(
        true_classes=true_classes,
        num_true=4,
        num_sampled=2,
        unique=False,
        range_max=4
    )

init_local = tf.local_variables_initializer()
init_global = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run([init_local, init_global])
    print(sess.run(sampled_candidates))
    print(sess.run(true_expected_count))
    print(sess.run(sampled_expected_count))
