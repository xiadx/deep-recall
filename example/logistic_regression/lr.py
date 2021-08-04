#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import tensorflow as tf


job_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sample_file = os.path.join(job_path, 'data/sample.csv')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('files', sample_file, 'local file list for samples')


# define the input
def input_fn():
    file_queue = tf.train.string_input_producer([FLAGS.files])
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(file_queue, num_records=128)
    batch_values = tf.train.batch([value], batch_size=128, capacity=64000, enqueue_many=True)
    v1, v2, v3, v4 = tf.decode_csv(batch_values, record_defaults=[[1.0]] * 4, field_delim=',')
    labels = tf.reshape(tf.cast(v4, tf.int32), [128])
    features = tf.stack([v1, v2, v3], axis=1)
    return features, labels


# construct the model
def model_fn(features, labels):
    W = tf.Variable(tf.zeros([3, 2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.matmul(features, W) + b

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels))

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    return loss, optimizer


features, labels = input_fn()
loss, optimizer = model_fn(features, labels)

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

sess = tf.Session()
sess.run([global_init, local_init])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord=coord)

for step in range(10000):
    _, c = sess.run([optimizer, loss])
    if step % 2000 == 0:
        print 'loss,', c
