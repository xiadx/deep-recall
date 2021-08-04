#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datetime
from tensorflow.python.platform import tf_logging as logging
import os
import json


job_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sample_file = os.path.join(job_path, 'data/sample.csv')
model_dir = os.path.join(job_path, 'data/model')

flags = tf.app.flags
tf.app.flags.DEFINE_string('files', sample_file, 'local file list for samples')
tf.app.flags.DEFINE_string("ps_hosts", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("worker_hosts", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", None, "Worker or server index")
flags.DEFINE_string("job_name", None, "worker/ps")
flags.DEFINE_string("checkpointDir", model_dir, "oss checkpointDir")
FLAGS = tf.app.flags.FLAGS


def model_fn(features, labels, mode):
    W = tf.Variable(tf.zeros([3, 2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.matmul(features, W) + b

    loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.GradientDescentOptimizer(0.05)
        train_op = opt.minimize(loss, global_step=global_step, name='train_op')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss)
    else:
        raise ValueError(
            "Only TRAIN and EVAL modes are supported: %s" % (mode))


def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    v1, v2, v3, v4 = tf.decode_csv(line, record_defaults=[[1.0]] * 4, field_delim=',')
    labels = tf.cast(v4, tf.int32)
    features = tf.stack([v1, v2, v3])
    return features, labels


def train_input_fn():
    dataset = tf.data.TextLineDataset(FLAGS.files.split(","))
    d = dataset.map(decode_line).shuffle(True).batch(128).repeat()
    return d


def eval_input_fn():
    dataset = tf.data.TextLineDataset(FLAGS.files.split(","))
    d = dataset.map(decode_line).batch(128)
    return d


def main(unused_argv):
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.estimator.RunConfig(model_dir=FLAGS.checkpointDir,
                                    session_config=sess_config,
                                    save_checkpoints_steps=100,
                                    save_summary_steps=100)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        model_dir=FLAGS.checkpointDir)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=6, throttle_secs=1)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("done")


if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    env_dist = os.environ
    print(env_dist.get('TF_CONFIG'))
    tf.app.run()