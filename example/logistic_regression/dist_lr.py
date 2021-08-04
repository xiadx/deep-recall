#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import datetime
from tensorflow.python.platform import tf_logging as logging


flags = tf.app.flags
flags.DEFINE_string('files', '', 'local file list for samples')
flags.DEFINE_string('ps_hosts', '', 'One of ps, worker')
flags.DEFINE_string('worker_hosts', '', 'One of ps, worker')
flags.DEFINE_integer('task_index', None, 'Worker or server index')
flags.DEFINE_string('job_name', None, 'worker/ps')
flags.DEFINE_string('checkpointDir', None, 'oss checkpointDir')

FLAGS = tf.app.flags.FLAGS


# define the input
def input_fn(slice_count,  slice_id):
    print('slice_count:%d, slice_id:%d' % (slice_count,  slice_id))
    file_queue = tf.train.string_input_producer([FLAGS.files], num_epochs=2)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(file_queue, num_records=128)
    batch_values = tf.train.batch([value], batch_size=128, capacity=64000, enqueue_many=True)
    v1, v2, v3, v4 = tf.decode_csv(batch_values, record_defaults=[[1.0]] * 4, field_delim=',')
    labels = tf.reshape(tf.cast(v4, tf.int32), [128])
    features = tf.stack([v1, v2, v3], axis=1)

    return features, labels


# construct the model
def model_fn(features, labels, global_step):
    W = tf.Variable(tf.zeros([3, 2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.matmul(features, W) + b

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels))

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step)

    return loss, optimizer


def train(worker_count, task_index, cluster, is_chief, target):
    worker_device = '/job:worker/task:%d/cpu:%d' % (task_index, 0)
    print 'worker_device = %s' % worker_device

    # assign io related variable and ops to local worder device
    with tf.device(worker_device):
        features, labels = input_fn(slice_count=worker_count, slice_id=task_index)

    available_worker_device = '/job:worker/task:%d/cpu:%d' % (task_index, 0)
    # assign global variables to ps node
    with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # construct the model structure
        loss, optimizer = model_fn(features, labels, global_step)
        print 'start training'
    sv = tf.train.Supervisor(is_chief=is_chief, global_step=global_step)
    with sv.managed_session(target) as sess:
        for step in range(100000):
            if not sv.should_stop():
                _, c, g = sess.run([optimizer, loss, global_step])
                if step % 2000 == 0:
                    print '[%s] step %d, global_step %d, loss is %f' % (datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S'), step, g, c)
    print 'training finished'


def main(unused_argv):
    print 'job name = %s' % FLAGS.job_name
    print 'task index = %d' % FLAGS.task_index
    is_chief = FLAGS.task_index == 0

    # construct the servers
    ps_spec = FLAGS.ps_hosts.split(',')
    worder_spec = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worder_spec})
    worker_count = len(worder_spec)

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    # join the ps server
    if FLAGS.job_name == 'ps':
        server.join()

    # start  the training
    try:
        train(worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief, target=server.target)
    except Exception, e:
        print 'catch a exception: %s' % e.message


if __name__ == '__main__':
    tf.app.run()
