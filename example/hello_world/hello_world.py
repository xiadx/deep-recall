#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import tensorflow as tf


job_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sample_file = os.path.join(job_path, 'data/sample.csv')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('files', sample_file, 'local file list for samples')


def main():
    print tf.__version__
    print job_path
    print sample_file
    print FLAGS.files


if __name__ == '__main__':
    main()
