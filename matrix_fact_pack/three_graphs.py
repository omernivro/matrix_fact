import tensorflow as tf
import itertools
import numpy as np
import scipy.io
import csv
import os.path
import decimal
import time
import fnmatch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():
    train_iterator = ...
    train_model = BuildTrainModel(train_iterator)
    initializer = tf.global_variables_initializer()

with eval_graph.as_default():
    eval_iterator = ...
    eval_model = BuildEvalModel(eval_iterator)

with infer_graph.as_default():
    infer_iterator, infer_inputs = ...
    infer_model = BuildInferenceModel(infer_iterator)

checkpoints_path = "/model/checkpoints"

train_sess = tf.Session(graph=train_graph)
eval_sess = tf.Session(graph=eval_graph)
infer_sess = tf.Session(graph=infer_graph)

train_sess.run(initializer)
train_sess.run(train_iterator.initializer)

for i in itertools.count():

    train_model.train(train_sess)

    if i % EVAL_STEPS == 0:
        checkpoint_path = train_model.saver.save(
            train_sess, checkpoints_path, global_step=i)
        eval_model.saver.restore(eval_sess, checkpoint_path)
        eval_sess.run(eval_iterator.initializer)
        while data_to_eval:
            eval_model.eval(eval_sess)

    if i % INFER_STEPS == 0:
        checkpoint_path = train_model.saver.save(
            train_sess, checkpoints_path, global_step=i)
        infer_model.saver.restore(infer_sess, checkpoint_path)
        infer_sess.run(infer_iterator.initializer, feed_dict={
                       infer_inputs: infer_input_data})
        while data_to_infer:
            infer_model.infer(infer_sess)
