#!/usr/bin/python
# coding=utf-8

# python3 predict_cancer_type.py -e 1000

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from helpers import *
from primitives import *
from sklearn import model_selection
import os
import numpy as np
from get_annotation_for_mutation_regions import *
import pandas as pd
import glob
from training_set import *

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

FLAGS = None

if os.path.isdir("/home/yulia/"):
	DIR = "/home/yulia/mnt/"
else:
	DIR = "/Users/yulia/Documents/mutational_signatures/"

DIR_DATA = "/mnt/raisin/yulia/pwgs/samples/new_bic2/vaf_corrected_cn_sampled_beta/results_onlyKnownSignatures_pcawgSigsBeta2_vaf_corrected_cn_sampled_beta/"
model_save_path = "trained_models/model.predict_cancer_type.ckpt"

#session_conf = tf.ConfigProto(gpu_options=gpu_options)
session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)

def make_model(inputs, labels, model_save_path, adam_rate = 1e-3):
	x_dim = inputs.shape[1]
	y_dim = labels.shape[1]

	x = tf.placeholder(tf.float32, [None, inputs.shape[1]])
	y_ = tf.placeholder(tf.float32, [None, y_dim])

	nn = init_neural_net_params([x_dim, 1000, 1000, 1000, y_dim])
	log_y_prediction = neural_net(x, nn['weights'], nn['biases'])

	with tf.name_scope('loss'):
		cross_entropy_all = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=log_y_prediction) # neural net
	cross_entropy = tf.reduce_mean(cross_entropy_all)

	predictions = tf.nn.softmax(log_y_prediction)
	with tf.name_scope('accuracy'):
		correct_prediction = correct_predictions_multiclass(predictions, y_)
	accuracy = tf.reduce_mean(correct_prediction)

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(adam_rate).minimize(cross_entropy)

	save_graph()
	saver = tf.train.Saver()

	return x, y_, predictions, correct_prediction, cross_entropy, accuracy, train_step, saver

def train(tf_vars, n_epochs, batch_size, model_save_path, train_data, test_data):
	x, y_, predictions, cross_entropy, accuracy, train_step = tf_vars

	x_train, y_train  = train_data
	x_test, y_test  = test_data
	train_dict = {x: x_train, y_: y_train}
	test_dict = {x: x_test, y_: y_test}

	print("Optimizing...")
	with tf.Session(config=session_conf) as sess:
		sess.run(tf.global_variables_initializer())
		for j in range(n_epochs):
			for i in range(x_train.shape[0] //batch_size+1):
				x_batch, y_batch = get_batch(train_data, batch_size, i)
				batch_dict = {x: x_batch, y_: y_batch}

				print('Epoch %d.%d: train CE %g; test CE %g; train ACC %g; test ACC %g' 
					% (j, i, cross_entropy.eval(feed_dict=train_dict), cross_entropy.eval(feed_dict=test_dict), 
						accuracy.eval(feed_dict=train_dict), accuracy.eval(feed_dict=test_dict) ))
				
				train_step.run(feed_dict=batch_dict)

			print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))
	
		save_path = saver.save(sess, model_save_path)
		print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train model to predict probability for region-mutation pair')
	parser.add_argument('--test', help='test mode: only read trained params and analyze', action="store_true")
	parser.add_argument('-e','--epochs', help='number of epochs', default=100)
	parser.add_argument('-b','--batch', help='batch size', default=10000)
	parser.add_argument('-a', '--adam', help='Rate for adam optimizer', default=1e-4,type=float)

	args = parser.parse_args()
	test_mode = args.test
	n_epochs = int(args.epochs)
	batch_size = int(args.batch)
	adam_rate = args.adam

	inputs = np.array(read_csv(DIR_DATA + "overall_distribution_per_tumor.csv")).astype(float)
	tumour_types = read_csv(DIR_DATA + "unique_tumortypes.csv")
	labels = np.array(read_csv(DIR_DATA + "tumortype_labels.csv")).astype(int)
	labels = get_one_hot_encoding(labels.ravel())

	tf.reset_default_graph()
	os.makedirs(model_save_path, exist_ok=True)

	x, y_, predictions, correct_prediction, cross_entropy, accuracy, train_step, saver = make_model(inputs, labels, model_save_path, adam_rate = adam_rate)

	# Split dataset into train / test
	x_train, x_test, y_train, y_test = model_selection.train_test_split(inputs, labels, test_size=0.2, random_state = 1991)
	test_dict = {x: x_test, y_: y_test}
	train_data = [x_train, y_train]
	test_data = [x_test, y_test]

	if not test_mode:
		train([x, y_, predictions, cross_entropy, accuracy, train_step], n_epochs, batch_size, model_save_path, train_data, test_data)
	else:
		if not os.path.exists(model_save_path):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)

			print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))

			print("Mean prediction within a class")
			pred = predictions.eval(feed_dict=test_dict)
			for i, name in enumerate(tumour_types):
				print(name[0] + ": " + str(np.mean(pred[:,i][np.where(y_test[:,i])])))

			print("\n")
			correct_prediction = correct_predictions_multiclass(predictions, y_).eval(feed_dict=test_dict)
			print("Mean accuracy within a class")
			for i, name in enumerate(tumour_types):
				print(name[0] + ": " + str(np.mean(correct_prediction[np.where(y_test[:,i])])))
			