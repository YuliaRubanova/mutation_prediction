#!/usr/bin/python
# coding=utf-8

# python3 predict_mut_type_from_chrom_features.py -n 50 -rs 10 -a 1e-3

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
from generate_data_mutations_only import generate_random_mutations

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

FLAGS = None

if os.path.isdir("/home/yulia/"):
	DIR = "/home/yulia/mnt/"
else:
	DIR = "/Users/yulia/Documents/mutational_signatures/"

DEF_FEATURE_PATH = DIR + "dna_features_ryoga/"
#dataset_path = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
#mut_dataset_path = [DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part*.pickle"]
mut_dataset_path = [DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part*.pickle"]
file_name = os.path.basename(__file__)[:-3]
model_dir = "trained_models/model." + file_name + ".tumours{}.mut{}/"
dataset_with_annotation = DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.region_size{region_size}.ID{{id}}.over_time.annotation.hdf5"

session_conf = tf.ConfigProto(gpu_options=gpu_options)
# session_conf = tf.ConfigProto(
#     device_count={'CPU' : 1, 'GPU' : 0},
#     allow_soft_placement=True,
#     log_device_placement=False
# )

def predict_mutation_rate_model_nn(X, y_, x_tumour_ids, tumour_latents, reg_latent_dim):
	x_dim = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value
	z_latent_dim = reg_latent_dim*2
	n_types = y_.get_shape()[1].value

	with tf.name_scope('region_latents'):
		prob_nn = init_neural_net_params([x_dim, 500, 500, 700, reg_latent_dim])
		y_region = neural_net(X, prob_nn['weights'], prob_nn['biases'])

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, x_tumour_ids))

	#prob_nn = init_neural_net_params([x_dim + z_latent_dim] + [200, 200, n_types])
	prob_nn = init_neural_net_params([x_dim] + [200, 200, n_types])

	#log_y_prediction = neural_net(tf.concat([X, z_t], 1), prob_nn['weights'], prob_nn['biases'])
	log_y_prediction = neural_net(X, prob_nn['weights'], prob_nn['biases'])

	with tf.name_scope('loss'):
		cross_entropy_all = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=log_y_prediction) # neural net
	cross_entropy = tf.reduce_mean(cross_entropy_all)

	predictions = tf.nn.softmax(log_y_prediction)
	with tf.name_scope('accuracy'):
		correct_prediction = correct_predictions_multiclass(predictions, y_)
	accuracy = tf.reduce_mean(correct_prediction)

	return log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions

def make_model(n_region_features, n_mut_types, n_unique_tumours, z_latent_dim, model_type, model_save_path, adam_rate = 1e-3):
	x = tf.placeholder(tf.float32, [None, n_region_features])
	x_tumour_ids = tf.placeholder(tf.int32, [None, 1])
	y_ = tf.placeholder(tf.float32, [None, n_mut_types])

	initial_tumour_latents = tf.abs(tf.concat([weight_variable([z_latent_dim//2,1]), weight_variable([z_latent_dim//2,1], mean=1)], axis = 0))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions =  predict_mutation_rate_model_nn(x, y_, x_tumour_ids, tumour_latents, reg_latent_dim)

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(adam_rate).minimize(cross_entropy)

	save_graph()
	saver = tf.train.Saver()

	tf_vars = [x, x_tumour_ids, y_, log_y_prediction]
	metrics = [cross_entropy, accuracy]
	meta = [train_step, saver]
	extra = [z_t, y_region, predictions]

	if model_type == "gaussian":
		extra.append(L)

	return tf_vars, metrics, meta, extra

def train(tf_vars, n_epochs, batch_size, model_save_path, train_data, test_data):
	x, x_tumour_ids, y_, log_y_prediction = tf_vars

	x_train, y_train, x_tumour_ids_train  = train_data
	x_test, y_test, x_tumour_ids_test  = test_data

	print("Optimizing...")
	# config=tf.ConfigProto(gpu_options=gpu_options)
	# config=session_conf
	with tf.Session(config=session_conf) as sess:
		sess.run(tf.global_variables_initializer())
		for j in range(n_epochs):
			for i in range(x_train.shape[0] //batch_size+1):
				x_batch, y_batch, x_tumour_ids_batch = get_batch(train_data, batch_size, i)
				batch_dict = {x: x_batch, y_: y_batch, x_tumour_ids: x_tumour_ids_batch}
				# print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
				# print('train accuracy %g' % accuracy.eval(feed_dict=train_dict))
				# print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))
				#print((tf.sigmoid(log_y_prediction)).eval(feed_dict=test_dict).ravel()[:10]) #neural net
				#print(y_.eval(feed_dict=test_dict).ravel()[:10])
				#print(L.eval(feed_dict=test_dict).ravel()[:10])
				#print((log_y_prediction).eval(feed_dict=test_dict).ravel()[:10])
				#print(z_t.eval(feed_dict=batch_dict)[0,:10])
				train_step.run(feed_dict=batch_dict)

				train_cross_entropy = cross_entropy.eval(feed_dict=train_dict)
				print('Epoch %d.%d: train CE %g; test CE %g; train ACC %g; test ACC %g' 
					% (j, i, train_cross_entropy, cross_entropy.eval(feed_dict=test_dict), 
						accuracy.eval(feed_dict=train_dict), accuracy.eval(feed_dict=test_dict) ))
				
			if j % 5 == 0:
				model_save_path_tmp = "trained_models/tmp/model.region_dataset.model{}.tumours{}.mut{}.ckpt".format(model_type, n_tumours, n_mut)
				save_path = saver.save(sess, model_save_path_tmp)
				print("Model saved in file: %s" % save_path)

			print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))
	
		save_path = saver.save(sess, model_save_path)
		print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train model to predict probability for region-mutation pair')
	parser.add_argument('--test', help='test mode: only read trained params and analyze', action="store_true")
	#parser.add_argument('--latents', help='number of latent dimensions', default=100, type=int)
	parser.add_argument('-n', '--tumours', help='number of tumours to include in the set', default=None)
	parser.add_argument('-m', '--mut', help='number of mutations per tumour', default=None)
	parser.add_argument('-e','--epochs', help='number of epochs', default=100)
	parser.add_argument('-b','--batch', help='batch size', default=10000)
	parser.add_argument('--model', help='Model type: gaussian likelihood or neural net', default='nn')
	#parser.add_argument('--loss', help = "loss type: poisson or mean_squared", default="poisson")
	parser.add_argument('-f', '--feature-path', help='feature file path', default=DEF_FEATURE_PATH)
	parser.add_argument('-rs', '--region-size', help='size of training regions surrounding a mutation', default=100,type=int)
	parser.add_argument('-a', '--adam', help='Rate for adam optimizer', default=1e-3,type=float)
	parser.add_argument('--train-garbage', help='Make garbage features to test if region features have some signal', action="store_true")

	args = parser.parse_args()
	test_mode = args.test
	n_tumours = args.tumours
	n_mut = args.mut
	n_epochs = int(args.epochs)
	batch_size = int(args.batch)
	model_type = args.model
	feature_path = args.feature_path
	region_size = args.region_size
	adam_rate = args.adam
	train_garbage = args.train_garbage
	#latent_dimension = args.latents

	print("Fitting model version: " + model_type)
	# model params
	reg_latent_dim = 100
	z_latent_dim = reg_latent_dim * 2

	n_tumours_per_batch = 40 # tumours in tumour_batch

	mut_features, unique_tumours, n_tumours, n_mut, available_tumours, num_features, n_unique_tumours = \
		load_filter_dataset(mut_dataset_path, feature_path, dataset_with_annotation, region_size, n_tumours, n_mut)

	print("Processing {} mutations from {} tumour(s) ...".format(n_mut, n_unique_tumours))

	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours)












	training_set, labels = make_set_for_predicting_mut_rate(training_set, labels, mut_annotation, feature_names)
	n_region_features = training_set.shape[1]
	n_mut_types = labels.shape[1]

	print(training_set.shape)
	print(labels.shape)

	model_dir, model_save_path = prepare_model_dir(sys.argv, model_dir, __file__, [n_tumours, n_mut])

	tf_vars, metrics, meta, extra = make_model(n_region_features, n_mut_types, n_unique_tumours, z_latent_dim, model_type, model_save_path, adam_rate = adam_rate)

	x, x_tumour_ids, y_, log_y_prediction = tf_vars
	cross_entropy, accuracy = metrics
	train_step, saver = meta
	z_t, y_region, predictions = extra
	
	# Split dataset into train / test
	x_train, x_test, y_train, y_test, x_tumour_ids_train, x_tumour_ids_test = train_test_split([training_set, labels, tumour_ids], split_by = mut_vaf, test_size=0.2)
	train_dict = {x: x_train, y_: y_train, x_tumour_ids: x_tumour_ids_train}
	test_dict = {x: x_test, y_: y_test, x_tumour_ids: x_tumour_ids_test}
	train_data = [x_train, y_train, x_tumour_ids_train]
	test_data = [x_test, y_test, x_tumour_ids_test]

	if not test_mode:
		train(tf_vars, n_epochs, batch_size, model_save_path, train_data, test_data)
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
			for i, name in enumerate(feature_names[:96]):
				print(name + ": " + str(np.mean(pred[y_test[:,i].ravel().astype(int)])))

			correct_prediction = correct_predictions_multiclass(predictions, y_).eval(feed_dict=test_dict)

			print("Mean accuracy within a class")
			for i, name in enumerate(feature_names[:96]):
				print(name + ": " + str(np.mean(correct_prediction[y_test[:,i].ravel().astype(int)])))
			
			print("Tumour representation std")
			print(np.mean(np.std(z_t.eval(feed_dict=test_dict), axis=0)))

			unique_tumours = np.unique(np.squeeze(x_tumour_ids_test))

			for tum in unique_tumours:
				ind = np.squeeze(x_tumour_ids_test) == tum
				x_cur = x_test[ind]
				y_cur = y_test[ind]
				tumour_ids = x_tumour_ids_test[ind]
				n_mutations = sum(ind)
				tum_dict = {x: x_cur, y_: y_cur, x_tumour_ids: tumour_ids}

				print('Tumour %d: #mut %d cross_entropy %g  accuracy %g' % (tum, n_mutations, cross_entropy.eval(feed_dict=tum_dict), accuracy.eval(feed_dict=tum_dict)))
				