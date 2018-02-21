#!/usr/bin/python
# coding=utf-8

# Predict a binary label -- if it is a true or false mutation -- depending on chromain features and mutation type


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
from plot_signatures import *

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
model_save_path = "trained_models/model.region_dataset.model{}.tumours{}.mut{}/model.train_test_vaf.ckpt"
dataset_with_annotation = DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.region_size{region_size}.ID{{id}}.over_time.annotation.hdf5"

#session_conf = tf.ConfigProto(gpu_options=gpu_options)
session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)

def predict_true_false_mut_model_gaussian(X, y_, x_tumour_ids, tumour_latents, reg_latent_dim):
	x_dim = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value
	n_tumours = tumour_latents.get_shape()[0].value
	tumour_latent_dim = tumour_latents.get_shape()[1].value

	with tf.name_scope('region_latents'):
		prob_nn = init_neural_net_params([x_dim, 500, 200, reg_latent_dim])
		y_region = neural_net(X, prob_nn['weights'], prob_nn['biases'])

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, x_tumour_ids))

	m_t, sigma_t = unpack_gaussian(z_t, reg_latent_dim)

	dist = tf.contrib.distributions.MultivariateNormalDiag(m_t, sigma_t)

	threshold = bias_variable([1], value = -0.919*reg_latent_dim)

	L = dist.log_prob(y_region) - threshold

	# how to normalize across other regions and types? !!!!
	#p = tf.minimum(L,0) #!!!!!!! 

	log_normalizer = weight_variable([n_tumours])
	normalize_per_sample = tf.gather_nd(log_normalizer, x_tumour_ids)

	#p = tf.minimum(L,-1e-6)#!!!!
	p = tf.minimum(L,-1e-6)

	log_y_prediction = tf.reshape(p, [-1,1])

	# make a different loss !!!! the loss is on the counts
	with tf.name_scope('loss'):
		cross_entropy_all = -tf.reduce_sum(y_ * log_y_prediction, reduction_indices=[1]) # gaussian
	cross_entropy = tf.reduce_mean(cross_entropy_all)

	with tf.name_scope('accuracy'):
		predictions = tf.exp(log_y_prediction)
		correct_prediction = correct_predictions(predictions, y_)
	accuracy = tf.reduce_mean(correct_prediction)

	return log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions, L, normalize_per_sample


def predict_true_false_mut_model_nn(X, y_, x_tumour_ids, tumour_latents, reg_latent_dim):
	x_dim = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value
	z_latent_dim = reg_latent_dim*2

	with tf.name_scope('region_latents'):
		prob_nn = init_neural_net_params([x_dim, 500, 500, 700, reg_latent_dim])
		y_region = neural_net(X, prob_nn['weights'], prob_nn['biases'])

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, x_tumour_ids))

	#prob_nn = init_neural_net_params([x_dim + z_latent_dim] + [200, 200, 1])
	prob_nn = init_neural_net_params([x_dim] + [200, 200, 1])

	#p = neural_net(tf.concat([X, z_t], 1), prob_nn['weights'], prob_nn['biases'])
	p = neural_net(tf.concat([X], 1), prob_nn['weights'], prob_nn['biases'])

	log_y_prediction = tf.reshape(p, [-1,1])

	# make a different loss !!!! the loss is on the counts
	with tf.name_scope('loss'):
		cross_entropy_all = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=log_y_prediction) # neural net
	cross_entropy = tf.reduce_mean(cross_entropy_all)

	with tf.name_scope('accuracy'):
		predictions = tf.sigmoid(log_y_prediction)
		correct_prediction = correct_predictions(predictions, y_)
	accuracy = tf.reduce_mean(correct_prediction)

	return log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions

def make_model(x_dim, n_unique_tumours, z_latent_dim, model_type, model_save_path, adam_rate = 1e-3):
	x = tf.placeholder(tf.float32, [None, x_dim])
	x_tumour_ids = tf.placeholder(tf.int32, [None, 1])
	y_ = tf.placeholder(tf.float32, [None, 1])

	initial_tumour_latents = tf.abs(tf.concat([weight_variable([z_latent_dim//2,1]), weight_variable([z_latent_dim//2,1], mean=1)], axis = 0))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	if model_type == "gaussian":
		log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions, L, normalize_per_sample =  predict_true_false_mut_model_gaussian(x, y_, x_tumour_ids, tumour_latents, reg_latent_dim)
	else:
		log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions =  predict_true_false_mut_model_nn(x, y_, x_tumour_ids, tumour_latents, reg_latent_dim)

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

def train(tf_vars, n_epochs, batch_size, model_save_path, available_tumours):
	x, x_tumour_ids, y_, log_y_prediction = tf_vars

	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours)
	print("Optimizing...")
	# config=tf.ConfigProto(gpu_options=gpu_options)
	# config=session_conf
	with tf.Session(config=session_conf) as sess:
		sess.run(tf.global_variables_initializer())
		for j in range(n_epochs):
			for k in range(len(available_tumours) // n_tumours_per_batch+1):
				if n_tumours_per_batch < len(available_tumours):
					training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours, n_tumours_per_batch, k)
					if training_set is None:
						continue # no samples in a batch

				# Split dataset into train / test
				x_train, x_test, y_train, y_test, x_tumour_ids_train, x_tumour_ids_test = train_test_split([training_set, labels, tumour_ids], split_by = mut_vaf, test_size=0.2)
				train_dict = {x: x_train, y_: y_train, x_tumour_ids: x_tumour_ids_train}
				test_dict = {x: x_test, y_: y_test, x_tumour_ids: x_tumour_ids_test}
				train_data = [x_train, y_train, x_tumour_ids_train]

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
					% (j, k, train_cross_entropy, cross_entropy.eval(feed_dict=test_dict), 
						accuracy.eval(feed_dict=train_dict), accuracy.eval(feed_dict=test_dict) ))
				
			if j % 5 == 0:
				model_save_path_tmp = "trained_models/tmp/model.region_dataset.model{}.tumours{}.mut{}.ckpt".format(model_type, n_tumours, n_mut)
				save_path = saver.save(sess, model_save_path_tmp)
				print("Model saved in file: %s" % save_path)

			print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))
	
		save_path = saver.save(sess, model_save_path)
		print("Model saved in file: %s" % save_path)

def visualize(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names, region_size):
	# choose only true mutations

	real_examples = np.where(labels.ravel() > 0)[0]
	real_mutations = training_set[real_examples]
	real_tumour_ids = tumour_ids[real_examples]
	real_mut_vaf = mut_vaf[real_examples]

	# plot distrubtion over types
	mut_types = real_mutations[:,:96]
	plot_mut_types(np.mean(mut_types, axis=0), "mut_types.pdf")

	# plot distribution over features
	real_mutations = np.reshape(real_mutations[:,96:], [real_mutations.shape[0], region_size, -1])
	print(np.sum(real_mutations[:,:,:] > 0) / np.prod(real_mutations[:,:,:].shape))
	# plot only features at the location of mutation (all feature types for all mutations)
	non_zero_features = np.transpose(real_mutations[:,50,:]).tolist()
	for i in range(len(non_zero_features)):
		# transforming into log scale!!
		non_zero_features[i] = [np.log(x) for x in non_zero_features[i] if x > 0]
	plot_boxplot(non_zero_features, "region_features.pdf", np.reshape(feature_names[96:], [region_size, -1])[50])

	# compare distr over types in different tumours

	# plot distrubtion over types over time

	# plot distribution over features for each type

	# plot distribution over features for each tumour


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
	parser.add_argument('-a', '--adam', help='Rate for adam optimizer', default=1e-4,type=float)

	args = parser.parse_args()
	test_mode = args.test
	n_tumours = args.tumours
	target_n_mut = args.mut
	n_epochs = int(args.epochs)
	batch_size = int(args.batch)
	model_type = args.model
	feature_path = args.feature_path
	region_size = args.region_size
	adam_rate = args.adam
	#latent_dimension = args.latents

	print("Fitting model version: " + model_type)
	# model params
	reg_latent_dim = 100
	z_latent_dim = reg_latent_dim * 2
	n_tumours_per_batch = 51 # tumours in tumour_batch

	print(target_n_mut)

	n_parts_to_load = 1000
	if n_tumours is not None:
		n_parts_to_load = int(n_tumours)

	print("Loading dataset...")
	mut_features, region_counts = load_dataset(mut_dataset_path, n_parts = n_parts_to_load)
	trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))

	dataset_with_annotation = dataset_with_annotation.format(region_size = region_size)
	unique_tumours = np.unique(np.asarray(mut_features.Tumour).ravel())

	if n_tumours is None:
		n_tumours = len(unique_tumours)
	else:
		n_tumours = int(n_tumours)

	unique_tumours = unique_tumours[:n_tumours]
	available_tumours = [dataset_with_annotation.replace("{id}", tum) for tum in unique_tumours]

	print(mut_features.shape)

	n_mut, num_features, n_unique_tumours = make_training_set(mut_features, region_counts, trinuc, feature_path, region_size, dataset_with_annotation, max_tumours = n_tumours)
	mut_features, region_counts, n_mut = filter_mutation(mut_features, region_counts, target_n_mut)

	print(mut_features.shape)
	print(n_mut)

	print("Processing {} mutations from {} tumour(s) ...".format(n_mut, n_unique_tumours))

	tf.reset_default_graph()
	model_save_path = model_save_path.format(model_type, n_tumours, n_mut)
	os.makedirs(model_save_path, exist_ok=True)

	tf_vars, metrics, meta, extra = make_model(num_features, n_unique_tumours, z_latent_dim, model_type, model_save_path, adam_rate = adam_rate)

	x, x_tumour_ids, y_, log_y_prediction = tf_vars
	cross_entropy, accuracy = metrics
	train_step, saver = meta
	
	if model_type == "gaussian":
		z_t, y_region, predictions, L = extra
	else: 
		z_t, y_region, predictions = extra

	if not test_mode:
		train(tf_vars, n_epochs, batch_size, model_save_path, available_tumours)
	else:
		if not os.path.exists(model_save_path):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)

			training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours)
			x_train, x_test, y_train, y_test, x_tumour_ids_train, x_tumour_ids_test = train_test_split([training_set, labels, tumour_ids], split_by = mut_vaf, test_size=0.2)
			test_dict = {x: x_test, y_: y_test, x_tumour_ids: x_tumour_ids_test}

			visualize(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names, region_size)

			print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))

			print("Mean prediction")
			pred = predictions.eval(feed_dict=test_dict).ravel()
			print("1: " + str(np.mean(pred[y_test.ravel().astype(int)])))
			print("0: " + str(np.mean(pred[np.logical_not(y_test.ravel().astype(int))])))

			correct_prediction = correct_predictions(predictions, y_).eval(feed_dict=test_dict).ravel()

			print("Mean accuracy within a class")
			print("1: " + str(np.mean(correct_prediction[y_test.ravel().astype(int)])))
			print("0: " + str(np.mean(correct_prediction[np.logical_not(y_test.ravel().astype(int))])))

			print("Tumour representation std")
			print(np.mean(np.std(z_t.eval(feed_dict=test_dict), axis=0)))

			if model_type == "gaussian":
				print("Likelihood of region representation:")
				print("1: " + str(np.mean(L.eval(feed_dict=test_dict)[y_test.ravel().astype(int)])))
				print("0: " + str(np.mean(L.eval(feed_dict=test_dict)[np.logical_not(y_test.ravel().astype(int))])))

			unique_tumours = np.unique(np.squeeze(x_tumour_ids_test))

			for tum in unique_tumours:
				ind = np.squeeze(x_tumour_ids_test) == tum
				x_cur = x_test[ind]
				y_cur = y_test[ind]
				tumour_ids = x_tumour_ids_test[ind]
				n_mutations = sum(ind)
				tum_dict = {x: x_cur, y_: y_cur, x_tumour_ids: tumour_ids}

				print('Tumour %d: #mut %d cross_entropy %g  accuracy %g' % (tum, n_mutations, cross_entropy.eval(feed_dict=tum_dict), accuracy.eval(feed_dict=tum_dict)))
				

# add time component (z_t comes from RNN over time)
# version with cnn to make a summary of region features
# try relu versus tanh
# try different architectures of neural net (number of layers, number of units)

# predict the 96-vector directly from the region (softmax layer on top)
# try mean squared loss
# make proper normalization in gaussian model

