#!/usr/bin/python
# coding=utf-8

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

FLAGS = None

# size of the context roundt the mutation
region_size = 100 #!!!!!
# number of mutation per time point batch
batch_size_per_tp=120

DEF_FEATURE_PATH = "/Users/yulia/Documents/mutational_signatures/dna_features_ryoga/"

#dataset_path = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
mut_dataset_path = ["/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part1.pickle",
						"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part2.pickle",
						"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part3.pickle",
						"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part4.pickle",
						"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part5.pickle"]
feature_path = "/Users/yulia/Documents/mutational_signatures/dna_features_ryoga/"
model_save_path = "trained_models/model.region_dataset.model{}.tumours{}.mut{}/model.ckpt"
# dataset_with_annotation = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.small.over_time.annotation.hdf5"
dataset_with_annotation = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.region_size" + str(region_size) + ".over_time.annotation.hdf5"

def compute_test_ll(x_test, tumours_test, time_estimates_test, log_likelihood):
	test_ll = 0
	for tumour_data, tid, time in zip(x_test, tumours_test, time_estimates_test):
		tumour_dict = {x: tumour_data, tumour_id: [int(tid)], time_estimates: time}
		test_ll += log_likelihood.eval(feed_dict=tumour_dict)
	return test_ll / len(tumour_data)

def mutation_rate_model_over_time(X, time_estimates, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, batch_size_per_tp):
	num_features = X.get_shape()[2].value
	time_steps = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value
	z_latent_dim = reg_latent_dim*2

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, [tumour_id]))
	hidden_state = weight_variable([batch_size_per_tp, lstm_size])

	#encoder = init_neural_net_params([ batch_size_per_tp * num_features + z_latent_dim] + [500, 500, lstm_size])
	encoder_data = init_neural_net_params([ batch_size_per_tp * num_features] + [500, 500, batch_size_per_tp * reg_latent_dim])
	#encoder_latents = init_neural_net_params([ batch_size_per_tp * reg_latent_dim] + [500, 500, lstm_size])
	#next_state_nn = init_neural_net_params([lstm_size + 1, 500, 500, lstm_size])
	#decoder_latents = init_neural_net_params([lstm_size, 500, 500, reg_latent_dim * 2])
	next_state_gaussian_nn = init_neural_net_params([ batch_size_per_tp * reg_latent_dim] + [500, 500, reg_latent_dim * 2])
	decoder_data = init_neural_net_params([reg_latent_dim, 500, 500, num_features])

	log_likelihood = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	reconstruction_loss = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	reconstruction_next_batch_loss = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	mse =  tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	mse_sum = []

	# !!! do we need this rescaler ?? !!!
	rescaler = tf.constant(-0.919* reg_latent_dim * batch_size_per_tp, dtype=tf.float32)

	for batch, next_batch, vaf in zip(tf.unstack(X)[:-1], tf.unstack(X)[1:], tf.unstack(time_estimates)):
		
		# !!!! no z_t
		#input = tf.concat([ tf.reshape(batch, [-1]), z_t], 0)

		input = tf.reshape(batch, [-1])

		latents = neural_net(tf.expand_dims(input, 0) , encoder_data['weights'], encoder_data['biases'])
		reconstruction = neural_net(tf.reshape(latents, [batch_size_per_tp, reg_latent_dim]), decoder_data['weights'], decoder_data['biases'])
		
		latents = tf.reshape(latents, [1,-1])
		
		# hidden_state = neural_net(latents, encoder_latents['weights'], encoder_latents['biases'])
		# next_state = neural_net(tf.concat([hidden_state, tf.expand_dims(vaf,0)], 1), next_state_nn['weights'], next_state_nn['biases'])
		# predicted_region_gaussian = neural_net(next_state, decoder_latents['weights'], decoder_latents['biases'])
		predicted_region_gaussian = neural_net(latents, next_state_gaussian_nn['weights'], next_state_gaussian_nn['biases'])
		
		predicted_batch_mean, predicted_log_batch_sigma = unpack_gaussian(predicted_region_gaussian, reg_latent_dim)

		# taking the mean of the gaussian and comparing it to the batch
		# !!!! taking the mean, not the gaussian
		predicted_batch_m = neural_net(predicted_batch_mean, decoder_data['weights'], decoder_data['biases'])

		# take an encoding of the new batch
		# compute the likelihood of this encoding under the predicted gaussian
		next_batch = tf.expand_dims(tf.reshape(next_batch, [-1]),0)
		next_batch_latents = neural_net(next_batch , encoder_data['weights'], encoder_data['biases'])
		next_batch_latents = tf.reshape(next_batch_latents, [batch_size_per_tp, reg_latent_dim])
		next_batch_reconstruction = neural_net(next_batch_latents, decoder_data['weights'], decoder_data['biases'])
		next_batch_reconstruction = tf.expand_dims(tf.reshape(next_batch_reconstruction, [1,-1]),0)

		dist = tf.contrib.distributions.MultivariateNormalDiag(predicted_batch_mean, tf.abs(predicted_log_batch_sigma)) #tf.exp(predicted_log_batch_sigma)) #tf.ones([reg_latent_dim]))# tf.abs(batch_sigma))
		L = tf.reduce_mean(dist.log_prob(next_batch_latents) - rescaler)

		log_likelihood = tf.add(log_likelihood, L)

		reconstruction_loss = tf.add(reconstruction_loss, mean_squared_error(batch, reconstruction))
		reconstruction_next_batch_loss = tf.add(reconstruction_next_batch_loss, mean_squared_error(next_batch, next_batch_reconstruction))

	# hidden_state = tf.zeros([batch_size_per_tp, lstm.state_size])
	# current_state = tf.zeros([batch_size_per_tp, lstm.state_size])
	# state = hidden_state, current_state
	# probabilities = []
	# loss = 0.0

	# lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	# for bi in time_steps:
	#     # The value of state is updated after processing each batch of words.
	#     output, state = lstm(current_batch_of_words, state)

	#     The LSTM output can be used to make next word predictions
	#     logits = tf.matmul(output, softmax_w) + softmax_b
	#     probabilities.append(tf.nn.softmax(logits))
	#     loss += loss_function(probabilities, target_words)

	#  mse, tf.reduce_mean(tf.stack(mse_sum)),
	return log_likelihood, reconstruction_loss, reconstruction_next_batch_loss, predicted_log_batch_sigma


def make_model(num_features, n_unique_tumours, z_latent_dim, time_steps, batch_size_per_tp, lstm_size, model_save_path):
	x = tf.placeholder(tf.float32, [time_steps, batch_size_per_tp, num_features])
	tumour_id = tf.placeholder(tf.int32, [1])
	time_estimates = tf.placeholder(tf.float32, [time_steps,1])

	initial_tumour_latents = tf.abs(tf.concat([weight_variable([z_latent_dim//2,1]), weight_variable([z_latent_dim//2,1], mean=1)], axis = 0))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	log_likelihood, reconstr_before, reconstr_after, predicted_log_batch_sigma = \
		mutation_rate_model_over_time(x, time_estimates, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, batch_size_per_tp)
	cost = -log_likelihood + reconstr_before + reconstr_after + tf.reduce_sum(predicted_log_batch_sigma)
	#cost = reconstr_before + reconstr_after

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-5).minimize(cost)
		#train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_sum)

	save_graph()
	saver = tf.train.Saver()

	tf_vars = [x, tumour_id, time_estimates]
	metrics = [log_likelihood, cost]
	meta = [train_step, saver]
	extra = []

	return tf_vars, metrics, meta, extra

def train(train_data, test_data, tf_vars, metrics, n_epochs, batch_size, model_save_path):
	x_train, tumours_train, time_estimates_train = train_data
	x_test, tumours_test, time_estimates_test = test_data

	x, tumour_id, time_estimates = tf_vars
	log_likelihood, cost = metrics
	train_step, saver = meta

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for j in range(n_epochs):
			for tumour_data, tid, time in zip(x_train,tumours_train, time_estimates_train):
				tumour_dict = {x: tumour_data, tumour_id: [int(tid)], time_estimates: time}

				train_log_likelihood = log_likelihood.eval(feed_dict=tumour_dict)
				print(len(train_log_likelihood))

				print('Epoch %d: tumour %s:  training log_likelihood %g' % (j, tid, train_log_likelihood))
		
				test_ll = compute_test_ll(x_test, tumours_test, time_estimates_test, log_likelihood)
				print('test log_likelihood %g' % test_ll)

				
				print('train cost %g' % cost.eval(feed_dict=tumour_dict))
				cost_test = compute_test_ll(x_test, tumours_test, time_estimates_test, cost)
				print('test cost %g' % cost_test)
				train_step.run(feed_dict=tumour_dict)
			
			#if j % 5 == 0:
		# 		model_save_path_tmp = "trained_models/tmp/model.region_dataset.model{}.tumours{}.mut{}.ckpt".format(model_type, n_tumours, n_mut)
		# 		save_path = saver.save(sess, model_save_path_tmp)
		# 		print("Model saved in file: %s" % save_path)

		test_ll = compute_test_ll(x_test, tumours_test, time_estimates_test, log_likelihood)
		print('test log_likelihood %g' % test_ll)
	
		save_path = saver.save(sess, model_save_path)
		print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train model to predict probability for region-mutation pair')
	parser.add_argument('--test', help='test mode: only read trained params and analyze', action="store_true")
	#parser.add_argument('--latents', help='number of latent dimensions', default=100, type=int)
	parser.add_argument('-n', '--tumours', help='number of tumours to include in the set', default=None)
	parser.add_argument('-m', '--mut', help='number of mutations per tumour', default=None)
	parser.add_argument('-e','--epochs', help='number of epochs', default=1000)
	parser.add_argument('-b','--batch', help='batch size', default=500)
	parser.add_argument('--model', help='Model type: gaussian likelihood or neural net', default='nn')
	#parser.add_argument('--loss', help = "loss type: poisson or mean_squared", default="poisson")
	parser.add_argument('-f', '--feature-path', help='feature file path', default=DEF_FEATURE_PATH)
	parser.add_argument('-rs', '--region-size', help='size of training regions surrounding a mutation', default=region_size,type=int)

	args = parser.parse_args()
	test_mode = args.test
	n_tumours = args.tumours
	n_mut = args.mut
	n_epochs = int(args.epochs)
	batch_size = int(args.batch)
	model_type = args.model
	feature_path = args.feature_path
	region_size = args.region_size
	#latent_dimension = args.latents

	print("Fitting model version: " + model_type)
	# model params
	# size of region latents
	reg_latent_dim = 80
	# size of lstm latent state
	lstm_size = 90
	z_latent_dim = reg_latent_dim * 2

	print("Loading dataset...")
	mut_features, region_counts = load_dataset(mut_dataset_path)
	trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))

	mut_features, region_counts, n_mut = filter_mutation(mut_features, region_counts, n_mut)

	if os.path.exists(dataset_with_annotation):
		dictionary = load_from_HDF(dataset_with_annotation)
		training_set = dictionary["training_set"]
		labels = dictionary["labels"]
		n_unique_tumours = int(dictionary["n_unique_tumours"])
		x_tumour_ids = dictionary["x_tumour_ids"]
		mut_vaf = dictionary["mut_vaf"]
	else:
		training_set, labels, n_unique_tumours, x_tumour_ids, mut_vaf = make_training_set(mut_features, region_counts, trinuc, feature_path, region_size)
		# !!!!!!
		# save_to_HDF(dataset_with_annotation, 
		# 	{"training_set": training_set, "labels": labels, "n_unique_tumours": np.array(n_unique_tumours), "x_tumour_ids" : x_tumour_ids, "mut_vaf": mut_vaf})

		save_to_HDF(dataset_with_annotation, 
		 	{"training_set": training_set, "labels": labels, "n_unique_tumours": np.array(n_unique_tumours), "x_tumour_ids" : x_tumour_ids, "mut_vaf": mut_vaf})

	indices = np.random.choice(training_set.shape[0], size=n_mut)
	training_set = training_set[indices]
	labels = labels[indices]
	x_tumour_ids = x_tumour_ids[indices]

	print("Processing {} mutations from {} tumour(s) ...".format(training_set.shape[0], n_unique_tumours))

	num_features = training_set.shape[1]

	tf.reset_default_graph()
	model_save_path = model_save_path.format(model_type, n_tumours, n_mut)
	os.makedirs(model_save_path, exist_ok=True)

	training_set_over_time, unique_tumours, time_estimates = make_batches_over_time(training_set, labels, n_unique_tumours, x_tumour_ids, mut_vaf, batch_size_per_tp)

	# Split dataset into train / test
	x_train, x_test, tumours_train, tumours_test, time_estimates_train, time_estimates_test  = model_selection.train_test_split(training_set_over_time, unique_tumours, time_estimates, test_size=0.2, random_state = 1991)

	time_steps = x_train.shape[1]
	
	tf_vars, metrics, meta, extra = make_model(num_features, n_unique_tumours, z_latent_dim, time_steps, batch_size_per_tp, lstm_size, model_save_path)

	x, tumour_id, time_estimates = tf_vars
	log_likelihood, cost = metrics
	train_step, saver = meta

	if not test_mode:
		print("Optimizing...")
		
		train_data = [x_train, tumours_train, time_estimates_train]
		test_data = [x_test, tumours_test, time_estimates_test]
		train(train_data, test_data, tf_vars, metrics, n_epochs, batch_size, model_save_path)
	else:
		if not os.path.exists(model_save_path):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)

			# print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			# print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))

			# print("Mean prediction")
			# pred = predictions.eval(feed_dict=test_dict).ravel()
			# print("1: " + str(np.mean(pred[y_test.ravel().astype(int)])))
			# print("0: " + str(np.mean(pred[np.logical_not(y_test.ravel().astype(int))])))

			# correct_prediction = correct_predictions(predictions, y_).eval(feed_dict=test_dict).ravel()

			# print("Mean accuracy within a class")
			# print("1: " + str(np.mean(correct_prediction[y_test.ravel().astype(int)])))
			# print("0: " + str(np.mean(correct_prediction[np.logical_not(y_test.ravel().astype(int))])))

			# print("Tumour representation std")
			# print(np.std(z_t.eval(feed_dict=test_dict), axis=0))

			# if model_type == "gaussian":
			# 	print("Likelihood of region representation:")
			# 	print("1: " + str(np.mean(L.eval(feed_dict=test_dict)[y_test.ravel().astype(int)])))
			# 	print("0: " + str(np.mean(L.eval(feed_dict=test_dict)[np.logical_not(y_test.ravel().astype(int))])))


# add time component (z_t comes from RNN over time)
# version with cnn to make a summary of region features
# try relu versus tanh
# try different architectures of neural net (number of layers, number of units)

# predict the 96-vector directly from the region (softmax layer on top)
# try mean squared loss
# make proper normalization in gaussian model

