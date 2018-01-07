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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

FLAGS = None

if os.path.isdir("/home/yulia/"):
	DIR = "/home/yulia/mnt/"
else:
	DIR = "/Users/yulia/Documents/mutational_signatures/"

DEF_FEATURE_PATH = DIR + "/dna_features_ryoga/"

#dataset_path = DIR + "/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
mut_dataset_path = [DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part*.pickle"]
feature_path = DIR + "/dna_features_ryoga/"
model_save_path = "trained_models/model.region_dataset.model{}.tumours{}.mut{}/model.ckpt"
# dataset_with_annotation = DIR + "mutation_prediction_data/region_dataset.small.over_time.annotation.hdf5"
# datasets with at most 10000 mutations per tumour
dataset_with_annotation = DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.region_size{region_size}.ID{{id}}.over_time.annotation.hdf5"

def evaluate_on_each_tumour(x_data, tumours_data, time_estimates_data, metric):
	evaluated_metric = 0
	n_tumour_batches = x_data.shape[0]

	for tumour_data, tid, time in zip(x_data, tumours_data, time_estimates_data):
		tumour_dict = {x: tumour_data, tumour_id: np.array(tid).astype(int)[:,np.newaxis], 
						time_estimates: time, 
						time_series_lengths: np.squeeze(np.apply_along_axis(sum, 1, time_estimates_data > 0), axis=0),
						sequences_per_batch_tf: tumour_data.shape[1]}
		evaluated_metric += metric.eval(feed_dict=tumour_dict) / n_tumour_batches
	return evaluated_metric

def collect_on_each_tumour(x_data, tumours_data, time_estimates_data, metric):
	evaluated_metric = []
	n_tumour_batches = x_data.shape[0]

	for tumour_data, tid, time in zip(x_data, tumours_data, time_estimates_data):
		tumour_dict = {x: tumour_data, tumour_id: np.array(tid).astype(int)[:,np.newaxis], 
						time_estimates: time, 
						time_series_lengths: np.squeeze(np.apply_along_axis(sum, 1, time_estimates_data > 0), axis=0),
						sequences_per_batch_tf: tumour_data.shape[1]}
		evaluated_metric.append(metric.eval(feed_dict=tumour_dict))
	return evaluated_metric

def mutation_rate_model_over_time(X, time_estimates, time_series_lengths, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, batch_size_per_tp, sequences_per_batch_tf):
	#with tf.device('/gpu:0'):
	num_features = X.get_shape()[3].value
	time_steps = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value
	z_latent_dim = reg_latent_dim*2

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, [tumour_id]))
	#hidden_state = weight_variable([sequences_per_batch, batch_size_per_tp, lstm_size])
	prev_hidden_state = tf.tile(tf.zeros([lstm_size], tf.float32), sequences_per_batch_tf)
	prev_hidden_state = tf.fill([sequences_per_batch_tf, lstm_size], tf.constant(0, dtype=tf.float32))
	prev_hidden_state = tf.reshape(prev_hidden_state, [-1, lstm_size])

	#encoder = init_neural_net_params([ batch_size_per_tp * num_features + z_latent_dim] + [500, 500, lstm_size])
	encoder_data = init_neural_net_params([num_features] + [500, 500, reg_latent_dim])
	#encoder_latents = init_neural_net_params([ batch_size_per_tp * reg_latent_dim] + [500, 500, lstm_size])
	next_state_nn = init_neural_net_params([batch_size_per_tp * reg_latent_dim + lstm_size + 1, 500, 500, lstm_size])
	decoder_latents = init_neural_net_params([lstm_size, 500, 500, reg_latent_dim * 2])
	#next_state_gaussian_nn = init_neural_net_params([ batch_size_per_tp * reg_latent_dim] + [500, 500, reg_latent_dim * 2])
	decoder_data = init_neural_net_params([reg_latent_dim, 500, 500, num_features])

	log_likelihood = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	reconstruction_loss = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	reconstruction_next_batch_loss = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	mse =  tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	mse_sum = []

	# !!! do we need this rescaler ?? !!!
	rescaler = tf.constant(-0.919* reg_latent_dim * batch_size_per_tp, dtype=tf.float32)

	prediction_means = []

	for batch, next_batch, vaf in zip(tf.unstack(X)[:-1], tf.unstack(X)[1:], tf.unstack(time_estimates)):
		# !!!! no z_t
		#input = tf.concat([ tf.reshape(batch, [-1]), z_t], 0)

		batch = tf.reshape(batch, [-1,num_features])
		next_batch = tf.reshape(next_batch, [-1,num_features])

		latents = neural_net(batch, encoder_data['weights'], encoder_data['biases'])
		reconstruction = neural_net(latents, decoder_data['weights'], decoder_data['biases'])
		
		latents = tf.reshape(latents, [sequences_per_batch_tf, batch_size_per_tp * reg_latent_dim])

		print("hello")
		print(prev_hidden_state.get_shape())
		print(latents.get_shape())
		print(vaf.get_shape())
		print(lstm_size)
		print(reg_latent_dim)
		print(batch_size_per_tp)
		print(batch_size_per_tp * reg_latent_dim + lstm_size + 1)
		print(tf.concat([prev_hidden_state, latents, vaf], 1).get_shape())

		#hidden_state = neural_net(latents, encoder_latents['weights'], encoder_latents['biases'])
		next_state = neural_net(tf.concat([tf.squeeze(prev_hidden_state), latents, vaf], 1), next_state_nn['weights'], next_state_nn['biases'])
		prev_hidden_state = next_state

		predicted_region_gaussian = neural_net(next_state, decoder_latents['weights'], decoder_latents['biases'])
		#predicted_region_gaussian = neural_net(latents, next_state_gaussian_nn['weights'], next_state_gaussian_nn['biases'])

		predicted_batch_mean, predicted_log_batch_sigma = unpack_gaussian(predicted_region_gaussian, reg_latent_dim)

		# taking the mean of the gaussian and comparing it to the batch
		# !!!! taking the mean, not the gaussian
		predicted_batch_m = neural_net(predicted_batch_mean, decoder_data['weights'], decoder_data['biases'])
		prediction_means.append(predicted_batch_m)

		# take an encoding of the new batch
		# compute the likelihood of this encoding under the predicted gaussian
		#next_batch = tf.expand_dims(tf.reshape(next_batch, [-1]),0)
		next_batch_latents = neural_net(next_batch , encoder_data['weights'], encoder_data['biases'])
		next_batch_latents = tf.reshape(next_batch_latents, [sequences_per_batch_tf * batch_size_per_tp, reg_latent_dim])
		next_batch_reconstruction = neural_net(next_batch_latents, decoder_data['weights'], decoder_data['biases'])
		#next_batch_reconstruction = tf.expand_dims(tf.reshape(next_batch_reconstruction, [1,-1]),0)

		print("check")
		print(batch.get_shape())
		print(next_batch.get_shape())
		print(next_state.get_shape())
		print(predicted_region_gaussian.get_shape())
		print(predicted_batch_mean.get_shape())
		print(next_batch_latents.get_shape())
		print(next_batch_reconstruction.get_shape())

		dist = tf.contrib.distributions.MultivariateNormalDiag(predicted_batch_mean, tf.abs(predicted_log_batch_sigma)) #tf.exp(predicted_log_batch_sigma)) #tf.ones([reg_latent_dim]))# tf.abs(batch_sigma))
		next_batch_latents = tf.reshape(next_batch_latents, [sequences_per_batch_tf, batch_size_per_tp, reg_latent_dim])
		
		L = tf.reduce_mean(dist.log_prob(tf.transpose(next_batch_latents,(1,0,2))) - rescaler, axis=0)
		L = L * tf.to_float(tf.greater(tf.squeeze(vaf),tf.constant(0.0))) / time_series_lengths

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

	predictions = tf.concat(tf.expand_dims(prediction_means, 0), 0)

	return tf.reduce_mean(log_likelihood), reconstruction_loss, reconstruction_next_batch_loss, predicted_log_batch_sigma, predictions

def make_model(num_features, n_unique_tumours, z_latent_dim, time_steps, batch_size_per_tp, lstm_size, model_save_path, adam_rate = 1e-3):
	x = tf.placeholder(tf.float32, [time_steps, None, batch_size_per_tp, num_features])
	tumour_id = tf.placeholder(tf.int32, [None, 1])
	time_estimates = tf.placeholder(tf.float32, [time_steps,None,1])
	time_series_lengths = tf.placeholder(tf.float32, [None,1])
	sequences_per_batch_tf = tf.placeholder(dtype=tf.int32)

	initial_tumour_latents = tf.abs(tf.concat([weight_variable([z_latent_dim//2,1]), weight_variable([z_latent_dim//2,1], mean=1)], axis = 0))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	log_likelihood, reconstr_before, reconstr_after, predicted_log_batch_sigma, predictions= \
		mutation_rate_model_over_time(x, time_estimates, time_series_lengths, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, batch_size_per_tp, sequences_per_batch_tf)
	
	#cost = -log_likelihood + 0.01*(reconstr_before + reconstr_after) + tf.maximum(tf.constant(-10000.0), tf.reduce_mean(predicted_log_batch_sigma))
	cost = reconstr_before + reconstr_after

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(adam_rate).minimize(cost)
		#train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_sum)

	save_graph()
	saver = tf.train.Saver()

	tf_vars = [x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions]
	metrics = [log_likelihood, cost]
	meta = [train_step, saver]
	extra = []

	return tf_vars, metrics, meta, extra

def train(tf_vars, metrics, n_epochs, model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps):
	x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions = tf_vars
	log_likelihood, cost = metrics
	train_step, saver = meta

	test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, annot, features = read_tumour_data(tumour_files_test)
	x_test, tumours_test, time_estimates_test = make_batches_over_time(test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, batch_size_per_tp, n_unique_tumours_test, n_timesteps)

	test_data = [x_test, tumours_test, time_estimates_test]
	time_steps = x_test.shape[1]

	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train)
	x_train, tumours_train, time_estimates_train = make_batches_over_time(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, n_unique_tumours, n_timesteps)
	sequences_per_batch = x_train.shape[2]

	print("Optimizing...")
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())

		for j in range(n_epochs):
				for k in range(len(tumour_files_train) // n_tumours_per_batch+1):
					if len(tumour_files_train) > n_tumours_per_batch:
						training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train, n_tumours_per_batch, k)
						if training_set is None:
							continue
						x_train, tumours_train, time_estimates_train = make_batches_over_time(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, sequences_per_batch, n_timesteps)
						sequences_per_batch = x_train.shape[2]

					train_data = [x_train, tumours_train, time_estimates_train]
					train_dict_batch = {x: x_train, tumour_id: tumours_train, time_estimates: time_estimates_train,
										time_series_lengths: np.squeeze(np.apply_along_axis(sum, 1, time_estimates_train > 0), axis=0),
										sequences_per_batch_tf: sequences_per_batch}

					for h in range(10): #how many epochs to we want to train on the same set of tumours
						for tumour_data, tid, time in zip(x_train, tumours_train, time_estimates_train):
							tumour_dict = {x: tumour_data, tumour_id: np.expand_dims(tid,1), time_estimates: time,
											time_series_lengths: np.apply_along_axis(sum, 0, time > 0),
											sequences_per_batch_tf: tumour_data.shape[1]}
							train_step.run(feed_dict=tumour_dict)

					train_log_likelihood = evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, log_likelihood)
					print('Epoch %d.%d:' % (j, k))
					print('training log_likelihood %g' % (train_log_likelihood))
			
					test_ll = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, log_likelihood)
					print('test log_likelihood %g' % test_ll)

					print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, cost))
					cost_test = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, cost)
					print('test cost %g' % cost_test)
				
					#if j % 5 == 0:
				# 		model_save_path_tmp = "trained_models/tmp/model.region_dataset.model{}.tumours{}.mut{}.ckpt".format(model_type, n_tumours, n_mut)
				# 		save_path = saver.save(sess, model_save_path_tmp)
				# 		print("Model saved in file: %s" % save_path)

		test_ll = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, log_likelihood)
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
	parser.add_argument('-rs', '--region-size', help='size of training regions surrounding a mutation', default=100,type=int)
	parser.add_argument('-a', '--adam', help='Rate for adam optimizer', default=1e-3,type=float)

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
	#latent_dimension = args.latents

	print("Fitting model version: " + model_type)
	# model params
	# number of mutation per time point batch
	batch_size_per_tp=120
	sequences_per_batch = 8
	n_tumours_per_batch = 100 # tumours in tumour_batch
	# size of region latents
	reg_latent_dim = 80
	# size of lstm latent state
	lstm_size = 150
	z_latent_dim = reg_latent_dim * 2

	n_timesteps = 30
	# !!! change this in the training_set function too!!!

	n_parts_to_load = 1000
	if n_tumours is not None:
		n_parts_to_load = int(n_tumours)

	print("Loading dataset...")
	# region_counts -- counts of mutations of different types in the region surrounding the position of interest
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
	mut_features, region_counts, n_mut = filter_mutation(mut_features, region_counts, n_mut)

	print("Processing {} mutations from {} tumour(s) ...".format(n_mut, n_unique_tumours))

	tf.reset_default_graph()
	model_save_path = model_save_path.format(model_type, n_tumours, n_mut)
	os.makedirs(model_save_path, exist_ok=True)
	
	tf_vars, metrics, meta, extra = make_model(num_features, n_unique_tumours, z_latent_dim, n_timesteps, batch_size_per_tp, lstm_size, model_save_path, adam_rate = adam_rate)

	x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions = tf_vars
	log_likelihood, cost = metrics
	train_step, saver = meta

	# # Split dataset into train / test
	tumour_files_train, tumour_files_test = model_selection.train_test_split(available_tumours, test_size=0.2, random_state = 1991)

	if not test_mode:
		train(tf_vars, metrics, n_epochs , model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps)
	else:
		if not os.path.exists(model_save_path):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)

			training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train)
			x_train, tumours_train, time_estimates_train = make_batches_over_time(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, n_unique_tumours, n_timesteps)

			test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, annot, features = read_tumour_data(tumour_files_test)
			x_test, tumours_test, time_estimates_test = make_batches_over_time(test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, batch_size_per_tp, n_unique_tumours_test, n_timesteps)

			train_log_likelihood = evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, log_likelihood)
			print('training log_likelihood %g' % (train_log_likelihood))
	
			test_ll = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, log_likelihood)
			print('test log_likelihood %g' % test_ll)

			print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, cost))
			cost_test = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, cost)
			print('test cost %g' % cost_test)

			collected_predictions = collect_on_each_tumour(x_test, tumours_test, time_estimates_test, predictions)
			print(len(collected_predictions))
			print(np.array(collected_predictions).shape)
			print(collected_predictions[0][0][0][0][:96])

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

