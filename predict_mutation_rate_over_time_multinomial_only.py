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
from plot_signatures import *
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

# Notes: use regular RNN to predict the multinomial distribution of mutation types (no region features)

FLAGS = None

if os.path.isdir("/home/yulia/"):
	DIR = "/home/yulia/mnt/"
else:
	DIR = "/Users/yulia/Documents/mutational_signatures/"

DEF_FEATURE_PATH = DIR + "/dna_features_ryoga/"

#dataset_path = DIR + "/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
mut_dataset_path = [DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part*.pickle"]
feature_path = DIR + "/dna_features_ryoga/"
file_name = os.path.basename(__file__)[:-3]
model_dir = "trained_models/model." + file_name + ".tumours{}.mut{}.timesteps{}/"
# dataset_with_annotation = DIR + "mutation_prediction_data/region_dataset.small.over_time.annotation.hdf5"
# datasets with at most 10000 mutations per tumour
dataset_with_annotation = DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.region_size{region_size}.ID{{id}}.over_time.annotation.hdf5"

def mutation_rate_model_over_time(X, time_estimates, time_series_lengths, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, \
									batch_size_per_tp, sequences_per_batch_tf, next_state_nn_dim, predictor_nn_dim, init_scale=0.01):
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

	next_state_nn_dim = [num_features + lstm_size + 1] + next_state_nn_dim + [lstm_size]
	predictor_nn_dim = [lstm_size] + predictor_nn_dim + [num_features]

	next_state_nn = init_neural_net_params(next_state_nn_dim, stddev = init_scale, bias = init_scale)
	predictor = init_neural_net_params(predictor_nn_dim, stddev = init_scale, bias = init_scale)
	loss = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	type_prob_sum = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)

	prediction_means = []

	for batch, next_batch, vaf in zip(tf.unstack(X)[:-1], tf.unstack(X)[1:], tf.unstack(time_estimates)):
		# !!!! no z_t
		#input = tf.concat([ tf.reshape(batch, [-1]), z_t], 0)

		# to compute likelihood for the batch of 120 mutations
		# uncomment mut_types = (np.sum(mut_types, axis =0)/sum(np.sum(mut_types, axis =0)))[np.newaxis,:] in training_set.py
		# set x = tf.placeholder(tf.float32, [time_steps, None, 1, 96])
		# batch = tf.reshape(batch, [-1,num_features])
		# next_batch = tf.reshape(next_batch, [-1,num_features])

		# #hidden_state = neural_net(latents, encoder_latents['weights'], encoder_latents['biases'])
		# next_state = neural_net(tf.concat([tf.squeeze(prev_hidden_state), batch, vaf], 1), next_state_nn['weights'], next_state_nn['biases'])
		# prev_hidden_state = next_state

		# predicted_mut_types = neural_net(next_state, predictor['weights'], predictor['biases'])
		# prediction_means.append(predicted_mut_types)

		# # in multinomial_only data is normalized by the number of mutations in the batch
		# next_batch_counts = next_batch * tf.cast(batch_size_per_tp, dtype=tf.float32)
		# dist = tf.contrib.distributions.Multinomial(total_count=tf.reduce_sum(next_batch_counts, axis=1), logits=predicted_mut_types, validate_args = True)
		# type_prob = dist.log_prob(next_batch_counts) * tf.to_float(tf.greater(tf.squeeze(vaf),tf.constant(0.0))) / (time_series_lengths - 1)
		# type_prob_sum = tf.add(type_prob_sum, type_prob)


		# to compute likelihood for each mutation separately
		# comment out mut_types = (np.sum(mut_types, axis =0)/sum(np.sum(mut_types, axis =0)))[np.newaxis,:] in training_set.py
		# set x = tf.placeholder(tf.float32, [time_steps, None, batch_size_per_tp, 96])
		frequencies = tf.reduce_sum(batch, axis =1)
		frequencies = frequencies / (tf.expand_dims(tf.reduce_sum(frequencies, axis =1), axis =1) + tf.constant(0.0001))
		frequencies = tf.reshape(frequencies, [-1,num_features])

		next_state = neural_net(tf.concat([tf.squeeze(prev_hidden_state), frequencies, vaf], 1), next_state_nn['weights'], next_state_nn['biases'])
		prev_hidden_state = next_state

		predicted_mut_types = neural_net(next_state, predictor['weights'], predictor['biases'])
		prediction_means.append(predicted_mut_types)	
		type_prob = compute_mut_type_prob(next_batch, n_mut_types, predicted_mut_types, vaf=vaf, time_series_lengths=time_series_lengths)
		type_prob_sum = tf.add(type_prob_sum, type_prob)


		mse_batch = mean_squared_error(next_batch, predicted_mut_types) * tf.to_float(tf.greater(tf.squeeze(vaf),tf.constant(0.0))) / (time_series_lengths -1)

		loss = tf.add(loss, mse_batch)
	
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

	return tf.reduce_mean(loss), tf.reduce_mean(type_prob_sum), predictions

def make_model(n_unique_tumours, z_latent_dim, time_steps, batch_size_per_tp, lstm_size, model_save_path, \
					next_state_nn_dim, predictor_nn_dim, adam_rate = 1e-3, init_scale = 0.01):
	x = tf.placeholder(tf.float32, [time_steps, None, batch_size_per_tp, 96])
	tumour_id = tf.placeholder(tf.int32, [None, 1])
	time_estimates = tf.placeholder(tf.float32, [time_steps,None,1])
	time_series_lengths = tf.placeholder(tf.float32, [None,1])
	sequences_per_batch_tf = tf.placeholder(dtype=tf.int32)

	initial_tumour_latents = tf.abs(tf.concat([weight_variable([z_latent_dim//2,1]), weight_variable([z_latent_dim//2,1], mean=1)], axis = 0))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	mse, likelihood, predictions = \
		mutation_rate_model_over_time(x, time_estimates, time_series_lengths, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, \
					batch_size_per_tp, sequences_per_batch_tf, next_state_nn_dim, predictor_nn_dim, init_scale = init_scale)
	
	cost = -likelihood

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(adam_rate).minimize(cost)
		#train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_sum)

	save_graph()
	saver = tf.train.Saver()

	tf_vars = [x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions]
	metrics = [mse, likelihood, cost]
	meta = [train_step, saver]
	extra = []

	return tf_vars, metrics, meta, extra

def train(tf_vars, metrics, meta, extra, n_epochs, model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps):
	x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions = tf_vars
	mse, likelihood, cost = metrics
	train_step, saver = meta
	#next_batch_counts = extra[0]

	test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, annot, features = read_tumour_data(tumour_files_test)
	x_test, tumours_test, time_estimates_test = make_batches_over_time_type_multinomials(test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, batch_size_per_tp, n_unique_tumours_test, n_timesteps)

	test_data = [x_test, tumours_test, time_estimates_test]
	time_steps = x_test.shape[1]

	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train)
	x_train, tumours_train, time_estimates_train = make_batches_over_time_type_multinomials(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, n_unique_tumours, n_timesteps)
	sequences_per_batch = x_train.shape[2]

	print("Optimizing...")
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())

		print('Initial')
		print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, cost))
		cost_test = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, cost)
		print('test cost %g' % cost_test)

		for j in range(n_epochs):
				for k in range(len(tumour_files_train) // n_tumours_per_batch+1):
					if len(tumour_files_train) > n_tumours_per_batch:
						training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train, n_tumours_per_batch, k)
						if training_set is None:
							continue
						x_train, tumours_train, time_estimates_train = make_batches_over_time_type_multinomials(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, sequences_per_batch, n_timesteps)
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

					print('Epoch %d.%d:' % (j, k))
					print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, cost))
					cost_test = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test,  tf_vars, cost)
					print('test cost %g' % cost_test)
				
					if j % 5 == 0:
						model_save_path_tmp = "trained_models/tmp/model.region_dataset.model{}.tumours{}.mut{}.ckpt".format(model_type, n_tumours, n_mut)
						save_path = saver.save(sess, model_save_path_tmp)
						print("Model saved in file: %s" % save_path)

		cost_test = evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test,  tf_vars, cost)
		print('test cost %g' % cost_test)
	
		save_path = saver.save(sess, model_save_path)
		print("Model saved in file: %s" % save_path)
		return cost_test

# # for spearmint
def main(job_id, params):
	print(params)

	# n_tumours = params["n_tumours"]
	# n_mut = params["n_mut"]
	# n_epochs = params["n_epochs"]
	# batch_size = params["batch_size"]
	# adam_rate = params["adam_rate"]
	# n_timesteps = params["n_timesteps"]
	# batch_size_per_tp=params["batch_size_per_tp"]
	# sequences_per_batch = params["sequences_per_batch"]
	# lstm_size = params["lstm_size"]
	# next_state_nn_dim = params["next_state_nn_dim"]
	# predictor_nn_dim = params["predictor_nn_dim"]

	n_tumours = 10
	n_mut = None
	n_epochs = 100
	batch_size = 500
	adam_rate = 1e-3
	n_timesteps = 10
	batch_size_per_tp=120
	sequences_per_batch = 8
	lstm_size = 150
	next_state_nn_dim = params["next_state_nn_dim"]
	predictor_nn_dim = params["predictor_nn_dim"]
	init_scale = 0.01

	mut_features, unique_tumours, n_tumours, n_mut, available_tumours, num_features, n_unique_tumours = \
		load_filter_dataset(mut_dataset_path, feature_path, dataset_with_annotation, region_size, n_tumours, n_mut)

	print("Processing {} mutations from {} tumour(s) ...".format(n_mut, n_unique_tumours))
	model_dir, model_save_path = prepare_model_dir(sys.argv, model_dir, __file__, [n_tumours, n_mut, n_timesteps])

	tf_vars, metrics, meta, extra = make_model(n_unique_tumours, z_latent_dim, n_timesteps, batch_size_per_tp, lstm_size, model_save_path, \
									next_state_nn_dim, predictor_nn_dim, adam_rate = adam_rate, init_scale = init_scale)

	tumour_files_train, tumour_files_test = model_selection.train_test_split(available_tumours, test_size=0.2, random_state = 1991)
	res = train(tf_vars, metrics, meta, extra, n_epochs, model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps)
	return res

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
	parser.add_argument('-t', '--time-steps', help='number of time steps for RNN. Other time steps are ignored', default=10,type=int)

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
	n_timesteps = args.time_steps
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

	next_state_nn_dim = [100, 100]
	predictor_nn_dim = [100, 100]

	mut_features, unique_tumours, n_tumours, n_mut, available_tumours, num_features, n_unique_tumours = \
		load_filter_dataset(mut_dataset_path, feature_path, dataset_with_annotation, region_size, n_tumours, n_mut)

	print("Processing {} mutations from {} tumour(s) ...".format(n_mut, n_unique_tumours))
	model_dir, model_save_path = prepare_model_dir(sys.argv, model_dir, __file__, [n_tumours, n_mut, n_timesteps])

	tf_vars, metrics, meta, extra = make_model(n_unique_tumours, z_latent_dim, n_timesteps, batch_size_per_tp, lstm_size, model_save_path, \
									next_state_nn_dim, predictor_nn_dim, init_scale = 0.1, adam_rate = adam_rate)

	x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions = tf_vars
	mse, likelihood, cost = metrics
	train_step, saver = meta

	# # Split dataset into train / test
	tumour_files_train, tumour_files_test = model_selection.train_test_split(available_tumours, test_size=0.2, random_state = 1991)

	if not test_mode:
		train(tf_vars, metrics, meta, extra, n_epochs , model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps)
	else:
		if not os.path.exists(model_save_path + ".index"):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)

			training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train)
			x_train, tumours_train, time_estimates_train = make_batches_over_time_type_multinomials(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, n_unique_tumours, n_timesteps)

			test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, annot, features = read_tumour_data(tumour_files_test)
			x_test, tumours_test, time_estimates_test = make_batches_over_time_type_multinomials(test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, batch_size_per_tp, n_unique_tumours_test, n_timesteps)

			#print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train,  tf_vars, cost))
			print('test cost %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test,  tf_vars, cost))

			# Plots
			# Test
			collected_predictions = collect_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, predictions)
			collected_predictions = np.array(np.squeeze(collected_predictions))
			predicted_types = collected_predictions[:,:,:n_mut_types]
			# show the logits
			predicted_types = softmax(predicted_types, axis=2)

			ground_truth = np.squeeze(x_test)
			ground_truth_types = np.sum(ground_truth[:,:,:,:n_mut_types], axis=2)
			ground_truth_types /= batch_size_per_tp
			# to remove the first time point -- it is not predicted
			ground_truth_types = ground_truth_types[1:]


			plot_types_over_time(ground_truth_types, plot_name=model_dir + "ground_types_test.pdf", ylabel = "Mutation types")
			plot_types_over_time(predicted_types, plot_name=model_dir +"predicted_types_test.pdf", ylabel = "Mutation types")

			# Train
			collected_predictions = collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, predictions)
			collected_predictions = np.array(np.squeeze(collected_predictions))
			predicted_types = collected_predictions[:,:,:n_mut_types]
			predicted_types = softmax(predicted_types, axis=2)

			ground_truth = np.squeeze(x_train)
			ground_truth_types = np.sum(ground_truth[:,:,:,:n_mut_types], axis=2)
			ground_truth_types /= batch_size_per_tp
			# to remove the first time point -- it is not predicted
			ground_truth_types = ground_truth_types[1:]
		
			plot_types_over_time(ground_truth_types, plot_name=model_dir +"ground_types_train.pdf", ylabel = "Mutation types")
			plot_types_over_time(predicted_types, plot_name=model_dir + "predicted_types_train.pdf", ylabel = "Mutation types")


			# Stats
			collected_predictions = collect_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, predictions)
			print(len(collected_predictions))
			print(np.array(collected_predictions).shape)
			print(collected_predictions[0][0][0][0][:96])

			collected_predictions = np.squeeze(collected_predictions)

			print("Difference between predictions within a tumour:")
			print(collected_predictions[:,0,:96])
			print(collected_predictions[:,1,:96])

			print("std")
			print(np.apply_along_axis(np.std, 0, collected_predictions[:,0,:96]))
			print(np.apply_along_axis(np.std, 0, collected_predictions[:,1,:96]))

			def find_max_diff(x):
				return max(x) - min(x)

			print("max diff")
			print(np.apply_along_axis(find_max_diff, 0, collected_predictions[:,0,:96]))
			print(np.apply_along_axis(find_max_diff, 0, collected_predictions[:,1,:96]))

			print("Difference in predictions across tumours:")
			print("Early")
			print(collected_predictions[0,0,:96] - collected_predictions[0,1,:96])

			print("Middle")
			print(collected_predictions[15,0,:96] - collected_predictions[15,1,:96])

			print("Late")
			print(collected_predictions[28,0,:96] - collected_predictions[28,1,:96])

			# How far are the predicted mutations from real ones (at least some in the batch?)
			print("Difference with the ground truth:")
			ground_truth = np.squeeze(x_test, axis=3)[0][1:] # The value of the first time point is not predicted
			print("Early")
			print(collected_predictions[0,0,:96]) 
			print(ground_truth[0,0,:96])
			
			print("Early: top 6 types:")
			print("Predicted: " + str(np.where(np.argsort(collected_predictions[0,0,:96]) >= 90)))
			print("Correct: " + str(np.where(np.argsort(ground_truth[0,0,:96]) >= 90)))

			print("Early: difference for non zero values:")
			non_zero_values = np.where(ground_truth[0,0,:96])[0]
			zero_values = np.where(ground_truth[0,0,:96] == 0)[0]

			print(np.mean((collected_predictions[0,0,:96] - ground_truth[0,0,:96])[non_zero_values]))
			print(np.mean(np.abs((collected_predictions[0,0,:96] - ground_truth[0,0,:96])[non_zero_values])))

			print("Early: difference for zero values:")
			print(np.mean((collected_predictions[0,0,:96] - ground_truth[0,0,:96])[zero_values]))
			print(np.mean(np.abs((collected_predictions[0,0,:96] - ground_truth[0,0,:96])[zero_values])))

			print("Early: average non zero values:")
			print(np.mean((collected_predictions[0,0,:96])[non_zero_values]))
			print("Early: average zero values:")
			print(np.mean((collected_predictions[0,0,:96])[zero_values]))

			print("Middle")
			print(collected_predictions[15,0,:96]) 
			print(ground_truth[15,0,:96])

			print("Middle: top 6 types:")
			print("Predicted: " + str(np.where(np.argsort(collected_predictions[15,0,:96]) >= 90)))
			print("Correct: " + str(np.where(np.argsort(ground_truth[15,0,:96]) >= 90)))

			non_zero_values = np.where(ground_truth[15,0,:96])[0]
			zero_values = np.where(ground_truth[15,0,:96] == 0)[0]

			print("Middle: difference for non zero values:")
			print(np.mean((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[non_zero_values]))
			print(np.mean(np.abs((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[non_zero_values])))

			print("Middle: difference for zero values:")
			print(np.mean((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[zero_values]))
			print(np.mean(np.abs((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[zero_values])))

			print("Middle: average non zero values:")
			print(np.mean((collected_predictions[15,0,:96])[non_zero_values]))
			print("Middle: average zero values:")
			print(np.mean((collected_predictions[15,0,:96])[zero_values]))

			print("Late")
			print(collected_predictions[28,0,:96]) 
			print(ground_truth[28,0,:96])

			print("Late: top 6 types:")
			print("Predicted: " + str(np.where(np.argsort(collected_predictions[28,0,:96]) >= 90)))
			print("Correct: " + str(np.where(np.argsort(ground_truth[28,0,:96]) >= 90)))

			non_zero_values = np.where(ground_truth[28,0,:96])[0]
			zero_values = np.where(ground_truth[28,0,:96] == 0)[0]

			print("Late: difference for non zero values:")
			print(np.mean((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[non_zero_values]))
			print(np.mean(np.abs((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[non_zero_values])))

			print("Late: difference for zero values:")
			print(np.mean((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[zero_values]))
			print(np.mean(np.abs((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[zero_values])))

			print("Late: average non zero values:")
			print(np.mean((collected_predictions[28,0,:96])[non_zero_values]))
			print("Late: average zero values:")
			print(np.mean((collected_predictions[28,0,:96])[zero_values]))

			# If we sample from a gaussian of predicted latents and decode, are the predicted mutations close to at least any of real mutations?




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

