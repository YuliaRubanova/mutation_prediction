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

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

FLAGS = None

if os.path.isdir("/home/yulia/"):
	DIR = "/home/yulia/mnt/"
else:
	DIR = "/Users/yulia/Documents/mutational_signatures/"

DEF_FEATURE_PATH = DIR + "/dna_features_ryoga/"
n_mut_types = 96

# notes: use the RNN with latent layer to predict the batch of mutation types + features

#dataset_path = DIR + "/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
mut_dataset_path = [DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part*.pickle"]
feature_path = DIR + "/dna_features_ryoga/"
file_name = os.path.basename(__file__)[:-3]
model_dir = "trained_models/model." + file_name + ".tumours{}.mut{}.timesteps{}/"
# dataset_with_annotation = DIR + "mutation_prediction_data/region_dataset.small.over_time.annotation.hdf5"
# datasets with at most 10000 mutations per tumour
dataset_with_annotation = DIR + "/mutation_prediction_data/region_dataset.mutTumour10000.region_size{region_size}.ID{{id}}.over_time.annotation.hdf5"

session_conf = tf.ConfigProto(gpu_options=gpu_options)
# session_conf = tf.ConfigProto(
#     device_count={'CPU' : 20, 'GPU' : 0},
#     allow_soft_placement=True,
#     log_device_placement=False
# )

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

	encoder_data = init_neural_net_params([num_features] + [150, 150, reg_latent_dim], stddev = 0.01, bias = 0.01)
	next_state_nn = init_neural_net_params([reg_latent_dim + lstm_size + 1, 150, 150, lstm_size], stddev = 0.01, bias = 0.01)
	predictor_mut_type = init_neural_net_params([lstm_size, 150, 150, n_mut_types], stddev = 0.01, bias = 0.01)
	predictor_features = init_neural_net_params([lstm_size, 150, 150, num_features-n_mut_types], stddev = 0.01, bias = 0.01)

	type_prob_sum = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	feature_prob_sum = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)
	cross_entropy = tf.Variable(tf.constant(0.0, shape=[1]), trainable=False)

	predictions = []

	for batch, next_batch, vaf in zip(tf.unstack(X)[:-1], tf.unstack(X)[1:], tf.unstack(time_estimates)):
		# !!!! no z_t
		#input = tf.concat([ tf.reshape(batch, [-1]), z_t], 0)

		batch = tf.reshape(batch, [-1,num_features])

		latents = neural_net(batch, encoder_data['weights'], encoder_data['biases'])
		latents = tf.reshape(latents, [sequences_per_batch_tf, batch_size_per_tp, reg_latent_dim])
		latents = tf.reduce_sum(latents, axis = 1)

		next_state = neural_net(tf.concat([tf.squeeze(prev_hidden_state), latents, vaf], 1), next_state_nn['weights'], next_state_nn['biases'])
		prev_hidden_state = next_state

		predicted_mut_types = neural_net(next_state, predictor_mut_type['weights'], predictor_mut_type['biases'])
		type_prob = compute_mut_type_prob(next_batch, n_mut_types, predicted_mut_types, vaf=vaf, time_series_lengths=time_series_lengths)
		type_prob_sum = tf.add(type_prob_sum, type_prob)

		b = tf.constant([batch_size_per_tp], dtype=tf.int32)
		# cross entropy -- should give the same result
		tiled_predictions2 = tf.tile(predicted_mut_types, [b[0],1])
		cross_entropy_all = tf.nn.softmax_cross_entropy_with_logits(labels=next_batch[:,:,:n_mut_types], logits=tiled_predictions2) # neural net
		print(cross_entropy_all.get_shape())

		cross_entropy_all = tf.reduce_mean(cross_entropy_all) * tf.reduce_mean(tf.to_float(tf.greater(tf.squeeze(vaf),tf.constant(0.0))) / (time_series_lengths - 1))
		cross_entropy = tf.add(cross_entropy, tf.reduce_mean(cross_entropy_all))

		# # prob
		# # correct :tada: !!!!
		# hello = tf.transpose(next_batch[:,:,:n_mut_types], perm = [1,0,2])
		# dist = tf.contrib.distributions.Multinomial(total_count=tf.reduce_sum(hello,axis=2), logits=predicted_mut_types, validate_args = True)
		# type_prob = tf.expand_dims(dist.log_prob(hello),1) * tf.to_float(tf.greater(tf.squeeze(vaf),tf.constant(0.0))) / (time_series_lengths - 1)
		# type_prob_sum = tf.add(type_prob_sum, type_prob)
		# #print(type_prob_sum.get_shape())

		predicted_features = neural_net(next_state, predictor_features['weights'], predictor_features['biases'])
		# features come in a batch of 100. axis = 1 -- index over mutations in a batch of 100 mutations
		# axis = 0 -- index over tumours in a tumour batch 
		features = tf.reduce_sum(next_batch[:,:,n_mut_types:], axis=1)
		num_region_features = features.get_shape()[1].value

		# features are binarized -- model as binomial
		# we want to predict the features in a batch of 100 mutations
		# model it the stupid way -- we don't want to sample each mutation separately
		# instead just compute the sum over all features -- how many mutations out of batch_size_per_tp have this feature on

		c = tf.stack([tf.transpose(features), tf.transpose(predicted_features)], axis=2)
		#resulting axis: 0) features 1) tumours in a batch 2) features + predicted features pairs
		
		c = tf.transpose(c, perm=[2, 1, 0 ])
		#resulting axis: 0) features + predicted features pairs 1) tumours in a batch 2) features

		c = tf.transpose(tf.reshape(c, [2, -1]))
		#resulting axis:  0) tumours in a batch + features 1) features + predicted features pairs

		def apply_bernoulli(x):
			dat, logit = tf.unstack(x)
			dat = tf.cast(tf.cast(dat, tf.int32),tf.float32)
			dist = tf.contrib.distributions.Binomial(total_count=tf.cast(batch_size_per_tp, tf.float32), logits = logit, validate_args = True)
			return(dist.log_prob(dat))
		
		# iterate over all features in all tumours
		d = tf.map_fn(apply_bernoulli, c, dtype=tf.float32)
		feature_prob = tf.reshape(d, [-1, num_region_features])
		# take sum over all features
		feature_prob = tf.expand_dims(tf.reduce_sum(feature_prob, axis = 1),1)
		feature_prob = tf.multiply(feature_prob, tf.to_float(tf.greater(vaf,tf.constant(0.0))))
		feature_prob = tf.divide(feature_prob, (time_series_lengths - 1))
		feature_prob_sum = tf.add(feature_prob_sum, feature_prob)

		predicted_mut_types = tf.nn.softmax(predicted_mut_types, axis = 1)
		predicted_features = tf.sigmoid(predicted_features)

		current_tp_prediction = tf.concat([predicted_mut_types, predicted_features], 1)
		predictions.append(tf.expand_dims(current_tp_prediction,0))

	predictions = tf.concat(predictions, 0)
	return predictions, tf.reduce_mean(type_prob_sum), tf.reduce_mean(feature_prob_sum), cross_entropy

def make_model(num_features, n_unique_tumours, z_latent_dim, time_steps, batch_size_per_tp, lstm_size, model_save_path, adam_rate = 1e-3):
	x = tf.placeholder(tf.float32, [time_steps, None, batch_size_per_tp, num_features])
	tumour_id = tf.placeholder(tf.int32, [None, 1])
	time_estimates = tf.placeholder(tf.float32, [time_steps,None,1])
	time_series_lengths = tf.placeholder(tf.float32, [None,1])
	sequences_per_batch_tf = tf.placeholder(dtype=tf.int32)

	initial_tumour_latents = tf.abs(tf.concat([weight_variable([z_latent_dim//2,1]), weight_variable([z_latent_dim//2,1], mean=1)], axis = 0))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	predictions, type_prob_sum, feature_prob_sum, ce = \
		mutation_rate_model_over_time(x, time_estimates, time_series_lengths, tumour_id, tumour_latents, lstm_size, reg_latent_dim, time_steps, batch_size_per_tp, sequences_per_batch_tf)

	cost = -type_prob_sum - feature_prob_sum

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(adam_rate).minimize(cost)
		#train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_sum)

	save_graph()
	saver = tf.train.Saver()

	tf_vars = [x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions]
	metrics = [cost, type_prob_sum, feature_prob_sum]
	meta = [train_step, saver]
	extra = [ce]

	return tf_vars, metrics, meta, extra

def train(tf_vars, metrics, meta, extra, n_epochs, model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps):
	x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions = tf_vars
	cost, type_prob_sum, feature_prob_sum = metrics
	train_step, saver = meta
	ce = extra[0]

	test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, annot, features = read_tumour_data(tumour_files_test, binarize_features=True)
	x_test, tumours_test, time_estimates_test = make_batches_over_time(test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, batch_size_per_tp, n_unique_tumours_test, n_timesteps)

	test_data = [x_test, tumours_test, time_estimates_test]
	time_steps = x_test.shape[1]

	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train, binarize_features=True)
	x_train, tumours_train, time_estimates_train = make_batches_over_time(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, n_unique_tumours, n_timesteps)
	sequences_per_batch = x_train.shape[2]

	print("Optimizing...")
	with tf.Session(config=session_conf) as sess:
		sess.run(tf.global_variables_initializer())

		# print("log probabilities of true distribution")
		# print(np.mean(collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, last_type_prob)))
		# print("mut_types")
		# print(collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, last_mut_types)[0,0])

		# print("predicted_mut_types")
		# print(collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, last_predicted_mut_types)[0,0])
		
		print('Initial')
		print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, cost))
		print('test cost %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, cost))

		print('train type_prob_sum %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, type_prob_sum))
		print('test type_prob_sum %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, type_prob_sum))

		print('train ce %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, ce))
		print('test ce %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, ce))

		# print('train feature_prob_sum %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, feature_prob_sum))
		# print('test feature_prob_sum %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, feature_prob_sum))


		for j in range(n_epochs):
				for k in range(len(tumour_files_train) // n_tumours_per_batch+1):
					if len(tumour_files_train) > n_tumours_per_batch:
						training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train, n_tumours_per_batch, k, binarize_features=True)
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

					print('Epoch %d.%d:' % (j, k))
					print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, cost))
					print('test cost %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, cost))

					print('train type_prob_sum %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, type_prob_sum))
					print('test type_prob_sum %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, type_prob_sum))

					# print('train feature_prob_sum %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, feature_prob_sum))
					# print('test feature_prob_sum %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, feature_prob_sum))

					print('train ce %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, ce))
					print('test ce %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, ce))

					# print("log probabilities of true distribution")
					# print(np.mean(collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, last_type_prob)))
					# print("mut_types")
					# print(collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, last_mut_types)[0,0])

					# print("predicted_mut_types")
					# print(collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, last_predicted_mut_types)[0,0])
					
					if j % 5 == 0:
						model_save_path_tmp = "trained_models/tmp/model.region_dataset.model{}.tumours{}.mut{}.ckpt".format(model_type, n_tumours, n_mut)
						save_path = saver.save(sess, model_save_path_tmp)
						print("Model saved in file: %s" % save_path)

		print('test cost %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, cost))
		print('test type_prob_sum %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, type_prob_sum))
		print('test feature_prob_sum %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, feature_prob_sum))

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
	sequences_per_batch = 10
	n_tumours_per_batch = 100 # tumours in tumour_batch
	# size of region latents
	reg_latent_dim = 120
	# size of lstm latent state
	lstm_size = 150
	z_latent_dim = reg_latent_dim * 2
	
	mut_features, unique_tumours, n_tumours, n_mut, available_tumours, num_features, n_unique_tumours = \
		load_filter_dataset(mut_dataset_path, feature_path, dataset_with_annotation, region_size, n_tumours, n_mut)

	print("Processing {} mutations from {} tumour(s) ...".format(n_mut, n_unique_tumours))

	model_dir, model_save_path = prepare_model_dir(sys.argv, model_dir, __file__, [n_tumours, n_mut, n_timesteps])
	
	tf_vars, metrics, meta, extra = make_model(num_features, n_unique_tumours, z_latent_dim, n_timesteps, batch_size_per_tp, lstm_size, model_save_path, adam_rate = adam_rate)

	x, tumour_id, time_estimates, time_series_lengths, sequences_per_batch_tf, predictions = tf_vars
	cost, type_prob_sum, feature_prob_sum = metrics
	train_step, saver = meta
	# = extra

	# # Split dataset into train / test
	tumour_files_train, tumour_files_test = model_selection.train_test_split(available_tumours, test_size=0.2, random_state = 1991)

	if not test_mode:
		train(tf_vars, metrics, meta, extra, n_epochs , model_save_path, tumour_files_train, tumour_files_test, batch_size_per_tp, sequences_per_batch, n_timesteps)
	else:
		if not os.path.exists(model_save_path):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)
			training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, annot, features = read_tumour_data(tumour_files_train, binarize_features=True)
			x_train, tumours_train, time_estimates_train = make_batches_over_time(training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, n_unique_tumours, n_timesteps)

			test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, annot, features = read_tumour_data(tumour_files_test, binarize_features=True)
			x_test, tumours_test, time_estimates_test = make_batches_over_time(test_set, labels_test, n_unique_tumours_test, tumour_ids_test, time_estimates_test, batch_size_per_tp, n_unique_tumours_test, n_timesteps)

			print('train cost %g' % evaluate_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, cost))
			print('test cost %g' % evaluate_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, cost))

			# Plots
			# Test
			# collected_predictions = collect_on_each_tumour(x_test, tumours_test, time_estimates_test, tf_vars, predictions)
			# collected_predictions = np.array(np.squeeze(collected_predictions))
			# predicted_types = collected_predictions[:,:,:n_mut_types]
			# predicted_features = collected_predictions[:,:,n_mut_types:]

			# ground_truth = np.squeeze(x_test)
			# ground_truth_types = np.sum(ground_truth[:,:,:,:n_mut_types], axis=2)
			# ground_truth_types /= batch_size_per_tp
			# ground_truth_features = np.sum(ground_truth[:,:,:,n_mut_types:], axis=2)
			# ground_truth_features /= batch_size_per_tp


			# plot_types_over_time(ground_truth_types, ylabel = "Mutation types")
			# plot_types_over_time(predicted_types, plot_name="tmp2.pdf", ylabel = "Mutation types")

			# plot_types_over_time(ground_truth_features, plot_name="tmp3.pdf", ylabel = "Region features")
			# plot_types_over_time(predicted_features, plot_name="tmp4.pdf", ylabel = "Region features")

			# Train
			# collected_predictions = collect_on_each_tumour(x_train, tumours_train, time_estimates_train, tf_vars, predictions)
			# collected_predictions = np.array(np.squeeze(collected_predictions))
			# predicted_types = collected_predictions[:,:,:n_mut_types]
			# predicted_features = collected_predictions[:,:,n_mut_types:]

			# ground_truth = np.squeeze(x_train)
			# ground_truth_types = np.sum(ground_truth[:,:,:,:n_mut_types], axis=2)
			# ground_truth_types /= batch_size_per_tp
			# ground_truth_features = np.sum(ground_truth[:,:,:,n_mut_types:], axis=2)
			# ground_truth_features /= batch_size_per_tp

			# plot_types_over_time(ground_truth_types, ylabel = "Mutation types")
			# plot_types_over_time(predicted_types, plot_name="tmp2.pdf", ylabel = "Mutation types")

			# plot_types_over_time(ground_truth_features, plot_name="tmp3.pdf", ylabel = "Region features")
			# plot_types_over_time(predicted_features, plot_name="tmp4.pdf", ylabel = "Region features")








			# Stats -- does not work -- just copied!
			# print("Type comparison: ground truth vs predicted")
			# print(ground_truth_types)
			# print(predicted_types[0][0])
			# print("Feature comparison:")
			# print(ground_truth_features)
			# print(predicted_features[0][0])

			# print("Difference across predictions of types within a tumour:")
			# print(predicted_types.shape)
			# print("Early")
			# print(predicted_types[0,0])
			# print(predicted_types[0,1])
			# print("Mid")
			# print(predicted_types[15,0])
			# print(predicted_types[15,1])
			# print("Late")
			# print(predicted_types[28,0])
			# print(predicted_types[28,1])

			# print("std")
			# print(np.apply_along_axis(np.std, 0, collected_predictions[:,0,:96]))
			# print(np.apply_along_axis(np.std, 0, collected_predictions[:,1,:96]))

			# def find_max_diff(x):
			# 	return max(x) - min(x)

			# print("max diff")
			# print(np.apply_along_axis(find_max_diff, 0, collected_predictions[:,0,:96]))
			# print(np.apply_along_axis(find_max_diff, 0, collected_predictions[:,1,:96]))

			# print("Difference in predictions across tumours:")
			# print("Early")
			# print(collected_predictions[0,0,:96] - collected_predictions[0,1,:96])

			# print("Middle")
			# print(collected_predictions[15,0,:96] - collected_predictions[15,1,:96])

			# print("Late")
			# print(collected_predictions[28,0,:96] - collected_predictions[28,1,:96])

			# How far are the predicted mutations from real ones (at least some in the batch?)
			# print("Difference with the ground truth:")
			# ground_truth = ground_truth[1:] # The value of the first time point is not predicted
			
			# print("Early: top 6 types:")
			# print("Predicted: " + str(np.where(np.argsort(ground_truth_types[0,0]) >= 90)))
			# print("Correct: " + str(np.where(np.argsort(ground_truth_types[0,0]) >= 90)))

			# print("Early: difference for non zero values:")
			# non_zero_values = np.where(ground_truth_types[0,0])[0]
			# zero_values = np.where(ground_truth_types[0,0] == 0)[0]

			# print(np.mean((predicted_types[0,0] - ground_truth_types[0,0])[non_zero_values]))
			# print(np.mean(np.abs((predicted_types[0,0] - ground_truth_types[0,0])[non_zero_values])))

			# print("Early: difference for zero values:")
			# print(np.mean((predicted_types[0,0] - ground_truth_types[0,0])[zero_values]))
			# print(np.mean(np.abs((predicted_types[0,0] - ground_truth_types[0,0])[zero_values])))

			# print("Early: average non zero values:")
			# print(np.mean((predicted_types[0,0])[non_zero_values]))
			# print("Early: average zero values:")
			# print(np.mean((predicted_types[0,0])[zero_values]))

			# print("Middle")
			# print(collected_predictions[15,0,:96]) 
			# print(ground_truth[15,0,:96])

			# print("Middle: top 6 types:")
			# print("Predicted: " + str(np.where(np.argsort(collected_predictions[15,0,:96]) >= 90)))
			# print("Correct: " + str(np.where(np.argsort(ground_truth[15,0,:96]) >= 90)))

			# non_zero_values = np.where(ground_truth[15,0,:96])[0]
			# zero_values = np.where(ground_truth[15,0,:96] == 0)[0]

			# print("Middle: difference for non zero values:")
			# print(np.mean((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[non_zero_values]))
			# print(np.mean(np.abs((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[non_zero_values])))

			# print("Middle: difference for zero values:")
			# print(np.mean((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[zero_values]))
			# print(np.mean(np.abs((collected_predictions[15,0,:96] - ground_truth[15,0,:96])[zero_values])))

			# print("Middle: average non zero values:")
			# print(np.mean((collected_predictions[15,0,:96])[non_zero_values]))
			# print("Middle: average zero values:")
			# print(np.mean((collected_predictions[15,0,:96])[zero_values]))

			# print("Late")
			# print(collected_predictions[28,0,:96]) 
			# print(ground_truth[28,0,:96])

			# print("Late: top 6 types:")
			# print("Predicted: " + str(np.where(np.argsort(collected_predictions[28,0,:96]) >= 90)))
			# print("Correct: " + str(np.where(np.argsort(ground_truth[28,0,:96]) >= 90)))

			# non_zero_values = np.where(ground_truth[28,0,:96])[0]
			# zero_values = np.where(ground_truth[28,0,:96] == 0)[0]

			# print("Late: difference for non zero values:")
			# print(np.mean((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[non_zero_values]))
			# print(np.mean(np.abs((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[non_zero_values])))

			# print("Late: difference for zero values:")
			# print(np.mean((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[zero_values]))
			# print(np.mean(np.abs((collected_predictions[28,0,:96] - ground_truth[28,0,:96])[zero_values])))

			# print("Late: average non zero values:")
			# print(np.mean((collected_predictions[28,0,:96])[non_zero_values]))
			# print("Late: average zero values:")
			# print(np.mean((collected_predictions[28,0,:96])[zero_values]))


