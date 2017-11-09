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

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

#dataset_path = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
dataset_path = ["/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.regionsize1000mutTumour.1tumour.part1.pickle",
					"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.regionsize1000mutTumour.1tumour.part2.pickle"]

feature_path = "/Users/yulia/Documents/mutational_signatures/dna_features_ryoga/"
model_save_path = "trained_models/model.region_dataset.model{}.tumours{}.mut{}/model.ckpt"

def load_dataset(dataset_path, n_parts = 1000):
    mut_features, region_counts, region_features = load_pickle(dataset_path[0])
    for path in dataset_path[1:n_parts]:
        mutf, counts, regf = load_pickle(path)
        mut_features = np.concatenate((mut_features, mutf[1:])) # the first row is annotation
        region_counts = np.concatenate((region_counts, counts))
        region_features = np.concatenate((region_features, regf))
    return mut_features, region_counts, region_features

def filter_tumours(training_set, labels, unique_tumours, x_tumour_ids, tumor_ids):
	# take a subset of tumours, if needed
	unique_tumours = unique_tumours[tumor_ids]
	samples_to_use = [x in tumor_ids for x in x_tumour_ids]

	training_set = training_set[samples_to_use]
	labels = labels[samples_to_use]
	x_tumour_ids = x_tumour_ids[samples_to_use]

	return training_set, labels, unique_tumours, x_tumour_ids

def correct_predictions(predictions, y_):
	predictions_binary = tf.cast(tf.less(tf.constant(0.5), predictions),tf.int64) # gaussian
	correct_prediction = tf.equal(predictions_binary, tf.cast(y_,tf.int64))
	correct_prediction = tf.cast(correct_prediction, tf.float32)
	return correct_prediction

def mutation_rate_model_gaussian(X, y_, x_tumour_ids, tumour_latents, reg_latent_dim):
	x_dim = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value
	n_tumours = tumour_latents.get_shape()[0].value
	tumour_latent_dim = tumour_latents.get_shape()[1].value

	# region_features = X[(x_dim-97):]
	# possible_mut_types = 

	with tf.name_scope('region_latents'):
		# W_fc1 = weight_variable([x_dim, reg_latent_dim])
		# b_fc1 = bias_variable([reg_latent_dim])
		# y_region = tf.matmul(X, W_fc1) + b_fc1

		prob_nn = init_neural_net_params([x_dim, 500, 200, 200, reg_latent_dim])
		y_region = neural_net(X, prob_nn['weights'], prob_nn['biases'])

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, x_tumour_ids))

	m_t, sigma_t = unpack_gaussian(z_t, reg_latent_dim)

	dist = tf.contrib.distributions.MultivariateNormalDiag(m_t, sigma_t)

	L = dist.log_prob(y_region)

	# how to normalize across other regions and types? !!!!
	#p = tf.minimum(L,0) #!!!!!!! 

	log_normalizer = weight_variable([n_tumours])
	normalize_per_sample = tf.gather_nd(log_normalizer, x_tumour_ids)

	p = tf.minimum(L,-1e-6)
	log_y_prediction = tf.reshape(p, [-1,1])

	# make a different loss !!!! the loss is on the counts
	with tf.name_scope('loss'):
		cross_entropy_all = -tf.reduce_sum(y_ * log_y_prediction + (1-y_)* tf.log(1-tf.exp(log_y_prediction)), reduction_indices=[1]) # gaussian
	cross_entropy = tf.reduce_mean(cross_entropy_all)

	with tf.name_scope('accuracy'):
		predictions = tf.exp(log_y_prediction)
		correct_prediction = correct_predictions(predictions, y_)
	accuracy = tf.reduce_mean(correct_prediction)

	return p, z_t, y_region, cross_entropy, accuracy, predictions, L, normalize_per_sample



def mutation_rate_model_nn(X, y_, x_tumour_ids, tumour_latents, reg_latent_dim, prob_neural_net_params):
	x_dim = X.get_shape()[1].value
	n_samples = X.get_shape()[0].value

	with tf.name_scope('region_latents'):
		W_fc1 = weight_variable([x_dim, reg_latent_dim])
		b_fc1 = bias_variable([reg_latent_dim])
		y_region = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)

	z_t = tf.squeeze(tf.gather_nd(tumour_latents, x_tumour_ids))

	prob_nn = init_neural_net_params(prob_neural_net_params)

	p = neural_net(tf.concat([X, z_t], 1), prob_nn['weights'], prob_nn['biases'])
	log_y_prediction = tf.reshape(p, [-1,1])

	# make a different loss !!!! the loss is on the counts
	with tf.name_scope('loss'):
		cross_entropy_all = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=log_y_prediction) # neural net
	cross_entropy = tf.reduce_mean(cross_entropy_all)

	with tf.name_scope('accuracy'):
		predictions = tf.sigmoid(log_y_prediction)
		correct_prediction = correct_predictions(predictions, y_)
	accuracy = tf.reduce_mean(correct_prediction)

	return p, z_t, y_region, cross_entropy, accuracy, predictions


def make_training_set(mut_features, region_counts, region_features, trinuc):
	feature_names = np.asarray(mut_features[0]).ravel()
	mut_features = mut_features[1:]
	region_features = region_features[:,1:]

	if not(mut_features.shape[0] == region_counts.shape[0] and region_counts.shape[0] == region_features.shape[0]):
		raise Exception("Dataset is incorrect: all parts should contain the same number of samples")

	mut_annotation = mut_features[:,:3]
	tumour_names = np.asarray(mut_annotation[:,np.where(feature_names == "Tumour")]).ravel()
	unique_tumours = np.unique(tumour_names)
	tumour_ids = np.squeeze(np.asarray([np.where(unique_tumours ==name) for name in tumour_names]))[:,np.newaxis]
	
	mut_types = mut_features[:,get_col_indices_trinucleotides([feature_names], trinuc)]

	n_samples = mut_features.shape[0]
	regionsize = region_features.shape[1]
	n_region_features = region_features.shape[2]
	n_mut_types = mut_types.shape[1]

	region_features = region_features.reshape((n_samples, regionsize*n_region_features))
	region_counts = region_counts[:,np.newaxis]

	dataset = np.concatenate((mut_types, region_features), axis=1)

	return dataset, region_counts, unique_tumours, tumour_ids

def make_model(x_dim, unique_tumours, z_latent_dim, model_type, model_save_path):
	x = tf.placeholder(tf.float32, [None, x_dim])
	x_tumour_ids = tf.placeholder(tf.int32, [None, 1])
	y_ = tf.placeholder(tf.float32, [None, 1])

	n_unique_tumours = len(unique_tumours)
	initial_tumour_latents = tf.abs(weight_variable([z_latent_dim,1]))
	tumour_latents = tf.transpose(tf.tile(initial_tumour_latents, [1,n_unique_tumours]))

	if model_type == "gaussian":
		log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions, L, normalize_per_sample =  mutation_rate_model_gaussian(x, y_, x_tumour_ids, tumour_latents, reg_latent_dim)
	else:
		log_y_prediction, z_t, y_region, cross_entropy, accuracy, predictions =  mutation_rate_model_nn(x, y_, x_tumour_ids, tumour_latents, 
				reg_latent_dim, prob_neural_net_params = prob_neural_net)

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

	save_graph()
	saver = tf.train.Saver()

	tf_vars = [x, x_tumour_ids, y_, log_y_prediction]
	metrics = [cross_entropy, accuracy]
	meta = [train_step, saver]
	extra = [z_t, y_region, predictions]

	if model_type == "gaussian":
		extra.append(L)

	return tf_vars, metrics, meta, extra

def train(train_data, test_dict, tf_vars, n_epochs, batch_size, model_save_path):
	x, x_tumour_ids, y_, log_y_prediction = tf_vars

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for j in range(n_epochs):
			for i in range(x_train.shape[0] //batch_size+1):
				x_batch, y_batch, x_tumour_ids_batch = get_batch(train_data, batch_size, i)
				batch_dict = {x: x_batch, y_: y_batch, x_tumour_ids: x_tumour_ids_batch}

				if i % 5 == 0:
					train_cross_entropy = cross_entropy.eval(feed_dict=batch_dict)
					print('Epoch %d.%d: training cross_entropy %g' % (j, i, train_cross_entropy))
					print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
					print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))
					#print((tf.sigmoid(log_y_prediction)).eval(feed_dict=test_dict).ravel()[:10]) #neural net
					print((tf.sigmoid(log_y_prediction)).eval(feed_dict=test_dict).ravel()[:10])
					print(y_.eval(feed_dict=test_dict).ravel()[:10])
					#print(L.eval(feed_dict=test_dict).ravel()[:10])
					print((log_y_prediction).eval(feed_dict=test_dict).ravel()[:10])
					print(z_t.eval(feed_dict=batch_dict)[0,:10])
				train_step.run(feed_dict=batch_dict)
			
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
	parser.add_argument('-e','--epochs', help='number of epochs', default=1000)
	parser.add_argument('-b','--batch', help='batch size', default=500)
	parser.add_argument('--model', help='Model type: gaussian likelihood or neural net', default='gaussian')
	#parser.add_argument('--loss', help = "loss type: poisson or mean_squared", default="poisson")

	args = parser.parse_args()
	test_mode = args.test
	n_tumours = args.tumours
	n_mut = args.mut
	n_epochs = int(args.epochs)
	batch_size = int(args.batch)
	model_type = args.model
	#latent_dimension = args.latents

	# model params
	reg_latent_dim = 10
	prob_neural_net = [200, 200, 1]
	z_latent_dim = reg_latent_dim * 2

	print("Loading dataset...")
	mut_features, region_counts, region_features = load_dataset(dataset_path)
	trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))

	training_set, labels, unique_tumours, x_tumour_ids = make_training_set(mut_features, region_counts, region_features, trinuc)

	if n_tumours is not None:
		tumor_ids = list(range(int(n_tumours)))
		training_set, labels, unique_tumours, x_tumour_ids = filter_tumours(training_set, labels, unique_tumours, x_tumour_ids, tumor_ids)
	else:
		n_tumours = len(unique_tumours)

	# take a subset of mutations, if needed
	if n_mut is not None:
		n_mut = int(n_mut)
		training_set = training_set[:n_mut]
		labels = labels[:n_mut]
		x_tumour_ids = x_tumour_ids[:n_mut]
	else:
		n_mut = training_set.shape[0]

	print("Processing {} mutations from {} tumour(s) ...".format(training_set.shape[0], n_tumours))

	x_dim = training_set.shape[1]
	prob_neural_net = [x_dim + z_latent_dim] + prob_neural_net

	tf.reset_default_graph()
	model_save_path = model_save_path.format(model_type, n_tumours, n_mut)
	os.makedirs(model_save_path, exist_ok=True)

	# Split dataset into train / test
	x_train, x_test, y_train, y_test, x_tumour_ids_train, x_tumour_ids_test = model_selection.train_test_split(training_set, labels, x_tumour_ids, test_size=0.2, random_state=1991)

	tf_vars, metrics, meta, extra = make_model(x_dim, unique_tumours, z_latent_dim, model_type, model_save_path)

	x, x_tumour_ids, y_, log_y_prediction = tf_vars
	cross_entropy, accuracy = metrics
	train_step, saver = meta
	if model_type == "gaussian":
		 z_t, y_region, predictions, L = extra
	else: 
		z_t, y_region, predictions = extra

	test_dict = {x: x_test, y_: y_test, x_tumour_ids: x_tumour_ids_test}

	if not test_mode:
		print("Optimizing...")
		train_data = [x_train, y_train, x_tumour_ids_train]
		tf_vars = [x, x_tumour_ids, y_, log_y_prediction]

		train(train_data, test_dict, tf_vars, n_epochs, batch_size, model_save_path)
	else:
		if not os.path.exists(model_save_path):
			print("Model folder not found: " + model_save_path)
			exit()

		with tf.Session() as sess:
			saver.restore(sess, model_save_path)

			print('test cross_entropy %g' % cross_entropy.eval(feed_dict=test_dict))
			print('test accuracy %g' % accuracy.eval(feed_dict=test_dict))

			print("Mean prediction")
			pred = predictions.eval(feed_dict=test_dict).ravel()
			print("1: " + str(np.mean(pred[y_test.ravel()])))
			print("0: " + str(np.mean(pred[np.logical_not(y_test.ravel())])))

			correct_prediction = correct_predictions(predictions, y_).eval(feed_dict=test_dict).ravel()
			
			print("Mean accuracy within a class")
			print("1: " + str(np.mean(correct_prediction[y_test.ravel()])))
			print("0: " + str(np.mean(correct_prediction[np.logical_not(y_test.ravel())])))

			print("Tumour representation std")
			print(np.std(z_t.eval(feed_dict=test_dict), axis=0))

			if model_type == "gaussian":
				print("Likelihood of region representation:")
				print("1: " + str(np.mean(L.eval(feed_dict=test_dict)[y_test.ravel()])))
				print("0: " + str(np.mean(L.eval(feed_dict=test_dict)[np.logical_not(y_test.ravel())])))


# add time component (z_t comes from RNN over time)
# make a training summary
# version with cnn to make a summary of region features
# try relu versus tanh
# try different architectures of neural net (number of layers, number of units)

# predict the 96-vector directly from the region (softmax layer on top)
# try mean squared loss
# make proper normalization in gaussian model

