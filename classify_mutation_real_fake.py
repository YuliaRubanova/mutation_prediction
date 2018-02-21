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
from scipy.stats.stats import pearsonr
import sklearn

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
model_save_path = model_dir + "model.ckpt"
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

	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours, region_size)
	print("Optimizing...")
	# config=tf.ConfigProto(gpu_options=gpu_options)
	# config=session_conf
	with tf.Session(config=session_conf) as sess:
		sess.run(tf.global_variables_initializer())
		for j in range(n_epochs):
			for k in range(len(available_tumours) // n_tumours_per_batch+1):
				if n_tumours_per_batch < len(available_tumours):
					training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours, region_size, n_tumours_per_batch, k)
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


def non_zero(data):
	return np.sum(data != 0) / np.prod(data.shape)

def correlation_binary(x_input, y_input):
	# see https://en.wikipedia.org/wiki/Phi_coefficient
	x = np.copy(x_input)
	y = np.copy(y_input)

	x[x > 0] = 1
	y[y > 0] = 1

	n1d = np.sum(x == 1)
	n0d = np.sum(x == 0)

	nd1 = np.sum(y == 1)
	nd0 = np.sum(y == 0)

	n11 = np.sum((x == 1) & (y == 1))
	n10 = np.sum((x == 1) & (y == 0))
	n01 = np.sum((x == 0) & (y == 1))
	n00 = np.sum((x == 0) & (y == 0))

	phi = (n11 * n00 - n10 * n01) / np.sqrt(n1d * n0d * nd0 * nd1)
	return phi

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
	prop_non_zero = non_zero(real_mutations)
	print("Proportion of non-zero values: "+ str(prop_non_zero))

	# identify useless features (with more zeros than others)
	features = np.reshape(feature_names[96:], [region_size, -1])[50]
	print("Non-zero features")

	if (False):
		print("Per position:")
		for i in range(real_mutations.shape[1]):
			print(str(i) + ": " + str(non_zero(real_mutations[:,i,:])))

		print("\n\n")
		print("Per feature:")
		for i in range(real_mutations.shape[2]):
			print(features[i])
			print(non_zero(real_mutations[:,50,i]))

	print(features)
	print(np.apply_along_axis(lambda x: np.mean(x != 0), 0, real_mutations[:,50,:].astype(float)))

	if (False):
		print("\n\n")
		print("Feature variability across positions:")
		for i in range(real_mutations.shape[2]):
			print(features[i])
			print(np.std(real_mutations[:,:,i]))

	print("\n\n")

	if (False):
		print("Correlation between type and features")
		for i in range(mut_types.shape[1]):
			print("mut_type " + feature_names[i])
			# binary correlation and pearson correlation give the same result
			#correlation = [pearsonr(mut_types[:,i], real_mutations[:,50,j])[0] for j in range(real_mutations.shape[2])]
			correlation = [correlation_binary(mut_types[:,i], real_mutations[:,50,j]) for j in range(real_mutations.shape[2])]
			maximum_val = np.argmax(np.array(np.abs(correlation)))
			print("max correlation: " + str(correlation[maximum_val]) + " " + features[maximum_val])
			#print(correlation)


	print("# non-zero values per sample")
	print(np.mean([len(real_mutations[i,50][real_mutations[i,50] != 0]) for i in range(real_mutations.shape[0])]))

	# plot only features at the location of mutation (all feature types for all mutations)
	# put the dimension over features first
	non_zero_features = np.transpose(real_mutations[:,50,:]).tolist()

	# plotting only non-zero values which make up a very small fraction of the data!
	for i in range(len(non_zero_features)):
		non_zero_features[i] = [x for x in non_zero_features[i] if x != 0]

	plot_boxplot(non_zero_features, "region_features.pdf", features,
		text_="Plot of non-zero features. Proportion of non-zero: " + str(round(prop_non_zero,6)))

	if (False):
	# plot distribution over features for each type
		percentage_non_zero_all_types = []
		type_names = []
		for mtype in range(6):
			mtype_name = feature_names[mtype*16][:3]
			type_names.append(mtype_name)
			print(mtype_name)

			mut_types_current = np.sum(mut_types[:,mtype*16:(mtype+1)*16], axis=1)
			mut_with_type = real_mutations[np.where(mut_types_current == 1)[0]]
			print(mut_with_type.shape)

			prop_non_zero_type = non_zero(mut_with_type)
			print("Proportion of non-zero: " + str(prop_non_zero_type))

			mut_with_type = np.transpose(mut_with_type[:,50,:])

			# plotting only non-zero values which make up a very small fraction of the data!
			non_zero_features_type = mut_with_type.tolist()
			for i in range(len(non_zero_features_type)):
				non_zero_features_type[i] = [x for x in non_zero_features_type[i] if x != 0]

			plot_boxplot(non_zero_features_type, "region_features" + mtype_name + ".pdf", features,
				text_="Plot of non-zero features. Proportion of non-zero: " + str(round(prop_non_zero_type,6)))

			plot_bars([non_zero(x) for x in mut_with_type], "percentage_non_zero" + mtype_name + ".pdf", features, 
				ylabel = "Percentage of non-zero values", ylim=[0, 0.2],
				text_="Type " + mtype_name + ". # mutations: " + str(int(mut_with_type.shape[1])))

			correlation = [correlation_binary(mut_types_current, real_mutations[:,50,j]) for j in range(real_mutations.shape[2])]
			maximum_val = np.argmax(np.array(np.abs(correlation)))
			print("max correlation: " + str(correlation[maximum_val]) + " " + features[maximum_val])

			non_zero_first_feature = (real_mutations[:,50,0] > 0).astype(float)
			random_types = np.random.multinomial(np.sum(mut_types_current), [1/mut_types_current.shape[0]]*mut_types_current.shape[0] )
			random_features = np.random.multinomial(np.sum(non_zero_first_feature), [1/non_zero_first_feature.shape[0]]*non_zero_first_feature.shape[0] )
			print("Correlation by random: " + str(correlation_binary(random_types, random_features)))

			percentage_non_zero_all_types.append([non_zero(x) for x in mut_with_type])

	if (False):
		# compare distr over types in different tumours
		for id_ in np.unique(real_tumour_ids):
			print("Tumour " + str(id_))

			mut_tumour = real_mutations[np.where(real_tumour_ids == id_)[0]]

			prop_non_zero_type = non_zero(mut_tumour)
			print("Proportion of non-zero: " + str(prop_non_zero_type))

			mut_tumour = np.transpose(mut_tumour[:,50,:])

			plot_bars([non_zero(x) for x in mut_tumour], "percentage_non_zero.tumour" + str(id_) + ".pdf", features, 
				ylabel = "Percentage of non-zero values", ylim=[0, 0.4],
				text_="Tumour " + str(id_) + ". # mutations: " + str(int(mut_tumour.shape[1])))

	if False:
		# Plot number of non-zero values for different VAFs
		print("VAF range " + "> 1")
		mut_vaf = real_mutations[np.where(real_mut_vaf > 1)[0]]
		prop_non_zero_vaf = non_zero(mut_vaf)
		print("Proportion of non-zero: " + str(prop_non_zero_vaf))

		mut_vaf = np.transpose(mut_vaf[:,50,:])

		plot_bars([non_zero(x) for x in mut_vaf], "percentage_non_zero.vaf>1.pdf", features, 
			ylabel = "Percentage of non-zero values", ylim=[0, 0.2],
			text_="VAF > 1. # mutations: " + str(int(mut_vaf.shape[1])))

		for vaf_range in np.arange(0.9,0,-0.1):
			vaf_range = round(vaf_range, 2)
			range_str = "[" + str(vaf_range) + "," + str(round(vaf_range+0.1,2)) + "]"
			print("VAF range " + range_str)
			mut_vaf = real_mutations[np.where((real_mut_vaf > vaf_range) & (real_mut_vaf < (vaf_range+0.1)))[0]]
			prop_non_zero_vaf = non_zero(mut_vaf)
			print("Proportion of non-zero: " + str(prop_non_zero_vaf))

			mut_vaf = np.transpose(mut_vaf[:,50,:])

			plot_bars([non_zero(x) for x in mut_vaf], "percentage_non_zero.vaf" + range_str + ".pdf", 
				features, ylabel = "Percentage of non-zero values", ylim=[0, 0.25],
				text_="VAF " + range_str + ". # mutations: " + str(int(mut_vaf.shape[1])))


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
	model_save_path = model_save_path.format(n_tumours, n_mut)
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

			training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = read_tumour_data(available_tumours, region_size)
			x_train, x_test, y_train, y_test, x_tumour_ids_train, x_tumour_ids_test = train_test_split([training_set, labels, tumour_ids], split_by = mut_vaf, test_size=0.2)
			test_dict = {x: x_test, y_: y_test, x_tumour_ids: x_tumour_ids_test}

			real_data = training_set[np.where(labels.ravel() > 0)[0]]
			real_data = np.reshape(real_data[:,96:], [real_data.shape[0], region_size, -1])

			print("Total real mutations: " + str(real_data.shape[0]))
			print("Open chromatin: " + str(round(np.mean(real_data[:,50,0]),6)))
			print("Transcribed: " + str(round(np.mean(real_data[:,50,1]),6)))
			print("Strand: " + str(round(np.mean(real_data[:,50,2]),6))) 

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
			#print(np.mean(np.std(z_t.eval(feed_dict=test_dict), axis=0)))

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

