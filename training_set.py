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

import tensorflow as tf

FLAGS = None

def load_dataset(dataset_path, n_parts = 1000):
	paths = []
	for p in dataset_path:
		paths.extend(glob.glob(p))

	if len(paths) == 0:
		raise Exception("No dataset provided")

	mut_features, region_counts = load_pickle(paths[0])
	colnames = mut_features.columns.values

	for path in paths[1:n_parts]:
		mutf, counts = load_pickle(path)
		mut_features = pd.concat([mut_features, mutf])
		region_counts = np.concatenate((region_counts, counts))

	mut_features = mut_features[colnames]
	return mut_features, region_counts

def filter_mutation(mut_features, region_counts, n_mut, true_mut_only = False):
	# take a subset of mutations, if needed
	if true_mut_only:
		mut_features = mut_features.iloc[region_counts == 1]

	if n_mut is not None:
		n_mut = int(n_mut)
		indices = np.random.randint(mut_features.shape[0], size=n_mut)
		mut_features = mut_features.iloc[indices]
		region_counts = region_counts[indices]
	else:
		n_mut = mut_features.shape[0]

	return mut_features, region_counts, n_mut

def make_training_set(mut_features, region_counts, trinuc, feature_path, region_size, dataset_with_annotation, max_tumours = None):
	mut_features = mut_features.reset_index(drop=True)

	mut_annotation = mut_features.iloc[:,:3]
	tumour_names = np.asarray(mut_annotation.Tumour).ravel()
	unique_tumours = np.unique(tumour_names)
	tumour_ids = np.squeeze(np.asarray([np.where(unique_tumours == name) for name in tumour_names]))[:,np.newaxis]
	unique_tumours_ind = np.unique(tumour_ids)
	
	print("Making dataset of {} mutations from {} tumours".format(mut_annotation.shape[0], len(unique_tumours)))
	trinuc_dict, trinuc_list = trinuc
	mut_types = mut_features.loc[:,trinuc_list]
	n_mut_types = mut_types.shape[1]
	mut_vaf = np.array(mut_features['VAF'])

	n_mut = 0
	if max_tumours is None:
		max_tumours = len(unique_tumours_ind)
	else:
		max_tumours = int(max_tumours)

	for tum in unique_tumours_ind[:max_tumours]:
		print("Reading dataset for tumour id " + str(tum) + " Tumour name: " + unique_tumours[tum])

		dataset_tum_name = dataset_with_annotation.format(id = unique_tumours[tum])
		feature_file_name = dataset_tum_name[:-len(".annotation.hdf5")] + ".feature_annotation.pickle"
		if os.path.exists(dataset_tum_name) and os.path.exists(feature_file_name):
			dictionary = load_from_HDF(dataset_tum_name)
			dataset = dictionary["training_set"]

			n_mut += dataset.shape[0]
		else:
			print("Dataset " + dataset_tum_name + " does not exist. Creating a dataset....")
			features_cur = mut_features.iloc[np.where(tumour_ids == tum)[0]]
			region_counts_cur = region_counts[np.where(tumour_ids == tum)[0]]
			mut_annotation_cur = mut_annotation.iloc[np.where(tumour_ids == tum)[0]]

			mut_types_cur = mut_types.iloc[np.where(tumour_ids == tum)[0]]
			mut_vaf_cur = mut_vaf[np.where(tumour_ids == tum)[0]]
			tumour_ids_cur = tumour_ids[np.where(tumour_ids == tum)[0]]

			region_features, rc = get_annotation_for_mutation_regions(features_cur, trinuc, feature_path, region_size)
			region_feature_names = region_features[0,0,:]
			region_features = region_features[:,1:]

			if not(features_cur.shape[0] == region_counts_cur.shape[0] and region_counts_cur.shape[0] == region_features.shape[0]):
				raise Exception("Dataset is incorrect: all parts should contain the same number of samples")

			n_samples = features_cur.shape[0]
			regionsize = region_features.shape[1]
			n_region_features = region_features.shape[2]

			region_features = region_features.reshape((n_samples, regionsize * n_region_features))
			region_feature_names = np.tile(region_feature_names, regionsize)

			region_counts_cur = region_counts_cur[:,np.newaxis]
			dataset = np.concatenate((mut_types_cur, region_features), axis=1)
			n_mut += dataset.shape[0]
			feature_names = np.concatenate((mut_types_cur.columns.values, region_feature_names))

			save_to_HDF(dataset_tum_name, 
				 	{"training_set": dataset, "labels": region_counts_cur, 
				 	"n_unique_tumours": np.array(1), "x_tumour_ids" : tumour_ids_cur, 
					"mut_vaf": mut_vaf_cur})

			pickle_name = dataset_tum_name[:-len(".annotation.hdf5")] + ".feature_annotation.pickle"
			dump_pickle([mut_annotation_cur, feature_names], pickle_name)

	num_features = dataset.shape[1]
	return n_mut, num_features, max_tumours
	
	#return dataset, region_counts, n_unique_tumours, tumour_ids, mut_vaf

def train_test_split(data_list, split_by, test_size=0.2, random_state=1991):
	split = int(len(split_by)*(1-test_size))
	split_vaf = sorted(split_by)[::-1][split]

	train_indices = np.where(split_by > split_vaf)[0]
	test_indices = np.where(split_by <= split_vaf)[0]

	np.random.shuffle(train_indices)
	np.random.shuffle(test_indices)

	splitted_data = []
	for data in data_list:
		splitted_data.append(data[train_indices])
		splitted_data.append(data[test_indices])
	return splitted_data

def read_tumour_data(files, tumour_batch_size = None, start = 0):
	if tumour_batch_size is not None:
		files = files[start*tumour_batch_size:(start+1)*tumour_batch_size] 

	if len(files) == 0:
		return None, None, None, None, None, None, None

	training_set = []
	labels = []
	x_tumour_ids = []
	mut_vaf = []
	n_unique_tumours = 0
	mut_annotation = feature_names = None


	for file in files:
		dictionary = load_from_HDF(file)
		training_set.append(dictionary["training_set"])
		labels.append(np.squeeze(dictionary["labels"]))
		n_unique_tumours += int(dictionary["n_unique_tumours"])
		x_tumour_ids.append(dictionary["x_tumour_ids"])
		mut_vaf.append(dictionary["mut_vaf"])

		pickle_annotation_file = file[:-len(".annotation.hdf5")] + ".feature_annotation.pickle"

		if os.path.exists(pickle_annotation_file):
			mut_annot_cur, feature_names = load_pickle(pickle_annotation_file)
			mut_annotation = pd.concat((mut_annotation, mut_annot_cur))
		else:
			print("Warning! "+ pickle_annotation_file + " does not exist")

	training_set = np.concatenate(training_set, axis = 0)
	labels = np.concatenate(labels)
	x_tumour_ids = np.concatenate(x_tumour_ids)
	mut_vaf = np.concatenate(mut_vaf)

	if not(mut_annotation.shape[0] == training_set.shape[0] and len(feature_names) == training_set.shape[1]):
		raise Exception("ERROR: Dataset is incorrect: all parts should contain the same number of samples")

	labels = labels[:,np.newaxis]

	print("Data has been read.")
	return training_set, labels, n_unique_tumours, x_tumour_ids, mut_vaf, mut_annotation, feature_names

def downsample_data(tumour_data, target_mut_features):
	training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names = tumour_data 
	target_annot = target_mut_features.iloc[:,:3]

	available_tumours = np.unique(mut_annotation.Tumour)
	target_tumours =  np.unique(target_annot.Tumour)

	if mut_annotation is None or len(available_tumours) != len(target_tumours):
		print("ERROR: Mutation annotation missing for tumours:")
		print(np.setdiff1d(target_tumours, available_tumours, assume_unique=True))
		return tumour_data

	pos_in_dataset = [tum + chr + pos for tum, chr, pos in zip(mut_annotation.Tumour, mut_annotation.Chr, mut_annotation.Pos)]
	target_pos = [tum + chr + pos for tum, chr, pos in zip(target_annot.Tumour, target_annot.Chr, target_annot.Pos)]

	indices = [i for i in range(len(pos_in_dataset)) if pos_in_dataset[i] in target_pos]
	training_set = training_set[indices]
	labels = labels[indices]
	tumour_ids = tumour_ids[indices]
	mut_vaf = mut_vaf[indices]
	mut_annotation = mut_annotation.iloc[indices]

	return training_set, labels, n_unique_tumours, tumour_ids, mut_vaf, mut_annotation, feature_names

def make_batches_over_time(dataset, region_counts, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, sequences_per_batch, n_timesteps):
	# Using only positive examples
	
	positive_examples = np.where(region_counts.ravel() > 0)[0]
	dataset = dataset[positive_examples]
	tumour_ids = tumour_ids[positive_examples]
	mut_vaf = mut_vaf[positive_examples]

	unique_tumours = np.unique(tumour_ids)
	np.random.shuffle(unique_tumours)
	
	all_tumours = []
	all_time_estimates = []
	tumour_name_batch = []
	
	n_tumour_batches = 1
	if len(unique_tumours) > sequences_per_batch:
		n_tumour_batches = len(unique_tumours) // sequences_per_batch

	for t_batch_ind in range(n_tumour_batches):
		tumour_batch = []
		tumour_batch_time_estimates = []
		tumour_names = []
		batch_size = min(sequences_per_batch, len(unique_tumours) - t_batch_ind * sequences_per_batch)
		print(batch_size)
		if batch_size == 0:
			next
		for k in range(batch_size):
			tum = unique_tumours[t_batch_ind * sequences_per_batch + k]

			data_cur = dataset[np.where(tumour_ids == tum)[0]]
			mut_vaf_cur = mut_vaf[np.where(tumour_ids == tum)[0]]
			tumour_ids_cur = tumour_ids[np.where(tumour_ids == tum)[0]]

			mut_vaf_cur = mut_vaf_cur.astype(float)
			sort_order = mut_vaf_cur.argsort()[::-1]
			data_cur = data_cur[sort_order]
			mut_vaf_cur = mut_vaf_cur[sort_order]
			n_mut = len(mut_vaf_cur)

			tumour_data = []
			time_estimates = []

			time_steps_in_tumour = n_mut // batch_size_per_tp

			for i in range(min(n_timesteps,time_steps_in_tumour)):
				time_estimates.append(np.mean(mut_vaf_cur[i * batch_size_per_tp : (i+1) * batch_size_per_tp]))
				tumour_data.append(data_cur[i * batch_size_per_tp : (i+1) * batch_size_per_tp])

			if n_timesteps > time_steps_in_tumour:
				shape = np.array(tumour_data).shape[1:]
				shape = (n_timesteps-time_steps_in_tumour, shape[0], shape[1])

				time_estimates.extend(np.zeros(n_timesteps-time_steps_in_tumour).tolist())
				tumour_data.extend(np.zeros(shape).tolist())

			tumour_batch.append(tumour_data)
			tumour_batch_time_estimates.append(np.array(time_estimates)[:,np.newaxis].tolist())
			tumour_names.append(tum)

		all_tumours.append(tumour_batch)
		all_time_estimates.append(tumour_batch_time_estimates)
		tumour_name_batch.append(tumour_names)

	# index of tumour batch is 0th dimension
	# over time is 1th dimension
	# batch of several tumours is 2nd dimension
	# number of mutations per time point is 3rd
	all_tumours = np.transpose(np.array(all_tumours), (0,2,1,3,4))
	all_time_estimates = np.transpose(np.array(all_time_estimates), (0,2,1,3))

	# !!!! how to avoid using the same length all the time?
	#truncate_to = min([x.shape[0] for x in all_time_estimates])
	# truncate_to = 3
	# all_time_estimates = [x[:truncate_to] for x in all_time_estimates]
	# all_tumours = [x[:truncate_to] for x in all_tumours]

	print("total")
	print(np.array(all_tumours).shape)
	print(np.array(all_time_estimates).shape)

	return np.array(all_tumours), tumour_name_batch, np.array(all_time_estimates)

def make_set_for_predicting_mut_rate(training_set, labels, mut_annotation, feature_names):
	if feature_names[0] != 'C_A_ACA' or feature_names[95] != 'T_G_TTT':
		print("Warning: mutation types are not the first 96 columns!")

	mut_types = training_set[:,:96]
	region_features = training_set[:,96:]
	training_set, labels = region_features, mut_types

	return training_set, labels


def make_batches_over_time_type_multinomials(dataset, region_counts, n_unique_tumours, tumour_ids, mut_vaf, batch_size_per_tp, sequences_per_batch, n_timesteps):
	# Using only positive examples
	positive_examples = np.where(region_counts.ravel() > 0)[0]
	dataset = dataset[positive_examples]
	tumour_ids = tumour_ids[positive_examples]
	mut_vaf = mut_vaf[positive_examples]

	unique_tumours = np.unique(tumour_ids)
	np.random.shuffle(unique_tumours)
	
	all_tumours = []
	all_time_estimates = []
	tumour_name_batch = []
	
	n_tumour_batches = 1
	if len(unique_tumours) > sequences_per_batch:
		n_tumour_batches = len(unique_tumours) // sequences_per_batch

	for t_batch_ind in range(n_tumour_batches):
		tumour_batch = []
		tumour_batch_time_estimates = []
		tumour_names = []
		batch_size = min(sequences_per_batch, len(unique_tumours) - t_batch_ind * sequences_per_batch)
		print(batch_size)
		if batch_size == 0:
			next
		for k in range(batch_size):
			tum = unique_tumours[t_batch_ind * sequences_per_batch + k]

			data_cur = dataset[np.where(tumour_ids == tum)[0]]
			mut_vaf_cur = mut_vaf[np.where(tumour_ids == tum)[0]]
			tumour_ids_cur = tumour_ids[np.where(tumour_ids == tum)[0]]

			mut_vaf_cur = mut_vaf_cur.astype(float)
			sort_order = mut_vaf_cur.argsort()[::-1]
			data_cur = data_cur[sort_order]
			mut_vaf_cur = mut_vaf_cur[sort_order]
			n_mut = len(mut_vaf_cur)

			tumour_data = []
			time_estimates = []

			time_steps_in_tumour = n_mut // batch_size_per_tp

			for i in range(min(n_timesteps,time_steps_in_tumour)):
				time_estimates.append(np.mean(mut_vaf_cur[i * batch_size_per_tp : (i+1) * batch_size_per_tp]))
				mut_types = data_cur[i * batch_size_per_tp : (i+1) * batch_size_per_tp,:96]
				mut_types = (np.sum(mut_types, axis =0)/sum(np.sum(mut_types, axis =0)))[np.newaxis,:]
				tumour_data.append(mut_types)

			if n_timesteps > time_steps_in_tumour:
				shape = np.array(tumour_data).shape[1:]
				shape = (n_timesteps-time_steps_in_tumour, shape[0], shape[1])

				time_estimates.extend(np.zeros(n_timesteps-time_steps_in_tumour).tolist())
				tumour_data.extend(np.zeros(shape).tolist())

			tumour_batch.append(tumour_data)
			tumour_batch_time_estimates.append(np.array(time_estimates)[:,np.newaxis].tolist())
			tumour_names.append(tum)

		all_tumours.append(tumour_batch)
		all_time_estimates.append(tumour_batch_time_estimates)
		tumour_name_batch.append(tumour_names)

	# index of tumour batch is 0th dimension
	# over time is 1th dimension
	# batch of several tumours is 2nd dimension
	# number of mutations per time point is 3rd
	all_tumours = np.transpose(np.array(all_tumours), (0,2,1,3,4))
	all_time_estimates = np.transpose(np.array(all_time_estimates), (0,2,1,3))

	# !!!! how to avoid using the same length all the time?
	#truncate_to = min([x.shape[0] for x in all_time_estimates])
	# truncate_to = 3
	# all_time_estimates = [x[:truncate_to] for x in all_time_estimates]
	# all_tumours = [x[:truncate_to] for x in all_tumours]

	print("total")
	print(np.array(all_tumours).shape)
	print(np.array(all_time_estimates).shape)

	return np.array(all_tumours), tumour_name_batch, np.array(all_time_estimates)


