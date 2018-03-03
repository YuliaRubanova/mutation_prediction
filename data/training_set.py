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

region_size = 100 #!!!!!

#dataset_path = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.regionsize1000.small.pickle"
mut_dataset_path = ["/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part1.pickle",
					"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part2.pickle",
					"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part3.pickle",
					"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part4.pickle",
					"/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour10000.mutations_only.part5.pickle"]
feature_path = "/Users/yulia/Documents/mutational_signatures/dna_features_ryoga/"
model_save_path = "trained_models/model.region_dataset.model{}.tumours{}.mut{}/model.ckpt"
dataset_with_annotation = "/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/region_dataset.mutTumour1000.region_size" + str(region_size) + ".annotation.hdf5"

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

def filter_mutation(mut_features, region_counts, n_mut):
	# take a subset of mutations, if needed
	if n_mut is not None:
		n_mut = int(n_mut)
		indices = np.random.randint(mut_features.shape[0], size=n_mut)
		mut_features = mut_features.iloc[indices]
		region_counts = region_counts[indices]
	else:
		n_mut = mut_features.shape[0]

	return mut_features, region_counts, n_mut

def make_training_set(mut_features, region_counts, trinuc, feature_path, region_size):
	mut_features = mut_features.reset_index(drop=True)

	mut_annotation = mut_features.iloc[:,:3]
	tumour_names = np.asarray(mut_annotation.Tumour).ravel()
	unique_tumours = np.unique(tumour_names)
	tumour_ids = np.squeeze(np.asarray([np.where(unique_tumours == name) for name in tumour_names]))[:,np.newaxis]
	
	trinuc_dict, trinuc_list = trinuc
	mut_types = mut_features.loc[:,trinuc_list]

	mut_vaf = np.array(mut_features['VAF'])

	# !!!! new region_counts
	region_features, rc = get_annotation_for_mutation_regions(mut_features, trinuc, feature_path, region_size)

	region_features = region_features[:,1:]

	if not(mut_features.shape[0] == region_counts.shape[0] and region_counts.shape[0] == region_features.shape[0]):
		raise Exception("Dataset is incorrect: all parts should contain the same number of samples")

	n_samples = mut_features.shape[0]
	regionsize = region_features.shape[1]
	n_region_features = region_features.shape[2]
	n_mut_types = mut_types.shape[1]
	n_unique_tumours = len(unique_tumours)

	region_features = region_features.reshape((n_samples, regionsize*n_region_features))
	region_counts = region_counts[:,np.newaxis]

	dataset = np.concatenate((mut_types, region_features), axis=1)

	return dataset, region_counts, n_unique_tumours, tumour_ids, mut_vaf

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

def make_batches_over_time(dataset, region_counts, n_unique_tumours, tumour_ids, mut_vaf, batch_size):
	# Using only positive examples
	positive_examples = np.where(region_counts.ravel() > 0)[0]
	dataset = dataset[positive_examples]
	tumour_ids = tumour_ids[positive_examples]
	mut_vaf = mut_vaf[positive_examples]

	unique_tumours = np.unique(tumour_ids)
	
	all_tumours = []
	all_time_estimates = []
	for tum in unique_tumours:
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
		for i in range(n_mut // batch_size):
			time_estimates.append(np.mean(mut_vaf_cur[i * batch_size : (i+1) * batch_size]))
			tumour_data.append(data_cur[i * batch_size : (i+1) * batch_size])

		all_tumours.append(tumour_data)
		all_time_estimates.append(np.array(time_estimates)[:,np.newaxis])

	# !!!! how to avoid using the same length all the time?
	#truncate_to = min([x.shape[0] for x in all_time_estimates])
	truncate_to = 2 # !!!!!!
	all_time_estimates = [x[:truncate_to] for x in all_time_estimates]
	all_tumours = [x[:truncate_to] for x in all_tumours]

	return np.array(all_tumours), unique_tumours, np.array(all_time_estimates)
