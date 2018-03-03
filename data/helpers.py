# coding=utf-8
import pickle
import numpy as np
import pandas as pd
import h5py
import os
import csv
import tensorflow as tf
from shutil import copyfile

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def save_to_HDF(fname, data):
	"""Save data (a dictionary) to a HDF5 file."""
	with h5py.File(fname, 'w') as f:
		grp=f.create_group('alist')
		for key, item in data.items():
			grp.create_dataset(key,data=item.astype(float))

def load_from_HDF(fname):
	"""Load data from a HDF5 file to a dictionary."""
	data = dict()
	with h5py.File(fname, 'r') as f:
		for group in f:
			for key in f[group]:
				data[key] = np.asarray(f[group][key])
				#print(key + ":", f[key])
	return data

def read_csv(filename):
	res = []
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for i,row in enumerate(reader):
			res.append(row)
	return(res)

def write_output_to_file(filename, data):
	# write content in data by line to filename
	with open(filename, "w") as output:
		for item in data:
			output.write("%s\n" % item)

def convert_onehot(one_hot_matrix):
		"""
		convert one hot encoding matrix back to vector
		:param one_hot_matrix:
		:return:
		"""
		vector = []
		for i in one_hot_matrix:
				vector.append(int(np.where(i == 1)[0]))
		return vector

def combine_column(l):
		"""
		combine all the matrix in l
		Ex.
		> a = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
		> b = np.array([["a"],["a"],["a"],["a"],["a"]])
		> l = [a,b]
		> add_column(l)
		> [['1' '1' 'a']
		> ['2' '2' 'a']
		> ['3' '3' 'a']
		> ['4' '4' 'a']
		> ['5' '5' 'a']]
		"""
		out = l[0]
		l = l[1:]
		for i in l:
				if i is None:
					continue
				out = np.append(out, i, 1)
		return out

def write_output_to_file(filename, data):
		with open(filename, "w") as output:
				for item in data:
						output.write("%s\n" % item)

def get_col(colname, matrix):
		"""
		return columns with colname in matrix
		"""
		cols = matrix[:, np.where(matrix[0,] == colname)]
		return cols[1:,:].astype(float).squeeze()
				
def get_one_hot_encoding(matrix, n_classes = None):
	"""
	get an one hot encoding matrix with corresponding position = 1
	matrix is zero indexed
	"""
	if not n_classes:
		n_classes = max(matrix)+1

	output = []
	for i in matrix:
		cla = int(i)
		one_hot = [0] * n_classes
		one_hot[cla] = 1
		output.append(one_hot)
	output = np.asarray(output)
	return output

def add_colname(matrix, colname):
	"""
	add colname to matrix
	:param matrix:
	:param colname:
	:return:
	"""
	if isinstance(matrix, list):
		return add_colname(np.asarray(matrix)[:,np.newaxis], colname)

	if isinstance(colname, str):
		# if the column name is just a string, duplicate it for all the columns
		colnames = np.repeat(colname, matrix.shape[1])
		colnames = colnames.reshape(1, matrix.shape[1])
		matrix = np.concatenate((colnames, matrix))

	if isinstance(colname, list):
		# if the column name is a list, check that number of dimensions is correct
		if matrix.shape[1] != len(colname):
			raise Exception("Matrix dimensions are not compatible with colnames: " + str(matrix.shape) + " vs " + str(len(colname)))
		
		matrix = np.concatenate((np.expand_dims(np.asarray(colname), axis=0), matrix), axis=0)
	return matrix

def find_closest_region(interval, sorted_region_list):
	"""
	Find one of the regions from the sorted_region_list that overlaps the input inteval. 
	Return: index and the found interval
	This function does not return the interval with the maximum overlap. It return one of the regions that have a overlap
	"""
	start = 0
	end = len(sorted_region_list) - 1
	while end >= start:
		mid = (end + start) // 2
		if have_intersection(interval, sorted_region_list[mid]):
			return mid, sorted_region_list[mid]
		elif interval[0] > sorted_region_list[mid][1]:
			start = mid + 1
		elif interval[1] < sorted_region_list[mid][0]:
			end = mid - 1
	return -1, None


def have_intersection(interval1, interval2):
	"""
	Returns boolean if the intervals have some intersection or not
	"""
	two_inside_one = interval1[0] <= interval2[0] and interval1[1] >= interval2[1]
	one_inside_two = interval1[0] >= interval2[0] and interval1[1] <= interval2[1]
	two_overlapps_left = interval1[0] <= interval2[1] and interval1[0] >= interval2[0]
	two_overlapps_right = interval1[1] <= interval2[1] and interval1[1] >= interval2[0]

	return two_inside_one or one_inside_two or two_overlapps_left or two_overlapps_right


def get_col_indices_trinucleotides(variant_features, trinuc):
	trinuc_dict, trinuc_list = trinuc

	if not isinstance(variant_features, pd.DataFrame) and not isinstance(variant_features, pd.Series):
		raise("ERROR: variant_features is not a pandas dataframe")
		
	colnames = list(variant_features.columns.values)
	return([colnames.index(tr) for tr in trinuc_list])


def get_counts_per_bin(variant_features, interval, trinuc):
	chrom, start, end = interval

	if not isinstance(variant_features, pd.DataFrame) and not isinstance(variant_features, pd.Series):
		raise("ERROR: variant_features is not a pandas dataframe")

	trinuc_indices = get_col_indices_trinucleotides(variant_features, trinuc)

	chr_column = int(np.where(variant_features.columns.values ==  'Chr')[0])
	pos_column = int(np.where(variant_features.columns.values ==  'Pos')[0])

	variants_chr = variant_features[variant_features.Chr == chrom]
	pos = int(variants_chr.Pos.values[0]) # check how this works with multiple regions !!!!

	counts_bin = [0] * len(trinuc_indices)
	pos_bin = np.where((pos >= start) & (pos <= end))[0]

	if len(pos_bin) != 0:
		variants_bin =  np.array(variants_chr)[pos_bin,:]
		counts_bin = np.sum(variants_bin[:,trinuc_indices].astype(int), axis = 0)

	return(counts_bin)


def filter_vcfs(vcf_list, vcf_filter):
	# filter tumors by the list provided in vcf_filter file
	with open(vcf_filter) as file:
		filter = [x.strip("\"\"\n") for x in file.read().splitlines()]
	vcf_list = [x for x in vcf_list if x.split(".")[0] in filter]
	return vcf_list


def filter_tumours(training_set, labels, unique_tumours, x_tumour_ids, tumor_ids):
	# take a subset of tumours, if needed
	unique_tumours = unique_tumours[tumor_ids]
	samples_to_use = [x in tumor_ids for x in x_tumour_ids]

	training_set = training_set[samples_to_use]
	labels = labels[samples_to_use]
	x_tumour_ids = x_tumour_ids[samples_to_use]

	return training_set, labels, unique_tumours, x_tumour_ids


def load_annotation(feature_path):
	try:
		mRNA = sort_dict(load_pickle(os.path.join(feature_path, "mRNA.pickle")))
		trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))
		#alex_signature_file = load_npy(os.path.join(feature_path,"signature.npy"))
		hg19 = load_pickle(os.path.join(feature_path,"hg.pickle"))
		chromatin = sort_dict(load_pickle(os.path.join(feature_path, "chromatin.pickle")))

		other_features = {}
		for file in os.listdir(feature_path):
			if file in ["mRNA.pickle", "trinucleotide.pickle", "hg.pickle", "chromatin.pickle", "signature.npy"]:
				continue
			if not file.endswith(".pickle"):
				continue

			new_feature = sort_dict(load_pickle(os.path.join(feature_path, file)))
			other_features[file[:-len(".pickle")]] = new_feature
	except Exception as error:
		raise Exception("Please provide valid compiled feature data files.")

	features_chromatin_mRNA= {'chromatin':chromatin, 'mRNA': mRNA}

	return features_chromatin_mRNA, other_features, hg19, trinuc

def sort_dict(dictionary):
	for key in dictionary.keys():
		dictionary[key] = sorted(dictionary[key])
	return dictionary

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - (np.max(x, axis=axis)[:,:,np.newaxis]))
    return e_x / e_x.sum(axis=axis)[:,:,np.newaxis]

def prepare_model_dir(call_args, model_dir, file_name, args):
	model_dir = model_dir.format(*args)
	model_save_path = model_dir + "model.ckpt"
	os.makedirs(model_dir, exist_ok=True)
	# copy current script to folder
	copyfile(file_name, model_dir + os.path.basename(file_name))
	# write the call to the folder
	with open(model_dir + "call.txt", "w") as text_file:
		text_file.write(" ".join(call_args))
	return model_dir, model_save_path
