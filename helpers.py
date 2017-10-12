#!/usr/bin/python
# coding=utf-8

import pickle
import numpy as np

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

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


