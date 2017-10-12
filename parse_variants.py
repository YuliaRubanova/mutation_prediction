#!/usr/bin/python
# coding=utf-8
#v2.1
from __future__ import division
from sklearn.model_selection import KFold
import vcf
import numpy as np
from helpers import *

np.random.seed(1)
max_variants = 10**5 # mutations in the tumour are subsampled to this number

class VariantsFileParser(object):

	def __init__(self, filename, hg, trinuc, time_points = None, chromatin_dict = None, mrna_dict = None, alex_sig = None, signatures = None):
		self._filename = filename
		self._chromatin_dict = chromatin_dict
		self._mrna_dict = mrna_dict
		self._hg = hg
		self._trinuc_dict, self._trinuc_list = trinuc
		self._time_points = time_points # mixture file
		self._alex_sig = alex_sig
		self._signatures = signatures

	def _get_variants(self):
		"""
		create a list of variants. Variants are represented as vcf.Record class.
		:return:
		"""
		matrix = []
		low_sup_matrix = []

		vcf_reader = vcf.Reader(open(self._filename))

		vcf_reader = list(vcf_reader)
		count = len(vcf_reader)

		if count > max_variants:
			ind = np.random.choice(count, size=max_variants, replace=False).astype(int)

		for i, record in enumerate(vcf_reader):
			if count > max_variants and i not in ind:
				continue

			if "VAF" in record.INFO.keys():
				record.INFO["VAF"] = float(record.INFO["VAF"]) * 2
			else:
				continue

			if record.CHROM == "chrY":
				continue

			if record.FILTER == ["LOWSUPPORT"]:
				low_sup_matrix.append(record)
			else:
				matrix.append(record)
		matrix = np.asarray(matrix)
		low_sup_matrix = np.asarray(low_sup_matrix)

		return matrix, low_sup_matrix

	def _read_from_validated_data(self, validated_matrix):
		"""
		separate data based on validated_matrix
		if filter == PASS: train/test
		else: negative
		:param validated_matrix: a matrix containing chr|pos|filter read from validated vcfs
		:return: matrix, negative
		"""
		#self._validated_matrix = validated_matrix
		matrix = []
		negative_matrix = []
		vcf_reader = vcf.Reader(open(self._filename))
		for record in vcf_reader:
			chro = "chr"+record.CHROM
			#print chro
			pos = str(record.POS-1)
			#print pos
			#print validated_matrix.shape
			entry = validated_matrix[np.where((validated_matrix[:,0]==chro)*(validated_matrix[:,1]==pos))]
			#print entry
			if len(entry) == 0:
			#        print "cannot find pos"
				continue
			else:
								#print entry[0][-1]
				filter = entry[0][-1]
			if "VAF" in record.INFO.keys():
				record.INFO["VAF"][0] = float(record.INFO["VAF"][0]) * 2
			else:
				continue
			if filter == "PASS":
				matrix.append(record)
			else:
				negative_matrix.append(record)
		return np.asarray(matrix), np.asarray(negative_matrix)

	def _nfold_shuffle(self, matrix, n):
		"""
		separate data based on n
		"""
		train =[]
		test = []
		kf = KFold(n_splits=n, shuffle=True, random_state=10)
		for train_index, test_index in kf.split(matrix):
			train.append(matrix[train_index])
			test.append(matrix[test_index])
		return train, test

	def _get_input_data(self, n, validate=[]):
		"""
		read variants file and separate train, test, lowsupport data
		:param n: n-fold cross validation
		:return: train, test, lowsupport
				"""
		if len(validate)!=0:
			data, low_support = self._read_from_validated_data(validate)
		else:
			data, low_support = self._save_as_matrix()
			print(data.shape)

			print(low_support.shape)
		train, test = self._nfold_shuffle(data, n)

		return train, test, low_support

	def _get_features(self, input_data):
		"""
		Get all the features for variants.
		:param input_data: a list of variants obtained _get_variants function
		:return: matrix with features for each variant
		"""
		chromosomes = [None] * len(input_data)
		positions = [None] * len(input_data)
		chromatin = [None] * len(input_data)
		trans_region = [None] * len(input_data)
		sense = [None] * len(input_data)
		mut_type = [None] * len(input_data)
		VAF_list = [None] * len(input_data)
		se = []
		p_ce = []

		compute_signatures = self._time_points is not None and  self._alex_sig is not None and self._signatures is not None

		for i, record in enumerate(input_data):
			#print("features: "+str(i))
			variant_parser = Variant(record)
			chromosome = "chr" + str(record.CHROM)
			position = record.POS 

			chromosomes[i] = chromosome
			positions[i] = position

			if self._chromatin_dict:
				find_region = variant_parser._find_variant_in_region(self._chromatin_dict[chromosome])
				chromatin[i] = int(find_region)
			else:
				chromatin[i] = -1

			if self._mrna_dict:
				trans_interval = variant_parser._find_variant_in_region(self._mrna_dict[chromosome])
				# trans.append(variant_parser._find_variant_in_region(mrna_dict[chromosome]))
				if trans_interval:
					trans_region[i] = 1 
					if trans_interval[2] == "+":
						if record.REF == "G" or record.REF == "A":  # the mutation is on antisense strand
							sense[i] == 1
						else:  # the mutation is on sense strand
							sense[i] == 0
					else:
						if record.REF == "T" or record.REF == "C":
							sense[i] == 1
						else:
							sense[i] == 0
				else:
					trans_region[i] = 0
					sense[i] == -1
			else:
					trans_region[i] = 0
					sense[i] == -1

			trinucleotide_class = variant_parser._get_trinucleotide(self._hg)
			int_type = self._trinuc_dict[trinucleotide_class]

			mut_type[i]= int_type

			VAF = float(record.INFO["VAF"])
			VAF_list[i] = VAF


			## get signature vector from time point file
			# vaf_values = self._time_points[0, :]
			# idx = (np.abs(vaf_values - VAF)).argmin()
			# print idx
			# signature_vector = self._time_points[:, idx][1:]
			# print signature_vector

			## get signature from overall
			se = p_ce = None
			if (compute_signatures):
				signature_vector = self._time_points
				se.append(signature_vector)

				pce = variant_parser._calculate_pce(signature_vector, self._alex_sig, self._signatures, int_type)
				p_ce.append(pce)

		chromosomes = add_colname(np.asarray(chromosomes)[:,np.newaxis], "Chr")
		positions = add_colname(np.asarray(positions)[:,np.newaxis], "Pos")
		feature_matrix = np.concatenate((chromosomes, positions), axis=1)

		VAF_list = add_colname(np.asarray(VAF_list)[:,np.newaxis], "VAF")
		feature_matrix = combine_column([feature_matrix, VAF_list])

		mut_type_one_hot = get_one_hot_encoding(mut_type, n_classes = len(self._trinuc_list))
		mut_type = add_colname(np.asarray(mut_type_one_hot), self._trinuc_list)

		feature_matrix = combine_column([feature_matrix, mut_type])

		if self._chromatin_dict:
			chromatin = add_colname(np.asarray(chromatin), "Chromatin")
			feature_matrix = combine_column([feature_matrix, chromatin])
		
		if self._mrna_dict:
			trans_region = add_colname(np.asarray(trans_region), "Transcribed")
			sense = add_colname(np.asarray(sense), "Strand")
			feature_matrix = combine_column([feature_matrix, trans_region, sense])

		if (compute_signatures):
			se = add_colname(np.asarray(se), "Exposure")
			p_ce = np.asarray(p_ce).reshape(np.asarray(p_ce).shape[0],1)
			p_ce = add_colname(np.asarray(p_ce), "p_ce")
			feature_matrix = combine_column([feature_matrix, se, p_ce])

		return feature_matrix

class Variant(object):

	def __init__(self, variant=None):
		self._variant = variant

	def _get_trinucleotide(self, hg19):
		"""
		get trinucleotide content from hg19 file
		:param hg19: load from hg19.pickle
		:return:
		"""
		reference = str(self._variant.REF)
		alt = str(self._variant.ALT[0])
		position = self._variant.POS - 1
		mut_pair = {"G": "C", "T": "A", "A": "T", "C": "G"}
		sequence = hg19[self._variant.CHROM] # sequence of corresponding chromosome
		if reference == "G" or reference == "A":
			reference = mut_pair[reference]
			alt = mut_pair[alt]
			tri = mut_pair[sequence[position - 1].upper()] + \
				  mut_pair[sequence[position].upper()] + mut_pair[sequence[position + 1].upper()]
		else:
			tri = sequence[position - 1].upper() + sequence[position].upper() \
				  + sequence[position + 1].upper()

		return (reference, alt, tri)

	def _find_variant_in_region(self, input_list):
		"""
		Use binary serach to find the range
		:param input_list: list of tuples : (start, end)
		:return:
		"""
		position = self._variant.POS - 1
		input_list.sort()
		start = 0
		end = len(input_list) - 1
		while end >= start:
			mid = (end + start) // 2
			if position in range(input_list[mid][0], input_list[mid][1] + 1):
				return input_list[mid]
			elif position > input_list[mid][1]:
				start = mid + 1
			elif position < input_list[mid][0]:
				end = mid - 1
		return None

	# def _get_overlap(self, phi_values_matrix):
	# 	"""
	# 	find out if the variant has VAF value
	# 	:param phi_values_matrix: matrix contains chr_pos = phi/vaf value
	# 	:return:
	# 	"""
	# 	# not used
	# 	# todo change here
	# 	combine_pos = self._variant.CHROM + "_" + str(self._variant.POS)
	# 	find = phi_values_matrix[phi_values_matrix[:, 0] == combine_pos]
	# 	# bool: (train, test, low_sup)
	# 	if len(find) != 0:
	# 		return (True, False, False)
	# 	if self._variant.FILTER == ["LOWSUPPORT"]:
	# 		return (False, False, True)
	# 	else:
	# 		return (False, True, False)

	def _calculate_pce(self, exposure_vector, alex_signature, sigs, mut_type):
		"""
		calculate p_ce for a given mutation

		:param exposure_vector: exposure at each time point
		:param alex_signature: alex signature file, rows corresponding to mut_type and cols=signatures
		:param sigs: active signature in a given tumor
		:param mut_type: 96 types of mutations (index: 0-95)
		:return:
		"""

		idx = np.where(np.in1d(alex_signature, sigs))
		select_signature_cols = alex_signature[:,2:][mut_type+1,idx].squeeze()
		sum = np.dot(np.asarray(select_signature_cols,dtype=np.float), exposure_vector)

		return sum

