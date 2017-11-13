#!/usr/bin/python
# coding=utf-8
#v2.1
from __future__ import division
from sklearn.model_selection import KFold
import vcf # pip3 install pyvcf
import numpy as np
from helpers import *

np.random.seed(1)
max_variants = 10**5 # mutations in the tumour are subsampled to this number

chromosome_lengths = {
'chr1':   249250621,  
'chr2':   243199373,  
'chr3':   198022430,  
'chr4':   191154276,   
'chr5':   180915260,  
'chr6':   171115067,  
'chr7':   159138663,  
'chr8':   146364022,  
'chr9':   141213431,  
'chr10':   135534747, 
'chr11':   135006516,    
'chr12':   133851895,  
'chr13':   115169878,  
'chr14':   107349540,  
'chr15':   102531392,  
'chr16':   90354753,  
'chr17':   81195210,   
'chr18':   78077248,  
'chr19':   59128983,  
'chr20':   63025520,  
'chr21':   48129895,   
'chr22':   51304566
}

class VariantParser(object):

	def __init__(self, filename, hg, trinuc, time_points = None, chromatin_dict = None, mrna_dict = None, alex_sig = None, signatures = None):
		self.filename = filename
		self.chromatin_dict = chromatin_dict
		self.mrna_dict = mrna_dict
		self.hg = hg
		self.trinuc = trinuc
		self.time_points = time_points # mixture file
		self.alex_sig = alex_sig
		self.signatures = signatures

	def get_variants(self):
		"""
		create a list of variants. Variants are represented as vcf.Record class.
		:return:
		"""
		matrix = []
		low_sup_matrix = []

		vcf_reader = vcf.Reader(open(self.filename))

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

			if record.CHROM == "Y" or record.CHROM == "X":
				continue

			if record.FILTER == ["LOWSUPPORT"]:
				low_sup_matrix.append(record)
			else:
				matrix.append(record)
		matrix = np.asarray(matrix)
		low_sup_matrix = np.asarray(low_sup_matrix)

		return matrix, low_sup_matrix

	# def nfold_shuffle(self, matrix, n):
	# 	"""
	# 	separate data based on n
	# 	"""
	# 	train =[]
	# 	test = []
	# 	kf = KFold(n_splits=n, shuffle=True, random_state=10)
	# 	for train_index, test_index in kf.split(matrix):
	# 		train.append(matrix[train_index])
	# 		test.append(matrix[test_index])
	# 	return train, test

	# def get_mut_list(self, n, validate=[]):
	# 	"""
	# 	read variants file and separate train, test, lowsupport data
	# 	:param n: n-fold cross validation
	# 	:return: train, test, lowsupport
	# 			"""
	# 	if len(validate)!=0:
	# 		data, low_support = self.read_from_validated_data(validate)
	# 	else:
	# 		data, low_support = self.save_as_matrix()
	# 		print(data.shape)

	# 		print(low_support.shape)
	# 	train, test = self.nfold_shuffle(data, n)

	# 	return train, test, low_support

	def get_features(self, mut_list, tumour_name = None):
		"""
		Get all the features for variants.
		:param mut_list: a list of variants obtained _get_variants function
		:return: matrix with features for each variant
		"""
		chromosomes = [None] * len(mut_list)
		positions = [None] * len(mut_list)
		chromatin = [None] * len(mut_list)
		trans_region = [None] * len(mut_list)
		sense = [None] * len(mut_list)
		mut_type = [None] * len(mut_list)
		VAF_list = [None] * len(mut_list)
		se = []
		p_ce = []

		compute_signatures = self.time_points is not None and  self.alex_sig is not None and self.signatures is not None

		for i, record in enumerate(mut_list):
			#print("features: "+str(i))
			chromosome = "chr" + str(record.CHROM)
			position = record.POS 

			chromosomes[i] = chromosome
			positions[i] = position

			if self.chromatin_dict:
				found_region = Variant.find_variant_in_region(record, self.chromatin_dict[chromosome])
				chromatin[i] = 1 if found_region else 0
			else:
				chromatin[i] = -1

			if self.mrna_dict:
				trans_interval = Variant.find_variant_in_region(record, self.mrna_dict[chromosome])
				# trans.append(variant.find_variant_in_region(mrna_dict[chromosome]))
				if trans_interval:
					trans_region[i] = 1 
					if trans_interval[2] == "+":
						if record.REF == "G" or record.REF == "A":  # the mutation is on antisense strand
							sense[i] = 1
						else:  # the mutation is on sense strand
							sense[i] = 0
					else:
						if record.REF == "T" or record.REF == "C":
							sense[i] = 1
						else:
							sense[i] = 0
				else:
					trans_region[i] = 0
					sense[i] = -1
			else:
					trans_region[i] = 0
					sense[i] = -1

			mut_type[i] = Variant.get_trinucleotide_one_hot(record, self.hg, self.trinuc)

			VAF = float(record.INFO["VAF"])
			VAF_list[i] = VAF


			## get signature vector from time point file
			# vaf_values = self.time_points[0, :]
			# idx = (np.abs(vaf_values - VAF)).argmin()
			# print idx
			# signature_vector = self.time_points[:, idx][1:]
			# print signature_vector

			## get signature from overall
			se = p_ce = None
			if (compute_signatures):
				signature_vector = self.time_points
				se.append(signature_vector)

				pce = Variant.calculate_pce(signature_vector, self.alex_sig, self.signatures, int_type)
				p_ce.append(pce)

		chromosomes = add_colname(chromosomes, "Chr")
		positions = add_colname(positions, "Pos")
		feature_matrix = np.concatenate((chromosomes, positions), axis=1)

		VAF_list = add_colname(VAF_list, "VAF")
		feature_matrix = combine_column([feature_matrix, VAF_list])

		mut_type = add_colname(np.asarray(mut_type), self.trinuc[1])

		feature_matrix = combine_column([feature_matrix, mut_type])

		if self.chromatin_dict:
			chromatin = add_colname(chromatin, "Chromatin")
			feature_matrix = combine_column([feature_matrix, chromatin])
		
		if self.mrna_dict:
			trans_region = add_colname(trans_region, "Transcribed")
			sense = add_colname(sense, "Strand")
			feature_matrix = combine_column([feature_matrix, trans_region, sense])

		if (compute_signatures):
			se = add_colname(se, "Exposure")
			p_ce = np.asarray(p_ce).reshape(np.asarray(p_ce).shape[0],1)
			p_ce = add_colname(p_ce, "p_ce")
			feature_matrix = combine_column([feature_matrix, se, p_ce])

		if tumour_name:
			feature_matrix = combine_column([ add_colname([tumour_name] * len(mut_list), "Tumour"), feature_matrix])
		
		feature_matrix = np.array(feature_matrix)
		feature_matrix = pd.DataFrame(feature_matrix[1:], columns = feature_matrix[0])

		return feature_matrix

	def get_region_around_mutation(self, mut, region_size):
		regions = []
		for i, record in enumerate(mut):
			chrom = "chr" + str(record.CHROM)
			position = record.POS 
			start, end = max(0,position - region_size//2), min(position + region_size//2, chromosome_lengths[chrom])
			regions.append((chrom, start, end))
		return regions

	def get_region_around_mutation_from_features(self, mut_feature_array, region_size):
		regions = []

		for i, record in enumerate(mut_feature_array):
			chrom = record['Chr'].values[0]
			position = int(record['Pos'])

			start, end = max(0,position - region_size//2), min(position + region_size//2, chromosome_lengths[chrom])
			regions.append((chrom, start, end))
		return regions

	def get_region_features(self, region_list, mut_features):
		counts_all = []
		features_all = []

		for region in region_list:
			chrom,start,end = region

			chromatin_chromosome = sorted(self.chromatin_dict[chrom])
			mrna_chromosome = sorted(self.mrna_dict[chrom])

			chromatin = add_colname(fill_data_for_region([start, end], chromatin_chromosome), "Chromatin")
			mrna = add_colname(np.matrix(fill_data_for_region([start, end], mrna_chromosome, fill_mRNA)), ["Transcribed", "Strand"])

			# find other mutations that might fall into this region
			counts = get_counts_per_bin(mut_features, (chrom, start, end), self.trinuc)

			counts_all.append(counts)
			features_all.append(combine_column([chromatin, mrna]))

		return counts_all, features_all

class Variant(object):

	@staticmethod
	def get_trinucleotide(variant, hg19):
		"""
		get trinucleotide content from hg19 file
		:param hg19: load from hg19.pickle
		:return:
		"""
		reference = str(variant.REF)
		alt = str(variant.ALT[0])
		position = variant.POS - 1
		mut_pair = {"G": "C", "T": "A", "A": "T", "C": "G"}
		sequence = hg19[variant.CHROM] # sequence of corresponding chromosome
		if reference == "G" or reference == "A":
			reference = mut_pair[reference]
			alt = mut_pair[alt]
			tri = mut_pair[sequence[position - 1].upper()] + \
				  mut_pair[sequence[position].upper()] + mut_pair[sequence[position + 1].upper()]
		else:
			tri = sequence[position - 1].upper() + sequence[position].upper() \
				  + sequence[position + 1].upper()

		return (reference, alt, tri)

	@staticmethod
	def find_variant_in_region(variant, region_list):
		"""
		Use binary serach to find the range
		:param region_list: list of tuples : (start, end)
		:return:
		"""
		position = variant.POS - 1
		region_list.sort()
		start = 0
		end = len(region_list) - 1
		while end >= start:
			mid = (end + start) // 2
			if position in range(region_list[mid][0], region_list[mid][1] + 1):
				return region_list[mid]
			elif position > region_list[mid][1]:
				start = mid + 1
			elif position < region_list[mid][0]:
				end = mid - 1
		return None

	@staticmethod
	def get_trinucleotide_one_hot(variant, hg19, trinuc):
		trinuc_dict, trinuc_list = trinuc
		mut_type = trinuc_dict[Variant.get_trinucleotide(variant, hg19)]
		mut_type_one_hot = get_one_hot_encoding([mut_type], n_classes = len(trinuc_list))[0]
		return mut_type_one_hot


	# def _get_overlap(self, phi_values_matrix):
	# 	"""
	# 	find out if the variant has VAF value
	# 	:param phi_values_matrix: matrix contains chr_pos = phi/vaf value
	# 	:return:
	# 	"""
	# 	# not used
	# 	# todo change here
	# 	combine_pos = self.variant.CHROM + "_" + str(self.variant.POS)
	# 	find = phi_values_matrix[phi_values_matrix[:, 0] == combine_pos]
	# 	# bool: (train, test, low_sup)
	# 	if len(find) != 0:
	# 		return (True, False, False)
	# 	if self.variant.FILTER == ["LOWSUPPORT"]:
	# 		return (False, False, True)
	# 	else:
	# 		return (False, True, False)

	@staticmethod
	def calculate_pce(exposure_vector, alex_signature, sigs, mut_type):
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

