#!/usr/bin/python
# coding=utf-8
#v2.1
import time
import argparse
import os
from read_files import *
from parse_variants import *
from helpers import *
import numpy as np

##################################################################
# Load region features for the mutations in the training set
##################################################################

def get_counts_by_type(region_counts, mut_features, trinuc):
	trinuc_dict, trinuc_list = trinuc

	feature_data = mut_features.loc[:,trinuc_list].astype(int)
	region_counts = np.matrix(region_counts)

	return np.sum(np.multiply(region_counts, feature_data), axis=1)

def get_random_regions(n, region_size):
	regions = []
	for i in range(n):
		np.random.seed(n)
		chrom = 'chr' + str(np.random.choice(len(chromosome_lengths)) + 1)
		start = int(np.random.uniform(0, chromosome_lengths[chrom] - region_size))
		regions.append((chrom, start, start+region_size))

	return regions


def load_annotation(feature_path):
	try:
		mRNA = load_pickle(os.path.join(feature_path, "mRNA.pickle"))
		trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))
		#alex_signature_file = load_npy(os.path.join(feature_path,"signature.npy"))
		hg19 = load_pickle(os.path.join(feature_path,"hg.pickle"))
		chromatin = load_pickle(os.path.join(feature_path, "chromatin.pickle"))
		
		other_features = {}
		for file in os.listdir(feature_path):
			if file in ["mRNA.pickle", "trinucleotide.pickle", "hg.pickle", "chromatin.pickle", "signature.npy"]:
				continue
			other_features[file[:-len(".pickle")]] = load_pickle(os.path.join(feature_path, file))
	except Exception as error:
		raise Exception("Please provide valid compiled feature data files.")

	features_chromatin_mRNA= {'chromatin':chromatin, 'mRNA': mRNA}

	return features_chromatin_mRNA, other_features, hg19

def get_annotation_for_mutation_regions(mut_features, trinuc, feature_path, region_size):
	n_samples = mut_features.shape[0]
	region_features_all = [None] * mut_features.shape[0]
	region_counts_all = [None] * mut_features.shape[0]

	features_chromatin_mRNA, other_features, hg19 = load_annotation(feature_path)

	print("Loading annotations for regions for " + str(n_samples) + " mutations...")
	t = time.time()

	for i, m in mut_features.iterrows():
		mut = pd.DataFrame(m.to_frame().transpose(), columns = mut_features.columns.values)
		counts_features = get_annotation_for_mutation_regions_one(mut, trinuc, features_chromatin_mRNA, other_features, hg19, region_size)
		region_counts_all[i], region_features_all[i] = counts_features

	print("Annotation loaded. Time elapsed: " + str(time.time()-t) + ". Per mutation: " + str((time.time()-t) / float(n_samples)))

	region_counts_all = np.squeeze(np.array(region_counts_all))
	region_features_all = np.squeeze(np.array(region_features_all), axis=1)

	return region_features_all, region_counts_all

def get_annotation_for_mutation_regions_one(mut, trinuc, features_chromatin_mRNA, other_features, hg19, region_size):
	tumour_name = mut['Tumour']

	variants_parser = VariantParser(tumour_name, hg19, trinuc,
		chromatin_dict = features_chromatin_mRNA['chromatin'], mrna_dict = features_chromatin_mRNA['mRNA'],
		other_features = other_features)

	mut_regions = variants_parser.get_region_around_mutation_from_features([mut], region_size)
	
	# region_counts -- counts of mutations there are in the regions that we selected
	# region_features -- features in the region, such as chromatin
	region_counts, region_features = variants_parser.get_region_features(mut_regions, mut)

	# take only region counts that correspond to the asked mutation type
	region_counts = get_counts_by_type(region_counts, mut, trinuc)

	return region_counts, region_features