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
# Generate training set for deep learning model from Ryoga's dataset from BRCA samples
# Each training example corresponds to one region. An example contains binary representation of 
# mutations at each position in the region and genomic annotation of chromatin, etc. in this region.
##################################################################

region_size = 10**3
maxsize_pickle = 20
max_mut_per_tumour = 10000

DEF_FEATURE_PATH = "/home/yulia/mnt/dna_features_ryoga/"
DEF_OUTPUT = "lsregion_dataset.regionsize" + str(region_size) + ".pickle"
DEF_VCF_DIR = "/home/ryogali/data/variants/"
DEF_VCF_FILTER = "/home/ryogali/dev/prob_model/BRCA_files.txt"

def get_random_regions(n, region_size):
	regions = []
	for i in range(n):
		np.random.seed(n)
		chrom = 'chr' + str(np.random.choice(len(chromosome_lengths)) + 1)
		start = int(np.random.uniform(0, chromosome_lengths[chrom] - region_size))
		regions.append((chrom, start, start+region_size))

	return regions

def generate_random_mutations(n_random, tumour_name, n_mut_types):
	tumour_names = [[tumour_name]] * n_random
	chrom = [["rand"]] * n_random
	pos = [[i] for i in range(n_random)]
	vaf = [[np.random.uniform()] for i in range(n_random)]

	mut_types = [np.random.multinomial(1, [1/float(n_mut_types)]*n_mut_types) for i in range(n_random)]

	chromatin = [[np.random.choice([0,1])] for i in range(n_random)]
	transcribed = [[np.random.choice([0,1])] for i in range(n_random)]
	strand = [[np.random.choice([0,1])] for i in range(n_random)]

	strand[transcribed == 0] = [-1]

	return combine_column([tumour_names, chrom, pos, vaf, mut_types, chromatin, transcribed, strand])

def get_counts_by_type(region_counts, mut_features, feature_header, trinuc):
	feature_data = mut_features[:,get_col_indices_trinucleotides([feature_header], trinuc)].astype(int)
	region_counts = np.matrix(region_counts)

	return np.sum(np.multiply(region_counts, feature_data), axis=1)

def generate_training_set(vcf_list, hg19, trinuc, features):
	mut_features_all = []
	region_counts_all = []
	region_features_all = []
	header = []

	for i, vcf_file in enumerate(vcf_list):
		if vcf_file.endswith(".vcf"):
			tumour_name = os.path.basename(vcf_file).split(".")[0]
		else:
			print("Only VCF files are accepted: " + vcf_file)
			continue
	
		print("Tumor name: {} ({}/{})".format(tumour_name, i, len(vcf_list)))

		variants_parser = VariantParser(vcf_file, hg19, trinuc,
			chromatin_dict = features['chromatin'], mrna_dict = features['mRNA'])

		mut, low_support = variants_parser.get_variants()
		np.random.shuffle(mut)

		mut_features = variants_parser.get_features(mut[:max_mut_per_tumour], tumour_name)

		header = mut_features[0]
		mut_features_data = mut_features[1:]

		mut_regions = variants_parser.get_region_around_mutation(mut[:max_mut_per_tumour], region_size)

		# region_counts -- counts of mutations there are in the regions that we selected
		# region_features -- features in the region, such as chromatin
		region_counts, region_features = variants_parser.get_region_features(mut_regions, mut_features)

		# take only region counts that correspond to the asked mutation type
		region_counts = get_counts_by_type(region_counts, mut_features_data, header, trinuc)

		mut_features_all.extend(mut_features_data)
		region_counts_all.extend(region_counts)
		region_features_all.extend(region_features)

		n_random = len(region_counts) //2
		print("Generating " + str(n_random) + " random regions")
		mut_features_random = generate_random_mutations(n_random, tumour_name, n_mut_types = len(trinuc[1]))

		# generate the same number of random regions as there are true regions.
		random_regions = get_random_regions(n_random,region_size)

		region_counts_random, region_features_random = variants_parser.get_region_features(random_regions, mut_features)

		# take only region counts that correspond to the asked mutation type
		region_counts_random = get_counts_by_type(region_counts_random, mut_features_random, header, trinuc)

		mut_features_all.extend(mut_features_random)
		region_counts_all.extend(region_counts_random)
		region_features_all.extend(region_features_random)

		print("DONE-{}".format(tumour_name))

	mut_features_all = np.concatenate(([header], mut_features_all))


	mut_features_all = np.matrix(mut_features_all)
	region_counts_all = np.squeeze(np.array(region_counts_all))
	region_features_all = np.array(region_features_all)

	return mut_features_all, region_counts_all, region_features_all


if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='Generate training set for deep model from Ryoga\'s data')
	parser.add_argument('-o', '--output', help='output file', default=DEF_OUTPUT)
	parser.add_argument('-f', '--feature-path', help='feature file path', default=DEF_FEATURE_PATH)
	parser.add_argument('-v', '--vcf-path', help='path to vcf dir', default=DEF_VCF_DIR)
	parser.add_argument('-vf', '--vcf-filter', help='list of vcf files to process', default=DEF_VCF_FILTER)
	parser.add_argument('-rs', '--region-size', help='size of training regions surrounding a mutation', default=1000,type=int)
	parser.add_argument('-ps', '--pickle-size', help='number of training samples per pickle file', default=maxsize_pickle,type=int)
	parser.add_argument('-g', '--group', help='part # of pickle dataset to generate: between 1 and n_vcfs // pickleSize + 1', default =None)

	np.random.seed(1991)

	args = parser.parse_args()
	output_file = args.output
	feature_path = args.feature_path
	vcf_path = args.vcf_path
	vcf_filter = args.vcf_filter
	region_size = args.region_size
	maxsize_pickle = args.pickle_size
	group = None if args.group is None else int(args.group)

	vcf_list = os.listdir(vcf_path)

	if (vcf_filter):
		vcf_list = filter_vcfs(vcf_list, vcf_filter)

	vcf_list = [os.path.join(vcf_path,x) for x in vcf_list]

	try:
		mRNA = load_pickle(os.path.join(feature_path, "mRNA.pickle"))
		trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))
		#alex_signature_file = load_npy(os.path.join(feature_path,"signature.npy"))
		hg19 = load_pickle(os.path.join(feature_path,"hg.pickle"))
		chromatin = load_pickle(os.path.join(feature_path, "chromatin.pickle"))
	except Exception as error:
		raise Exception("Please provide valid compiled feature data files.")

	features= {'chromatin':chromatin, 'mRNA': mRNA}
	
	vcf_group_lists = get_vcfs_in_group(vcf_list, maxsize_pickle, group)

	for i, vcfs in enumerate(vcf_group_lists):
		output_file_part = output_file[:output_file.index(".pickle")] + ".part" + str(i+1) + ".pickle"

		training_set = generate_training_set(vcfs, hg19, trinuc, features) 
		dump_pickle(training_set, output_file_part)
		
		print("DONE! Dataset {} created.".format(output_file_part))
