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
# Each training example corresponds to one region. 

# An example contains mutation position and binary representation of mutation type. 
# Region annotation is unloaded dynamically during training.
##################################################################

maxsize_pickle = 1
max_mut_per_tumour = 1000

DEF_FEATURE_PATH = "/home/yulia/mnt/dna_features_ryoga/"
DEF_OUTPUT = "/home/yulia/mnt/mutation_prediction_data/region_dataset.mutTumour" + str(max_mut_per_tumour) + ".mutations_only.pickle"
DEF_VCF_DIR = "/home/ryogali/data/variants/"
DEF_VCF_FILTER = "/home/ryogali/dev/prob_model/BRCA_files.txt"

def generate_random_mutations(n_random, tumour_name, colnames, n_mut_types):
	tumour_names = [[tumour_name]] * n_random
	chrom = [[np.random.choice(list(chromosome_lengths.keys()))] for i in range(n_random)]
	pos = [[np.random.randint(chromosome_lengths[chrom[i][0]])] for i in range(n_random)]
	vaf = [[np.random.uniform()] for i in range(n_random)]

	mut_types = [np.random.multinomial(1, [1/float(n_mut_types)]*n_mut_types) for i in range(n_random)]

	chromatin = np.array([[np.random.choice([0,1])] for i in range(n_random)])
	transcribed = np.array([[np.random.choice([0,1])] for i in range(n_random)])
	strand = np.array([[np.random.choice([0,1])] for i in range(n_random)])

	strand[np.where(transcribed == 0)] = [-1]

	random_mutations = combine_column([tumour_names, chrom, pos, vaf, mut_types, chromatin, transcribed, strand])
	return pd.DataFrame(random_mutations, columns = colnames)

def generate_training_set(vcf_list, hg19, trinuc, features_chromatin_mRNA, other_features):
	mut_features_all = pd.DataFrame()
	region_counts_all = []
	header = []

	for i, vcf_file in enumerate(vcf_list):
		if vcf_file.endswith(".vcf"):
			tumour_name = os.path.basename(vcf_file).split(".")[0]
		else:
			print("Only VCF files are accepted: " + vcf_file)
			continue
	
		print("Tumor name: {} ({}/{})".format(tumour_name, i, len(vcf_list)))

		variants_parser = VariantParser(vcf_file, hg19, trinuc,
			chromatin_dict = features_chromatin_mRNA['chromatin'], mrna_dict = features_chromatin_mRNA['mRNA'],
			other_features = other_features)

		mut, low_support = variants_parser.get_variants()
		np.random.shuffle(mut)

		mut = mut[:max_mut_per_tumour]
		mut_features = variants_parser.get_features(mut, tumour_name)

		region_counts = [[1]]* len(mut_features)

		mut_features_all = pd.concat([mut_features_all, mut_features])
		region_counts_all.extend(region_counts)
		
		n_random = len(mut_features) //2
		print("Generating " + str(n_random) + " random regions")
		mut_features_random = generate_random_mutations(n_random, tumour_name, mut_features.columns.values, n_mut_types = len(trinuc[1]))

		region_counts_random = [[0]]* len(mut_features_random)

		mut_features_all = pd.concat([mut_features_all, mut_features_random])
		region_counts_all.extend(region_counts_random)
	
		print("DONE-{}".format(tumour_name))

	region_counts_all = np.squeeze(np.array(region_counts_all))

	return mut_features_all, region_counts_all


if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='Generate training set for deep model from Ryoga\'s data')
	parser.add_argument('-o', '--output', help='output file', default=DEF_OUTPUT)
	parser.add_argument('-f', '--feature-path', help='feature file path', default=DEF_FEATURE_PATH)
	parser.add_argument('-v', '--vcf-path', help='path to vcf dir', default=DEF_VCF_DIR)
	parser.add_argument('-vf', '--vcf-filter', help='list of vcf files to process', default=DEF_VCF_FILTER)
	parser.add_argument('-ps', '--pickle-size', help='number of training samples per pickle file', default=maxsize_pickle,type=int)
	parser.add_argument('-g', '--group', help='part # of pickle dataset to generate: between 1 and n_vcfs // pickleSize + 1', default =None)

	np.random.seed(1991)

	args = parser.parse_args()
	output_file = args.output
	feature_path = args.feature_path
	vcf_path = args.vcf_path
	vcf_filter = args.vcf_filter
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

		features_chromatin_mRNA= {'chromatin':chromatin, 'mRNA': mRNA}

		other_features = {}
		for file in os.listdir(feature_path):
			if file in ["mRNA.pickle", "trinucleotide.pickle", "hg.pickle", "chromatin.pickle", "signature.npy"]:
				continue
			other_features[file] = load_pickle(os.path.join(feature_path, file))

	except Exception as error:
		raise Exception("Please provide valid compiled feature data files.")

	n_pickle_files = len(vcf_list) // maxsize_pickle + 1

	if group is None:
		group = range(0, len(vcf_list) // maxsize_pickle + 1)
	else:
		if group > len(vcf_list) // maxsize_pickle + 1:
			raise Exception("Group # should be between 1 and {}".format(len(vcf_list) // maxsize_pickle + 1))
		group = [group-1]


	for i in group:
		vcfs = vcf_list[i * maxsize_pickle : (i+1)*maxsize_pickle]
		output_file_part = output_file[:output_file.index(".pickle")] + ".part" + str(i+1) + ".pickle"

		training_set = generate_training_set(vcfs, hg19, trinuc, features_chromatin_mRNA, other_features) 

		dump_pickle(training_set, output_file_part)
		
		print("DONE! Dataset {} created.".format(output_file_part))
