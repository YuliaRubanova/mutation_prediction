#!/usr/bin/python
# coding=utf-8
#v2.1
import time
import argparse
import logging.config
import os
from read_files import *
from parse_variants import *
from helpers import *

##################################################################
# Generate an input tensor with three dimensions.
# First dimension: DNA location. DNA is splitted into bins of length `binsize`. The first dimension are those bins.
# Second dimension: 96 mutation types
# Third dimension: tumour

# Each cell of tensor is the number of mutations within a specific DNA region (bin) of a specific type in one tumour.
##################################################################

DEF_OUTPUT = "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.pickle"
DEF_VCF_DIR = "/home/ryogali/data/variants/"
DEF_FEATURE_PATH = "/home/yulia/mnt/dna_features_ryoga/"

binsize = 10**6
maxsize_pickle = 500

def get_col_indices_trinucleotides(variant_features, trinuc):
	trinuc_dict, trinuc_list = trinuc

	colnames = list(variant_features[0,])	
	return([colnames.index(tr) for tr in trinuc_list])


def get_counts_per_bin(variant_features, binsize, trinuc):
	counts_in_bins = []

	chr_list = ["chr" + str(i) for i in range(1,22)]

	trinuc_indices = get_col_indices_trinucleotides(variant_features, trinuc)

	for chr in chr_list:
		variants_chr = variant_features[variant_features[:,0] == chr,:]
		n_bins = chromosome_lengths[chr] // binsize + 1

		for bin in range(n_bins):
			pos = variants_chr[:,1].astype(int)
		
			counts_bin = [0] * len(trinuc_indices)
			pos_bin = np.where((pos >= bin*binsize) & (pos < (bin+1)*binsize))[0]

			if len(pos_bin) != 0:
				variants_bin =  np.array(variants_chr[pos_bin,:])

				counts_bin = np.sum(variants_bin[:,trinuc_indices].astype(int), axis = 0)

			counts_in_bins.append(counts_bin)

	return(counts_in_bins)

def get_tumour_name(vcf_file):
	return os.path.basename(vcf_file).split(".")[0]

def generate_training_set(vcf_list, hg19_file, trinuc, binsize):
	counts_all_tumours = []
	tumour_names = []

	for vcf_file in vcf_list:
		if vcf_file.endswith(".vcf"):
			tumour_name = get_tumour_name(vcf_file)
		else:
			print("Only VCF files are accepted: " + vcf_file)
			continue
	
		print("Processing " + tumour_name + " (" + str(vcf_list.index(vcf_file)) + "/" + str(len(vcf_list)) + ")")
		print("Tumor name: %s", tumour_name)

		variants_parser = VariantParser(vcf_file, hg19_file, trinuc)

		mut, low_support = variants_parser.get_variants()

		variant_features = variants_parser.get_features(mut)

		counts = get_counts_per_bin(variant_features, binsize, trinuc)
		counts_all_tumours.append(counts)

		tumour_names.append(tumour_name)

		print("DONE-%s",tumour_name)

	return(np.asarray(counts_all_tumours), tumour_names)


if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='Generate training set for deep model from Ryoga\'s data')
	parser.add_argument('-o', '--output', help='output file', default=DEF_OUTPUT)
	parser.add_argument('-v', '--vcf-path', help='path to vcf dir', default=DEF_VCF_DIR)
	parser.add_argument('-f', '--feature-path', help='feature file path', default=DEF_FEATURE_PATH)
	parser.add_argument('-vf', '--vcf-filter', help='list of vcf files to process', default=None)
	parser.add_argument('-rs', '--bin-size', help='size of DNA binning', default=binsize,type=int)
	parser.add_argument('-ps', '--pickle-size', help='number of training samples per pickle file', default=maxsize_pickle,type=int)
	parser.add_argument('-g', '--group', help='part # of pickle dataset to generate: between 1 and n_vcfs // pickleSize + 1', default =None)


	args = parser.parse_args()
	output_file = args.output
	feature_path = args.feature_path
	vcf_path = args.vcf_path
	vcf_filter = args.vcf_filter
	binsize = args.bin_size
	maxsize_pickle = args.pickle_size
	group = None if args.group is None else int(args.group)

	vcf_list = os.listdir(vcf_path)

	if (vcf_filter):
		vcf_list = filter_vcfs(vcf_list, vcf_filter)
		
	vcf_list = [os.path.join(vcf_path,x) for x in vcf_list]
	tumour_names = [get_tumour_name(vcf) for vcf in vcf_list]

	try:
		trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))
		#alex_signature_file = load_npy(os.path.join(feature_path,"signature.npy"))
		hg19 = load_pickle(os.path.join(feature_path,"hg.pickle"))
	except Exception as error:
		raise Exception("Please provide valid compiled feature data files.")
		exit()


	vcf_group_lists = get_vcfs_in_group(vcf_list, maxsize_pickle, group)

	for i, vcfs in enumerate(vcf_group_lists):
		output_file_part = output_file[:output_file.index(".pickle")] + ".part" + str(i+1) + ".pickle"

		training_set, tumour_names = generate_training_set(vcfs, hg19, trinuc, binsize) 
		dump_pickle([training_set,tumour_names], output_file_part)

		print("DONE! Dataset %s created.", output_file_part)
