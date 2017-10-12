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

DEF_OUTPUT = "/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.pickle"
DEF_VCF_DIR = "/home/ryogali/data/variants/"
DEF_FEATURE_PATH = "/home/yulia/mnt/dna_features_ryoga/"
binsize = 10**6
maxsize_pickle = 500

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
			main_logger.debug("Only VCF files are accepted: " + vcf_file)
			continue
	
		print("Processing " + tumour_name + " (" + str(vcf_list.index(vcf_file)) + "/" + str(len(vcf_list)) + ")")
		main_logger.info("Tumor name: %s", tumour_name)

		variants_parser = VariantsFileParser(vcf_file, hg19_file, trinuc)

		mut, low_support = variants_parser._get_variants()

		print(mut.shape)

		variant_features = variants_parser._get_features(mut)

		counts = get_counts_per_bin(variant_features, binsize, trinuc)
		counts_all_tumours.append(counts)

		tumour_names.append(tumour_name)

		main_logger.info("DONE-%s",tumour_name)

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

	
	#logging.config.fileConfig("logging.conf")
	main_logger = logging.getLogger("generate_data")

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
		# filter tumors by the list provided in vcf_filter file
		with open(vcf_filter) as file:
			filter = [x.strip("\"\"\n") for x in file.read().splitlines()]
		vcf_list = [x for x in vcf_list if x.split(".")[0] in filter]

	vcf_list = [os.path.join(vcf_path,x) for x in vcf_list]

	# some tumours are too big to load and are causing memory error
	nasty_tumours = ["2df02f2b-9f1c-4249-b3b4-b03079cd97d9",
						"b07bad52-d44c-4b27-900a-960985bfadec",
						"98e8f23c-5970-4fce-9551-4b11a772fe1b",
						"bcf858fd-cc3b-4fde-ab10-eb96216f4366",
						"8853cbee-7931-49a6-b063-a806943a10ad",
						"2790b964-63e3-49aa-bf8c-9a00d3448c25"]

	tumour_names = [get_tumour_name(vcf) for vcf in vcf_list]
	# for nasty in nasty_tumours:
	# 	if nasty in tumour_names:
	# 		ind = tumour_names.index(nasty)
	# 		vcf_list = vcf_list[:ind] + vcf_list[(ind+1):]
	# 		tumour_names = [get_tumour_name(vcf) for vcf in vcf_list]

	try:
		trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))
		#alex_signature_file = load_npy(os.path.join(feature_path,"signature.npy"))
		hg19 = load_pickle(os.path.join(feature_path,"hg.pickle"))
	except Exception as error:
		main_logger.exception("Please provide valid compiled feature data files.")
		exit()

	# due to large size of training set, training set is stored in several parts
	n_pickle_files = len(vcf_list) // maxsize_pickle + 1

	if group is None:
		group = range(0, len(vcf_list) // maxsize_pickle + 1)
	else:
		if group > len(vcf_list) // maxsize_pickle + 1:
			print("Group # should be between 1 and {}".format(len(vcf_list) // maxsize_pickle + 1))
			exit()

		group = [group-1]

	for i in group:
		vcfs = vcf_list[i * maxsize_pickle : (i+1)*maxsize_pickle]
		output_file_part = output_file[:output_file.index(".pickle")] + ".part" + str(i+1) + ".pickle"

		training_set, tumour_names = generate_training_set(vcfs, hg19, trinuc, binsize) 
		dump_pickle([training_set,tumour_names], output_file_part)

		main_logger.info("DONE! Dataset %s created.", output_file_part)
