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
# Generate training set for deep learning model from Ryoga's dataset from BRCA samples
# Each training example corresponds to one region. An example contains binary representation of 
# mutations at each position in the region and genomic annotation of chromatin, etc. in this region.
##################################################################

DEF_FEATURE_PATH = "/home/yulia/mnt/dna_features_ryoga/"
DEF_OUTPUT = "/home/yulia/mnt/predict_mutations/region_dataset.pickle"
DEF_VCF_DIR = "/home/ryogali/data/variants/"
DEF_VCF_FILTER = "/home/ryogali/dev/prob_model/BRCA_files.txt"

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


"""
TO IMPLEMENT

	for vcf
		get_mutation_set()
		get_region_around_mutations()
		get_random_regions()

		get_features_for_regions()

+ making more clear classes
"""	

def get_random_regions(n, reg_size):
	regions = []
	for i in range(n):
		np.random.seed(n)
		chr = 'chr' + str(np.random.choice(len(chromosome_lengths)))
		start = int(np.random.uniform(0, chromosome_lengths[chr] - reg_size))
		regions.append((start, start+reg_size))

	return regions

def get_region_around_mutation(mut, reg_size):
	pass

def generate_training_set(vcf_list, hg19_file, trinuc, features):
	for vcf_file in vcf_list:
		if vcf_file.endswith(".vcf"):
			tumour_name = os.path.basename(vcf_file).split(".")[0]
		else:
			main_logger.debug("Only VCF files are accepted: " + vcf_file)
			continue
	
		main_logger.info("Tumor name: %s", tumour_name)

		variants_parser = VariantsFileParser(vcf_file, hg19_file, trinuc,
			chromatin_dict = features['chromatin'], mrna_dict = features['mRNA'])

		mut, low_support = variants_parser._get_variants()

		print(get_random_regions(10,10**6))

		print("got variants!")
		print(mut[:10])

		features_region = variants_parser._get_features(mut[:10])

		print("got features!")
		print(features_region[:10])

		main_logger.info("DONE-%s",tumour_name)

		
"""

def fill_the_data_by_region(annot, regions, chromosome_lengths):
	for chr in chromosome_lengths.keys():
		chr = chr.strip()
		regions_chr = regions[chr]
		start = time.time()

		# fill all zeros up to the start of the first region
		annot[chr] = [0] * regions_chr[0][0]
		for reg in regions_chr:
			start, end = reg
			if len(annot[chr]) < end:
				annot[chr] = annot[chr] + [0]* (end - len(annot[chr]))
				assert(len(annot[chr]) == end)

			annot[chr][start:(end+1)] = [1] * (end-start)

		if len(annot[chr]) < chromosome_lengths[chr]:
			annot[chr] = annot[chr] + [0]* (chromosome_lengths[chr] - len(annot[chr]))

		assert(len(annot[chr]) == chromosome_lengths[chr])
		
		print(annot[chr][:10])
		print(time.time() - start)

"""



if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='Generate training set for deep model from Ryoga\'s data')
	parser.add_argument('-o', '--output', help='output file', default=DEF_OUTPUT)
	parser.add_argument('-f', '--feature-path', help='feature file path', default=DEF_FEATURE_PATH)
	parser.add_argument('-v', '--vcf-path', help='path to vcf dir', default=DEF_VCF_DIR)
	parser.add_argument('-vf', '--vcf-filter', help='list of vcf files to process', default=DEF_VCF_FILTER)
	parser.add_argument('-rs', '--region-size', help='size of training regions surrounding a mutation', default=1000,type=int)
	
	logging.config.fileConfig("logging.conf")
	main_logger = logging.getLogger("generate_data")

	args = parser.parse_args()
	output_file = args.output
	feature_path = args.feature_path
	vcf_path = args.vcf_path
	vcf_filter = args.vcf_filter

	vcf_list = os.listdir(vcf_path)

	if (vcf_filter):
		# filter tumors by the list provided in vcf_filter file
		with open(vcf_filter) as file:
			filter = [x.strip("\"\"\n") for x in file.read().splitlines()]
		vcf_list = [x for x in vcf_list if x.split(".")[0] in filter]

	vcf_list = [os.path.join(vcf_path,x) for x in vcf_list]

	try:
		mRNA = load_pickle(os.path.join(feature_path, "mRNA.pickle"))
		trinuc = load_pickle(os.path.join(feature_path,"trinucleotide.pickle"))
		#alex_signature_file = load_npy(os.path.join(feature_path,"signature.npy"))
		hg19 = load_pickle(os.path.join(feature_path,"hg.pickle"))
		chromatin = load_pickle(os.path.join(feature_path, "chromatin.pickle"))
	except Exception as error:
		main_logger.exception("Please provide valid compiled feature data files.")
		exit()

	features= {'chromatin':chromatin, 'mRNA': mRNA}
	
	training_set = generate_training_set(vcf_list, hg19, trinuc, features) 
	dump_pickle(training_set, output_file)
	
	main_logger.info("DONE! Dataset %s created.", output_file)
