#!/usr/bin/python
# coding=utf-8
# v2.0
from read_files import *
import os
import fnmatch
import argparse
import logging.config

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def load_npy(filename):
	filecontent = np.load(filename)
	return filecontent

def save_as_matrix(filename):
	"""
	save file as matrix
	:param filename: 
	:return: 
	"""
	matrix = []
	with open(filename, "r") as mix:
		for line in mix:
			line = line.strip().split(",")
			line = [i.strip("\"\"\n") for i in line]
			matrix.append(line)
	return np.asarray(matrix)

def find(pattern, path):
	"""
	Find file in path that with filename matches pattern
	:param pattern:
	:param path:
	:return: file path
	"""
	result = None
	for root, dirs, files in os.walk(path):
		for name in files:
			if fnmatch.fnmatch(name, pattern):
				result = os.path.join(root, name)
	return result

def write_output_to_file(filename, data):
	# write content in data by line to filename
	with open(filename, "w") as output:
		for item in data:
			output.write("%s\n" % item)

def three_fold_validation():
	# three fold validation
	for variants_file in vcf_list[group]:
		# select only breast cancer
		bc = []
		with open("/home/ryogali/dev/prob_model/BRCA_files.txt") as files:
			for line in files:
				line = line.strip("\"\"\n")
				bc.append(line)

		if variants_file.endswith(".vcf"):
			tumour_name = variants_file.split(".")[0]
		else:
			main_logger.debug("Only VCF files are accepted.")
			continue

		if tumour_name not in bc: # not a brca tumor
			print("not bc")
			continue
		# # find mixture directory
		# mixture = None
		# mixture_path = os.path.join(mixture_dir, tumour_name)
		# # check if tumour exist in mixture path
		# if os.path.exists(mixture_path):
		# 	mixture = find("mixtures.csv", mixture_path)
		# # check if mixture.csv found
		# if mixture == None:
		# 	continue
		#
		# main_logger.info("Tumor name: %s", tumour_name)
		# # get signatures and time points
		# sigs, mixture = read_mix_file(mixture)
		# main_logger.info("%s signatures analyzed", str(len(sigs)))

		# to do get new mixture here
		sigs = []
		mixture = []
		# convert file to matrix
		mixture_matrix = save_as_matrix(mixture_overall)
		# print(mixture_matrix)
		# print(tumour_name)
		# select tumor name from the matrix
		tumor_sig = mixture_matrix[(mixture_matrix[:,0]==tumour_name)|(mixture_matrix[:,0]=="")]
		# print(tumor_sig)
		if tumor_sig.shape[0]<2:
			continue
		# select where signatures != 0
		for i in range(len(tumor_sig[1])):
			# print(i)
			if tumor_sig[1][i] != "0":
				# print(tumor_sig[1][i])
				sigs.append(tumor_sig[0][i])
				mixture.append(tumor_sig[1][i])
		for i in range(len(sigs)):
			sigs[i] = "Signature " + sigs[i]

		mixture = [float(i) for i in mixture[1:]]
		# print(sigs)
		# print(mixture)

		##################################################################
		##################################################################

		vf = os.path.join(input_dir, variants_file)
		variants_parser = VariantsFileParser(vf, chromatin_file, mRNA_file, hg19_file, trinuc, mixture, alex_signature_file, sigs)

		# get input data to the model
		# n = n fold validation, 1/n as train data and 2/n as test data
		# low support data are those mutations that has FILTER = LOWSUPPORT
		test, train, low_support = variants_parser._get_input_data(3)

		# get low support data feature
		low_support_data = variants_parser._get_features(low_support)

		main_logger.info("DONE-%s-%s",tumour_name, str(i))

def one_file_model():
	# todo extract all mixture files in yulia's directory
	# select corresponding validated file
	# open and read validated file into matrix
	validated_file_parser = ValidatedVCFParser(validated_file)
	validated_matrix_data = validated_file_parser._parse()
	# save vcf to matrix based on validated file name
	vf = ""
	for file in os.listdir(input_dir):
		if "84ca6ab0-9edc-4636-9d27-55cdba334d7d" in file:
			vf = os.path.join(input_dir, file)
	if vf == "":
		print("exited here")
		exit(0)
	# to do get new mixture here
	sigs = []
	mixture = []
	# convert file to matrix
	mixture_matrix = save_as_matrix(mixture_overall)
	# print(mixture_matrix)
	# print(tumour_name)
	# select tumor name from the matrix
	tumor_sig = mixture_matrix[(mixture_matrix[:, 0] ==\
							 "84ca6ab0-9edc-4636-9d27-55cdba334d7d") | (mixture_matrix[:, 0] == "")]
	# print(tumor_sig)
	if tumor_sig.shape[0] < 2:
		exit(0)
	# select where signatures != 0
	for i in range(len(tumor_sig[1])):
		# print(i)
		if tumor_sig[1][i] != "0":
			# print(tumor_sig[1][i])
			sigs.append(tumor_sig[0][i])
			mixture.append(tumor_sig[1][i])
	for i in range(len(sigs)):
		sigs[i] = "Signature " + sigs[i]

	mixture = [float(i) for i in mixture[1:]]
	# get train and test data from matrix
	# 3 fold
	variants_parser = VariantsFileParser(vf, chromatin_file, mRNA_file, hg19_file, trinuc, mixture,
										 alex_signature_file, sigs)


	# train on train data
	test, train, negative = variants_parser._get_input_data(3,validated_matrix_data)
	negative = variants_parser._get_features(negative)
	print(negative.shape)
		# predict on test and negative data
	for i in range(len(train)):
		# get features from train data
		train_data = variants_parser._get_features(train[i])
		print(train_data.shape)
		# get features from test data
		test_data = variants_parser._get_features(test[i])
		print(test_data.shape)
		# train the model
		train_matrix = ProbModel(train_data)
		train_matrix._fit()

		# predict probabilities for train data
		train_pa, train_pt, train_ps = \
			train_matrix._predict_proba(train_matrix._mut,train_matrix._tr_X,train_matrix._strand_X,train_matrix._strand)
		test_matrix = ProbModel(test_data)
		test_pa, test_pt, test_ps = \
			train_matrix._predict_proba(test_matrix._mut,test_matrix._tr_X,test_matrix._strand_X,test_matrix._strand)
		# predict probabilities for low_sup data
		low_support_matrix = ProbModel(negative)
		lowsup_pa, lowsup_pt, lowsup_ps = train_matrix._predict_proba(low_support_matrix._mut, low_support_matrix._tr_X,low_support_matrix._strand_X, low_support_matrix._strand)
		print(train_pa)
		print(lowsup_pa)
		print(test_pa)
		# write the probabilities to file
		write_output_to_file(os.path.join(train_prob_dir, validated_file) + "." + str(i) + ".train.txt",train_matrix._calculate_proba(train_pa, train_pt, train_ps))
		write_output_to_file(os.path.join(test_prob_dir, validated_file) + "." + str(i) + ".test.txt", test_matrix._calculate_proba(test_pa, test_pt, test_ps))
		write_output_to_file(os.path.join(lowsup_prob_dir, validated_file) + "." + str(i) + ".neg.txt",low_support_matrix._calculate_proba(lowsup_pa, lowsup_pt, lowsup_ps))
		
		main_logger.info("DONE-%s-%s", validated_file, str(i))

if __name__ == '__main__':
	# all mixture in one file
	mixture_overall = "/home/ryogali/dev/prob_model/data/overall_exposures.sigsBeta2.csv"

	# logging configration
	parser = argparse.ArgumentParser(description='Predict mutation probabilities')
	parser.add_argument('-o', '--output', help='output directory', required=True)
	parser.add_argument('-c', '--chromatin', help='chromatin', required=False)
	parser.add_argument('--file', help='input file', required=False)
	parser.add_argument('-i', '--input', help='input directory', required=True)
	parser.add_argument('-f', '--features', help="feature file path", required=True)
	parser.add_argument('-m', '--mixture', help='mixture files, associated with test files', required=True)
	parser.add_argument('-p', '--psub', help='psub file, use to filter out train data', required=False)
	parser.add_argument('-se', '--signature', help='signature exposure files, associated with train files',
						required=False)

	parser.add_argument('--group', default=-1,type=int, required=False)
	args = parser.parse_args()

	logging.config.fileConfig("logging.conf")
	main_logger = logging.getLogger("main")

	##################################################################
	##################################################################

	OUTPUTDIR = args.output
	chromatin = args.chromatin
	train_prob_dir = os.path.join(OUTPUTDIR, "train_prob/")
	test_prob_dir = os.path.join(OUTPUTDIR, "test_prob/")
	random_prob_dir = os.path.join(OUTPUTDIR, "random_prob/")
	lowsup_prob_dir = os.path.join(OUTPUTDIR, "lowsup_prob/")

	main_logger.info("output files will be saved into: %s", OUTPUTDIR)

	##################################################################
	##################################################################
	feature_data = args.features
	try:
		mRNA_file = load_pickle(os.path.join(feature_data, "mRNA.pickle"))
		#main_logger.info("mRNA loaded")
		trinuc = load_pickle(os.path.join(feature_data,"trinucleotide.pickle"))
		#main_logger.info("trinuc loaded")
		alex_signature_file = load_npy(os.path.join(feature_data,"signature.npy"))
		#main_logger.info("alex_signature loaded")
		hg19_file = load_pickle(os.path.join(feature_data,"hg.pickle"))
		#main_logger.info("hg file loaded")
		if chromatin == "True":
			chromatin_file = load_pickle(os.path.join(feature_data, "chromatin.pickle"))
			#main_logger.info("chromatin loaded")
		else:
			#main_logger.info("Chromatin info will not be analyzed")
			chromatin_file = False
	except Exception as error:
		main_logger.exception("Please provide valid compiled feature data files.")
		exit()

	##################################################################
	##################################################################

	input_dir = args.input  # input vcf files
	mixture_dir = args.mixture  # input mixture files
	psub_dir = args.psub  # phi/vaf values associated with each tumor

	all_vcf = os.listdir(input_dir)

	# following code is for paralizing jobs
	if args.group is not None:
		group = args.group
		GROUP_SIZE = 147 #change this based on cores
		vcf_list = [all_vcf[i:i + GROUP_SIZE] for i in xrange(0, len(all_vcf), GROUP_SIZE)]
	else:
		group = 0
		vcf_list = [all_vcf]
	# end

	##################################################################
	##################################################################
#	three_fold_validation()
	##################################################################
	##################################################################

	# one file
	validated_file = args.file
	one_file_model()



