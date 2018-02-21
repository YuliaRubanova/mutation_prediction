#!/usr/bin/python
# coding=utf-8
#v2.0

# save feature files into pickle files to reduce runtime
from __future__ import division
from Bio import SeqIO # pip3 install biopython
import numpy as np
import pickle
import argparse
import os
import shutil

feature_data = "/home/yulia/mnt/dna_features_ryoga/"

def check(var): 
	print((var).get_shape())
	quit()
	
def read_psub(filename):
	"""
	:param filename: chr_pos=phi/vaf
	:return: [chr_pos,phi/vaf]
	"""

	output = []
	with open(filename, "r") as input_file:
		for line in input_file:
			line = line.strip().split("=")
			output.append(line)
	output = np.asarray(output)
	return output

def read_mRNA(filename):
	"""
	output = {chromosome: (start_pos, end_pos, strand)}
	"""
	output = {}
	with open(filename, "r") as input_file:
		for line in input_file:
			line = line.split()
			chrom = line[14]
			if chrom not in output.keys():
				output[chrom] = []
			start_pos = line[16]
			end_pos = line[17]
			strand = line[9]
			output[chrom].append((int(start_pos), int(end_pos), strand))
	
	for key in output.keys():
		output[key] = sorted(output[key])

	with open(feature_data+"mRNA"+".pickle", 'wb') as handle:
		pickle.dump(output, handle, protocol=2)


def read_chromatin(filename):
	"""
	Read a file and convert it into dictionary based on chromosomes
	chromatin: 0, 1, 2
	"""
	output = {}
	with open(filename, "r") as input_file:
		for line in input_file:
			line = line.split()
			chrom = line[0]
			if chrom not in output.keys():
				output[chrom] = []
			start_pos = line[1]
			end_pos = line[2]
			output[chrom].append((int(start_pos), int(end_pos)))
	
	for key in output.keys():
		output[key] = sorted(output[key])

	with open(feature_data+"chromatin"+".pickle", 'wb') as handle:
		pickle.dump(output, handle, protocol=2)

def read_data_1000scale(filename, save_to):
	"""
	Read a file and and convert the 5th coumn into scale 0-1
	"""
	output = {}
	with open(filename, "r") as input_file:
		for line in input_file:
			line = line.split()
			chrom = line[0]
			if chrom not in output.keys():
				output[chrom] = []
			start_pos = line[1]
			end_pos = line[2]
			score = line[4]
			output[chrom].append((int(start_pos), int(end_pos), float(score)/1000.0))

	for key in output.keys():
		output[key] = sorted(output[key])

	with open(save_to, 'wb') as handle:
		pickle.dump(output, handle, protocol=2)

def read_tri(trinucleotide):
	"""
	Read trinucleotide file and assign a class number to each mutation type.
	[0-95] for 96 types of mutations
	"""
	triClass_dict = {}
	triClass_list = []
	i = 0
	with open(trinucleotide, "r") as t:
		for line in t:
			line = line.split()
			triClass_dict[(line[0], line[1], line[2])] = i
			triClass_list.append(line[0] + "_" + line[1] + "_" + line[2])
			i += 1
	# save dictionary
	with open(feature_data+"trinucleotide"+".pickle", 'wb') as handle:
		pickle.dump([triClass_dict,triClass_list], handle, protocol=2)


def read_se_file(signature_file):
	# used for the first version of signature probability file
	"""
	convert se prob file into a numpy matrix
	signature prob file content:
	chr | pos | phi | ref | alt | context | S1 | S2 | ... | prob 
	"""
	matrix = []
	with open(signature_file,"r") as se:
		# se.readline()
		for line in se:
			# print line
			line = line.strip().split('\t')
			matrix.append(line)
	matrix = np.asarray(matrix)
	# np.save("signature_exposure.npy", matrix)
	return matrix


def read_mix_file(phi_file):
	"""
	Convert phi file into a numpy matrix
	"""
	matrix = []
	s = []
	with open(phi_file, "r") as phi:
		# check what does phi file look like 
		for line in phi:
			line = line.strip().split(",")
			if len(line[0]) > 2:
				s.append(line[0].strip("\"\"\n"))
			line = line[1:]
			# print line
			line = [float(i.strip("\"\"\n")) for i in line]
			matrix.append(line)
	matrix = np.asarray(matrix)
	# make sure s here has the same name with exposure file
	for i in range(len(s)):
		s[i] = "Signature " + s[i]
	# np.save("phi.npy", matrix)
	return s, matrix


def read_alexSignature(alex_signature):
	"""
	:return file_content: a list contains 96 entries corresponding to 96 lines in signature_prob file
	"""
	# convert the se file into a list
	# index corresponding to tri nucleotide
	# len(file_content) = 96
	file_content = []
	with open(alex_signature, "r") as sp:
		for line in sp:
			line = line.strip().split(",")
			file_content.append(line)
			# for the older version
			# start = 0
			# end = 30
			# for i in range(len(line)):
			# 	if end > 2880:
			# 		break
			# 	l = line[start:end]
			# 	file_content.append(l)
			# 	start = end
			# 	end += 30
	file_content = np.asarray(file_content)
	np.save(feature_data+"signature.npy", file_content)
	return file_content


def read_hg19(hg19_file):

	# open hg19.fa
	refSeq = SeqIO.parse(open(hg19_file,'r'), 'fasta')
	# store the sequence into dic
	chromSeqPair = {}

	for chro in refSeq:
		chromSeqPair[chro.id[3:]] = str(chro.seq)

	# save dictionary
	with open(feature_data+"hg"+".pickle", 'wb') as handle:
		pickle.dump(chromSeqPair, handle,protocol=2)


def read_vaf(vaf_file):
	matrix = []
	with open(vaf_file,"r") as vaf:
		for line in vaf:
			line=line.strip().split("=")
			matrix.append(line)
	matrix = np.asarray(matrix)
	return matrix


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Generate feature files')
	parser.add_argument('-m', '--mRNA', help='mRNA file, all_mRNA_hg19.txt')
	#/home/q/qmorris/ryogali/data/all_mrna_hg19.txt
	parser.add_argument('-s', '--Signatures', help='signature file')
	#/home/q/qmorris/ryogali/data/alexSignatures.txt
	parser.add_argument('-hg', '--humanGenome', help='Human genome file, hg19.fa')
	#/home/q/qmorris/ryogali/data/hg19.fa
	parser.add_argument('-tri', '--trinucleotide', help='trinucleotide file')
	#/home/q/qmorris/ryogali/data/trinucleotide.txt
	parser.add_argument('-c', '--chromatin', help='Chromatin file, cancer type specific', required=False)
	#/home/q/qmorris/ryogali/data/ENCFF001UVV.bed
	parser.add_argument('-o', '--other', help='Folder with histone marks, methylation, etc., cancer type specific', required=False)
	#/home/yulia/mnt/mutation_prediction_data/breast/
	
	args = parser.parse_args()

	mRNA_file = args.mRNA
	trinuc = args.trinucleotide
	alex_signature_file = args.Signatures
	hg19_file = args.humanGenome
	other_data_folder = args.other

	chromatin_file = args.chromatin
	
	if not os.path.exists(feature_data):
		os.makedirs(feature_data)
	else:
		shutil.rmtree(feature_data)
		os.makedirs(feature_data)

	read_mRNA(mRNA_file)
	read_tri(trinuc)
	read_alexSignature(alex_signature_file)
	read_hg19(hg19_file)
	if args.chromatin:
		read_chromatin(chromatin_file)

	for file in os.listdir(other_data_folder):
		read_data_1000scale(os.path.join(other_data_folder,file), os.path.join(feature_data,file + ".pickle"))

	print("Processed feature files are saved into: " + feature_data)
	print([i for i in os.listdir(feature_data)])


	
