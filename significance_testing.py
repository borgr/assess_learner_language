import os
import subprocess
import scikits.bootstrap
from m2scorer import m2scorer
import numpy as np
from multiprocessing import Pool
POOL_SIZE = 4
ALTERNATIVE_GOLD_MS = np.arange(10) + 1


def main():
	ACL2016RozovskayaRothOutput_file = "conll14st.output.1cleaned"
	char_based_file = "filtered_test.txt"
	learner_file = "conll.tok.orig"
	amu_file = "AMU"
	cuui_file = "CUUI"
	iitb_file = "IITB"
	ipn_file = "IPN"
	nthu_file = "NTHU"
	pku_file = "PKU"
	post_file = "POST"
	rac_file = "RAC"
	sjtu_file = "SJTU"
	ufc_file = "UFC"
	umc_file = "UMC"
	camb_file = "CAMB"
	gold_file = "corrected_official-2014.0.txt.comparable"
	files = [ACL2016RozovskayaRothOutput_file,
	char_based_file,
	amu_file,
	cuui_file,
	iitb_file,
	ipn_file,
	nthu_file,
	pku_file,
	post_file,
	rac_file,
	sjtu_file,
	ufc_file,
	umc_file,
	camb_file]
	last = False
	correction = 0
	count = 0
	# # calculate the number of sentences with corrections and ratio 
	# with open(r"/home/borgr/ucca/data/conll14st-test-data/noalt/official-2014.combined.m2", "r") as fl:
	# 	for line in fl:
	# 		if last and line.startswith("A"):
	# 			correction += 1
	# 		last = False 
	# 		if line.startswith("S"):
	# 			count +=1
	# 			last = True
	# print(count, correction, 1.0*correction/count)
	# return

	# systems' significance
	pool = Pool(POOL_SIZE)
	results = pool.imap_unordered(m2score_sig, files)
	pool.close()
	pool.join()
	results = [[1,0,0],[1,0,0]] + list(results) + [[1,1,1],[1,1,1]]
	print(results)

	# perfect annotator significance
	perfect_dir = "/home/borgr/ucca/assess_learner_language/calculations_data/"
	pool = Pool(POOL_SIZE)
	files = ["perfect_output_for_" + str(m) + "_sgss.m2" for m in ALTERNATIVE_GOLD_MS]
	gold_files = [perfect_dir + str(m) + "_sgss.m2" for m in ALTERNATIVE_GOLD_MS]
	input_dirs = [perfect_dir] * len(files)
	results = pool.imap_unordered(m2score_sig_in_one,list(zip(files, gold_files, input_dirs)))
	pool.close()
	pool.join()
	results = list(results)
	print(results)

def m2score_sig_in_one(tpl):
	if len(tpl) == 2:
		return m2score_sig(tpl[0], tpl[1])
	if len(tpl) == 3:
		return m2score_sig(tpl[0], tpl[1], tpl[2])	
	return m2score_sig(tpl[0], tpl[1], tpl[2], tpl[3])


def m2score(m, system_file=None):
	directory = r"./calculations_data/"
	system_file = system_file if system_file else directory+"perfect_output_for_" + str(m) + "_sgss.m2" 
	gold_file = directory + str(m) + "_sgss.m2"
	print("testing score of " + system_file)
	# load source sentences and gold edits
	source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

	# load system hypotheses
	fin = m2scorer.smart_open(system_file, 'r')
	system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
	fin.close()

	print(len(system_sentences), len(source_sentences), len(gold_edits))
	return m2scorer.get_score(system_sentences, source_sentences, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False)

def m2score_sig(filename, gold_file=r"/home/borgr/ucca/data/conll14st-test-data/noalt/official-2014.combined.m2", input_dir = r"/home/borgr/ucca/data/paragraphs/", output_dir = r"/home/borgr/ucca/assess_learner_language/results/significance/"):
	system_file = input_dir + filename
	n_samples = 1000
	print("testing significance of " + filename)
	# load source sentences and gold edits
	source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

	# load system hypotheses
	fin = m2scorer.smart_open(system_file, 'r')
	system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
	fin.close()

	statfunction = lambda source, gold, system: m2scorer.get_score(system, source, gold, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False)
	data = (source_sentences, gold_edits, system_sentences)
	test_significance(statfunction, data, output_dir + str(n_samples) + "_" + filename, n_samples=n_samples)

def test_significance(statfunction, data, filename=None, alpha=0.05, n_samples=100, method='bca', output='lowhigh', epsilon=0.001, multi=True):
	""" checks the confidence rate of alpha over n_samples based on the empirical distribution data writes to file the results.
		if filename already exists its content is considered to be the results of the function
		if filename is None the results are not save to any file"""
	if filename == None:
		res = scikits.bootstrap.ci(data, statfunction, alpha, n_samples, method, output, epsilon, multi)
	elif not os.path.isfile(filename):
		print("calculating for " + str(filename))
		res = scikits.bootstrap.ci(data, statfunction, alpha, n_samples, method, output, epsilon, multi)
		with open(filename, "w") as fl:
			fl.write(str(res))
	else:
		with open(filename, "r") as fl:
			res = fl.readlines()
	print(filename, "results:", res)
	return res

if __name__ == '__main__':
	main()