import os
import subprocess
import scikits.bootstrap
from m2scorer import m2scorer
import numpy as np
from multiprocessing import Pool

def main():
	ACL2016RozovskayaRothOutput_file = "conll14st.output.1cleaned"
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
	learner_file,
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
	camb_file,
	gold_file]
	pool = Pool(3)
	results = pool.imap(m2score_sig, files)
	pool.close()
	pool.join()
	print(list(results))
	# m2score_sig(files[0])
count=0
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
	# test_significance(statfunction, data, output_dir + str(n_samples)+"_2changes" + filename, n_samples=n_samples)
	n_samples = 1403
	def temp(x,y,z):
		global count
		count += 1
		print (count)
		print(x[0],y[0],z[0])
		return np.random.rand(3)
	test_significance(temp, data, None, n_samples=n_samples)



def test_significance(statfunction, data, filename=None, alpha=0.05, n_samples=100, method='bca', output='lowhigh', epsilon=0.001, multi=True):
	""" checks the confidence rate of alpha over n_samples based on the empirical distribution data writes to file the results.
		if filename already exists its content is considered to be the results of the function
		if filename is None the results are not save to any file"""
	print("calculating for " + str(filename))
	if filename == None:
		res = scikits.bootstrap.ci(data, statfunction, alpha, n_samples, method, output, epsilon)
	elif not os.path.isfile(filename):
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