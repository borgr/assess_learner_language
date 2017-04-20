import codecs
import numpy as np
import time
from m2scorer import m2scorer
import re
import os
from multiprocessing import Pool
POOL_SIZE = 4

def main():
	k_best_dir = "/home/borgr/ucca/data/K-best/"
	system_file = k_best_dir + "conll14st.output.1.best100"
	gold_file = "/home/borgr/ucca/data/conll14st-test-data/noalt/official-2014.combined.m2"
	calculations_dir = "/home/borgr/ucca/assess_learner_language/calculations_data/"
	output_file = "first_rank_results.txt"

	source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

	# load system hypotheses
	fin = m2scorer.smart_open(system_file, 'r')
	system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
	fin.close()

	candidate_num = 0
	packed_system_sentences = []
	for sentence_num, (source, this_edits) in enumerate(zip(source_sentences, gold_edits)):
		curr_sentences = []
		while (candidate_num < len(system_sentences) and
			  system_sentences[candidate_num].split()[0] == str(sentence_num)):
			sentence = re.sub("\|\d+-\d+\| ","",system_sentences[candidate_num].split("|||")[1][1:])
			candidate_num += 1
			curr_sentences.append(sentence)
		packed_system_sentences.append(curr_sentences)

	pool = Pool(POOL_SIZE)
	assert(len(packed_system_sentences) == len(gold_edits) and len(gold_edits) == len(source_sentences))
	results = pool.imap(oracle, zip(source_sentences, gold_edits, packed_system_sentences))
	pool.close()
	pool.join()
	results = list(results)
	sentences = "\n".join(zip(*results)[0])
	results = zip(*results)[1]
	results = "\n".join([str(x) for x in results])
	with codecs.open(calculations_dir + output_file, "w+", "utf-8") as fl:
		fl.write(sentences)
	with open(calculations_dir + "prf_" + output_file, "w+") as fl:
		fl.write(results)

def oracle(tple):
	maximum = 0
	source, this_edits, system_sentences = tple
	for sentence in system_sentences:
		p,r,f = score(source, this_edits, sentence)
		if maximum <= f:
			maximum = f
			chosen = sentence, (p,r,f)
	print(chosen)
	return chosen

def score(source, gold_edits, system):
	return sentence_m2(source, gold_edits, system)

def sentence_m2(source, gold_edits, system):
	return m2scorer.get_score([system], [source], [gold_edits], max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)

if __name__ == '__main__':
	main()