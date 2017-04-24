import codecs
import numpy as np
import time
from m2scorer import m2scorer
import re
import os
from multiprocessing import Pool
POOL_SIZE = 7

def main():
	data_dir = "data/"
	k_best_dir = data_dir + "K-best/"
	system_file = k_best_dir + "conll14st.output.1.best100"

	reference_dir = data_dir + "references/"
	first_nucle =  reference_dir + "NUCLEA.m2"
	combined_nucle = reference_dir + "NUCLE.m2"
	BN = reference_dir + "BN.m2"
	ALL =  reference_dir + "ALL.m2"
	gold_files = [first_nucle, combined_nucle, BN, ALL]


	calculations_dir = "calculations_data/"
	output_file = "first_rank_results"
	for gold_file in gold_files:
		out_text_file = calculations_dir + output_file + name_extension(gold_file)[0]
		out_res_file = calculations_dir + "prf_" + output_file + name_extension(gold_file)[0]
		if not os.path.isfile(out_text_file):
			print("processing " + gold_file)
			source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

			# load system hypotheses
			fin = m2scorer.smart_open(system_file, 'r')
			system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
			fin.close()

			# pack and parse RoRo's k-best
			candidate_num = 0
			packed_system_sentences = []
			for sentence_num, (source, this_edits) in enumerate(zip(source_sentences, gold_edits)):
				curr_sentences = []
				# keep packing until reached another sentence, assumes k-best are consequetive
				while (candidate_num < len(system_sentences) and
					  system_sentences[candidate_num].split()[0] == str(sentence_num)):
					sentence = re.sub("\|\d+-\d+\| ","",system_sentences[candidate_num].split("|||")[1][1:])
					candidate_num += 1
					curr_sentences.append(sentence)
				packed_system_sentences.append(curr_sentences)

			# find top ranking
			pool = Pool(POOL_SIZE)
			assert(len(packed_system_sentences) == len(gold_edits) and len(gold_edits) == len(source_sentences))
			results = pool.imap(oracle, zip(source_sentences, gold_edits, packed_system_sentences))
			pool.close()
			pool.join()
			results = list(results)
			sentences = "\n".join(zip(*results)[0])
			results = zip(*results)[1]
			results = "\n".join([str(x) for x in results])
			
			print("writing to " + out_text_file)
			with codecs.open(out_text_file, "w+", "utf-8") as fl:
				fl.write(sentences)
			with open(out_res_file, "w+") as fl:
				fl.write(results)

def oracle(tple):
	maximum = 0
	source, this_edits, system_sentences = tple
	for sentence in system_sentences:
		p,r,f = score(source, this_edits, sentence)
		if maximum <= f:
			maximum = f
			chosen = sentence, (p,r,f)
	return chosen

def score(source, gold_edits, system):
	return sentence_m2(source, gold_edits, system)

def sentence_m2(source, gold_edits, system):
	return m2scorer.get_score([system], [source], [gold_edits], max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)

def basename(name):
	return name.split("\\")[-1].split("/")[-1]

def name_extension(name):
	return basename(name).split(".")
if __name__ == '__main__':
	main()