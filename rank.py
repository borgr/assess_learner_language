import time
import sys
# UCCA_DIR = '/home/borgr/ucca/ucca'
# ASSESS_DIR = '/home/borgr/ucca/assess_learner_language'
TUPA_DIR = '/cs/labs/oabend/borgr/tupa/'
UCCA_DIR = TUPA_DIR +'ucca'
ASSESS_DIR = '/cs/labs/oabend/borgr/assess_learner_language'
sys.path.append(ASSESS_DIR + '/m2scorer/scripts')
# sys.path.append(UCCA_DIR)
# sys.path.append(UCCA_DIR + '/scripts/distances')
# sys.path.append(UCCA_DIR + '/ucca')
# sys.path.append(TUPA_DIR)
from ucca.ioutil import file2passage
from subprocess import call
import subprocess
import codecs
import numpy as np
# from m2scorer import m2scorer
import re
import os
from multiprocessing import Pool
# import align
import pickle
import json
from functools import reduce
import operator
# from significance_testing import m2score
import platform
from ucca.ioutil import passage2file
from ucca.convert import from_text
from correction_quality import word_diff

from simplification import SARI
import annalyze_crowdsourcing as an

POOL_SIZE = 7
full_rerank = True

# from tupa.parse import Parser
# model_path = "/cs/labs/oabend/borgr/tupa/models/bilstm"
# parser = Parser(model_path, "bilstm")


def main():
	# parse_JFLEG()
	# # rerank_by_m2()
	# for gamma in np.linspace(0,1,11):
	# 	print(m2score(system_file="calculations_data/uccasim_rerank/" + str(gamma) + "_" + "uccasim_rank_results",
	# 				  gold_file=r"/home/borgr/ucca/assess_learner_language/data/references/ALL.m2"))
	# 	# rerank_by_uccasim(gamma)
	# 	rerank_by_uccasim(gamma)
	# print(m2score(system_file=r"/home/borgr/ucca/assess_learner_language/data/paragraphs/conll14st.output.1cleaned",
	# 			  gold_file=r"/home/borgr/ucca/assess_learner_language/data/references/ALL.m2"))
	# reduce_k_best(100, 10, filename)
	# rerank_by_wordist()
	anounce_finish()


def parse_JFLEG():
	JFLEG_dir = ASSESS_DIR + "/data/jfleg/dev"
	(path, dirs, files) = next(os.walk(JFLEG_dir))
	filenames = [path + os.sep + fl for fl in files]
	ucca_parse_files(filenames, JFLEG_dir + os.sep + "xmls")
	


def rerank_by_uccasim(gamma=0.27):
	data_dir = "data/"
	first_nucle =  data_dir + "references/" + "NUCLEA.m2" # only used to extract source sentences
	k_best_dir = data_dir + "K-best/"
	system_file = k_best_dir + "conll14st.output.1.best100"
	calculations_dir = "calculations_data/uccasim_rerank/"
	ucca_parse_dir = calculations_dir + "/ucca_parse/"
	full = "full" if full_rerank else ""
	output_file = full + str(gamma) + "_" + "uccasim_rank_results"
	out_text_file = calculations_dir + output_file
	out_res_file = calculations_dir + "score_" + output_file

	if not os.path.isfile(out_text_file):
		gold_file = first_nucle # only used to extract source sentences
		print("acquiring source")
		source_sentences, _ = m2scorer.load_annotation(gold_file)

		source_sentences = source_sentences
		# load system hypotheses
		fin = m2scorer.smart_open(system_file, 'r')
		system_sentences = [line.strip() for line in fin.readlines()]
		fin.close()

		packed_system_sentences = get_roro_packed(system_sentences)

		print("parsing")
		# print(reduce(operator.add, packed_system_sentences))
		ucca_parse(reduce(operator.add, packed_system_sentences) + source_sentences, ucca_parse_dir)

		print("reranking")
		# find top ranking
		pool = Pool(POOL_SIZE)
		assert(len(packed_system_sentences) == len(source_sentences))
		if full_rerank:
			results = pool.starmap(referece_less_full_rerank, zip(source_sentences, packed_system_sentences, [ucca_parse_dir] * len(packed_system_sentences), [gamma] * len(packed_system_sentences)))
		else:
			results = pool.starmap(referece_less_oracle, zip(source_sentences, packed_system_sentences, [ucca_parse_dir] * len(packed_system_sentences), [gamma] * len(packed_system_sentences)))
		pool.close()
		pool.join()
		results = list(results)
		if full_rerank:
			results = [x for y in results for x in y]
		sentences = "\n".join(list(zip(*results))[0])
		results = list(zip(*results))[1]
		results = "\n".join([str(x) for x in results])
		
		print("writing to " + out_text_file)
		with codecs.open(out_text_file, "w+", "utf-8") as fl:
			fl.write(sentences)
		with open(out_res_file, "w+") as fl:
			fl.write(results)

def rerank_by_wordist():
	data_dir = "data/"
	first_nucle =  data_dir + "references/" + "NUCLEA.m2" # only used to extract source sentences
	k_best_dir = data_dir + "K-best/"
	system_file = k_best_dir + "conll14st.output.1.best100"
	calculations_dir = "calculations_data/uccasim_rerank/"
	# ucca_parse_dir = calculations_dir + "/ucca_parse/"
	min_change = 2
	output_file = str(min_change) + "wordist_rank_results"
	out_text_file = calculations_dir + output_file
	out_res_file = calculations_dir + "score_" + output_file
	out_source_file = calculations_dir + "source" + output_file
	if not os.path.isfile(out_text_file):
		gold_file = first_nucle # only used to extract source sentences
		print("acquiring source")
		source_sentences, _ = m2scorer.load_annotation(gold_file)

		source_sentences = source_sentences
		# load system hypotheses
		fin = m2scorer.smart_open(system_file, 'r')
		system_sentences = [line.strip() for line in fin.readlines()]
		fin.close()

		packed_system_sentences = get_roro_packed(system_sentences)

		# print("parsing")
		# print(reduce(operator.add, packed_system_sentences))
		# ucca_parse(reduce(operator.add, packed_system_sentences) + source_sentences, ucca_parse_dir)

		print("reranking")
		# find top ranking
		pool = Pool(POOL_SIZE)
		assert(len(packed_system_sentences) == len(source_sentences))
		results = pool.starmap(wordist_oracle, zip(source_sentences, packed_system_sentences))
		pool.close()
		pool.join()
		results = list(results)
		tmp = []
		out_sentences = []
		for (k,n), sent in zip(results, source_sentences):
			if n > min_change:
				tmp.append((k,n))
				out_sentences.append(sent)
		results = tmp

		sentences = "\n".join(list(zip(*results))[0])
		results = list(zip(*results))[1]
		results = "\n".join([str(x) for x in results])
		out_sentences = "\n".join([str(x) for x in out_sentences])
		
		print("writing to " + out_text_file)
		with codecs.open(out_text_file, "w+", "utf-8") as fl:
			fl.write(sentences)
		with codecs.open(out_source_file, "w+", "utf-8") as fl:
			fl.write(out_sentences)
		with open(out_res_file, "w+") as fl:
			fl.write(results)

def rerank_by_m2():
	data_dir = "data/"
	k_best_dir = data_dir + "K-best/"
	system_file = k_best_dir + "conll14st.output.1.best100"

	reference_dir = data_dir + "references/"
	first_nucle =  reference_dir + "NUCLEA.m2"
	combined_nucle = reference_dir + "NUCLE.m2"
	BN = reference_dir + "BN.m2"
	ALL =  reference_dir + "ALL.m2"
	gold_files = [first_nucle, combined_nucle, BN, ALL]

	(path, dirs, files) = next(os.walk(reference_dir))
	for fl in files:
		if "subset" in fl:
			gold_files.append(path + fl)

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
			packed_system_sentences = get_roro_packed(source_sentences)
			# candidate_num = 0
			# for sentence_num, (source, this_edits) in enumerate(zip(source_sentences, gold_edits)):
			# 	curr_sentences = []
			# 	# keep packing until reached another sentence, assumes k-best are consequetive
			# 	while (candidate_num < len(system_sentences) and
			# 		  system_sentences[candidate_num].split()[0] == str(sentence_num)):
			# 		sentence = re.sub("\|\d+-\d+\| ","",system_sentences[candidate_num].split("|||")[1][1:])
			# 		candidate_num += 1
			# 		curr_sentences.append(sentence)
			# 	packed_system_sentences.append(curr_sentences)

			# find top ranking
			pool = Pool(POOL_SIZE)
			assert(len(packed_system_sentences) == len(gold_edits) and len(gold_edits) == len(source_sentences))
			results = pool.imap(RBM_oracle, zip(source_sentences, packed_system_sentences))
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


def rerank_by_SARI():
	data_dir = "data/simplification/"
	k_best_dir = data_dir + "K-best/"
	system_file = k_best_dir + "Moses_based"

	DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep + "/simplification/data/"
	TURKERS_DIR = DATA_DIR + "turkcorpus/truecased/"

	ORIGIN = "origin"

	db = []
	for root, dirs, files in os.walk(TURKERS_DIR):
		for filename in files:
			cur_db = pd.read_table(TURKERS_DIR + filename, names=["index", ORIGIN, 1, 2, 3, 4, 5, 6, 7, 8])
			db.append(cur_db)
	db = pd.concat(db, ignore_index=True)
	db.drop("index", inplace=True, axis=1)
	db.dropna(inplace=True, axis=0)
	db.applymap(an.normalize_sentence)
	source_sentences = db[ORIGIN].tolist()
	references = db.iloc[:, -8:].values


	calculations_dir = "calculations_data/"
	output_file = "simplification_rank_results"
	for ref_num in [1, 2, 3, 4, 5, 6, 7, 8]:
		out_text_file = calculations_dir + output_file + str(ref_num) + "refs"
		out_res_file = calculations_dir + "SARI_" + output_file + str(ref_num) + "refs"
		if not os.path.isfile(out_text_file):
			print("ranking with", ref_num, "refs")

			# load system hypotheses

			# pack k-best
			packed_system_sentences = []
			for source, refs, system in zip(source_sentences, references, system_sentences):
				packed_system_sentences.append(source, references, system_sentences[np.random.randint(0, 8, ref_num)].tolist())

			# find top ranking
			pool = Pool(POOL_SIZE)
			assert(len(packed_system_sentences) == len(gold_edits) and len(gold_edits) == len(source_sentences))
			results = pool.imap(RBM_oracle, zip(source_sentences, packed_system_sentences))
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


def reduce_k_best(big_k, small_k, filename, outfile=None):
	if outfile is None:
		outfile = os.path.normpath(filename)
		outfile = os.path.split(outfile)
		outfile[1] = str(small_k) + "_" + outfile[1]
		outfile = "".join(outfile)
	output = []
	with open(outfile) as fl:
		for i, line in enumerate(fl):
			if i % big_k < small_k:
				output.append(line)
	# finish that 
	raise


def referece_less_full_rerank(source, system_sentences, parse_dir, gamma):
	combined_scores = []
	for sentence in set(system_sentences):
		combined_scores.append((sentence, reference_less_score(source, sentence, parse_dir, gamma)))

	return sorted(combined_scores, key=lambda x:x[1])


def wordist_oracle(source, system_sentences):
	maximum = 0
	for sentence in set(system_sentences):
		combined_score = word_diff(source, sentence)
		if maximum <= combined_score:
			maximum = combined_score
			chosen = sentence, combined_score
	return chosen


def referece_less_oracle(source, system_sentences, parse_dir, gamma):
	maximum = 0
	for sentence in set(system_sentences):
		combined_scores = reference_less_score(source, sentence, parse_dir, gamma)
		if maximum <= combined_score:
			maximum = combined_score
			chosen = sentence, combined_score
	# print(chosen)
	return chosen


def RBM_oracle(tple):
	maximum = 0
	source, this_edits, system_sentences = tple
	for sentence in system_sentences:
		p,r,f = score(source, this_edits, sentence)
		if maximum <= f:
			maximum = f
			chosen = sentence, (p,r,f)
	return chosen


def SARI_oracle(tple):
	maximum = 0
	source, references, system_sentences = tple
	for sentence in system_sentences:
		score = SARI_score(source, references, sentence)
		if maximum <= score:
			maximum = score
			chosen = sentence
	return chosen


def ucca_parse_files(filenames, output_dir):
	# parse_command = "python ../tupa/tupa/parse.py -c bilstm -m ../tupa/models/bilstm -o "+ output_dir +" "
	# print("parsing with:", parse_command)

	if filenames:
		for filename in filenames:
			print("parsing " + filename)
			with open(filename, "r") as fl:
				text = fl.readlines()
			text = from_text(text, split=True)
			print(text)
			for i, passage in enumerate(parser.parse(text)):
				passage2file(passage, output_dir + os.sep + os.path.basename(filename) + str(i) + ".xml")
			print("printed all xmls from " + output_dir + os.sep + os.path.basename(filename))
		# res = subprocess.run(parse_command.split() + list(files), stdout=subprocess.PIPE)


def ucca_parse(sentences, output_dir):
	parse_command = "python ../tupa/tupa/parse.py -c bilstm -m ../tupa/models/bilstm -o "+ output_dir +" "
	# print("parsing with:", parse_command)
	filenames = []
	count = 0
	for sentence in list(set(sentences)):
		# print("parsing" + sentence[:20])
		filename = str(get_sentence_id(sentence, output_dir))
		txt_file = filename + ".txt"
		xml_file = filename + ".xml"
		if not os.path.isfile(output_dir + txt_file):
			with open(output_dir + txt_file, "w+") as fl:
				fl.write(sentence)
		if not os.path.isfile(output_dir + xml_file):
			filenames.append(output_dir + txt_file)

	# check = []
	# for sentence in list(set(filenames)):
	# 	if sentence not in check:
	# 		check.append(sentence)
	# 	else:
	# 		print("repeats")
	# 		return
	# print(sorted(filenames))
	if filenames:
		print("parsing sentences")
		res = subprocess.run(parse_command.split() + filenames, stdout=subprocess.PIPE)
	# print(res)
	# call(parse_command.split() + filenames)


def get_roro_packed(system_sentences):
	""" pack and parse RoRo's k-best"""
	candidate_num = 0
	packed_system_sentences = []
	for sentence_num, source in enumerate(system_sentences):
		curr_sentences = []
		# keep packing until reached another sentence, assumes k-best are consequetive
		while (candidate_num < len(system_sentences) and
			  system_sentences[candidate_num].split()[0] == str(sentence_num)):
			sentence = re.sub("\|\d+-\d+\| ","",system_sentences[candidate_num].split("|||")[1][1:])
			candidate_num += 1
			curr_sentences.append(sentence)
		if curr_sentences:
			packed_system_sentences.append(curr_sentences)
	return packed_system_sentences


def semantics_score(source, sentence, parse_dir):
	source_xml = file2passage(parse_dir + str(get_sentence_id(source, parse_dir, False)) + ".xml")
	sentence_xml = file2passage(parse_dir + str(get_sentence_id(sentence, parse_dir, False)) + ".xml")
	return align.fully_aligned_distance(source_xml, sentence_xml)


def grammaticality_score(source, sentence, parse_dir):
	command = "java -jar ../softwares/LanguageTool-3.7/languagetool-commandline.jar --json -l en-US" 
	filename = str(get_sentence_id(sentence, parse_dir, False)) + ".txt"
	with open(os.devnull, 'wb') as devnull:
		res = subprocess.run(command.split() + [parse_dir + filename], stdout=subprocess.PIPE, stderr=devnull)
	out = res.stdout.decode("utf-8")
	out = re.sub(r"\\'", "'", out)
	res = json.loads(out)
	return 1/(1 + len(res["matches"]))


_id_dics = {}
def get_sentence_id(sentence, parse_dir, graceful=True):
	""" returns the sentence id in the parse_dir, 
		if graceful is true adds a new sentence id 
		if the sentence does not exist in the ids list,
		otherwise throws exception"""
	filename = "sentenceIds.pkl"
	max_id = "max"
	if parse_dir in _id_dics:
		id_dic = _id_dics[parse_dir]
	elif not os.path.isfile(parse_dir + os.sep + filename):
		print("creating a new id list")
		id_dic = {max_id: -1}
		_id_dics[parse_dir] = id_dic
	else:
		with open(parse_dir + os.sep + filename, "rb") as fl:
			id_dic = pickle.load(fl)
			_id_dics[parse_dir] = id_dic
	if graceful and not sentence in id_dic:
		# print("dumping" + sentence + "\n")
		id_dic[max_id] += 1
		id_dic[sentence] = id_dic[max_id]
		with open(parse_dir + os.sep + filename, "wb+") as fl:
			pickle.dump(id_dic, fl)
	# print(sentence)
	return id_dic[sentence]


def reference_less_score(source, sentence, parse_dir, gamma):
	return gamma * grammaticality_score(source, sentence, parse_dir) + (1 - gamma) * semantics_score(source, sentence, parse_dir)


def score(source, gold_edits, system):
	return sentence_m2(source, gold_edits, system)


def SARI_score(source, references, system):
	return SARI.SARIsent(system, source, references)


def sentence_m2(source, gold_edits, system):
	return m2scorer.get_score([system], [source], [gold_edits], max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)


def basename(name):
	return name.split("\\")[-1].split("/")[-1]


def name_extension(name):
	return basename(name).split(".")

def anounce_finish():
	if sys.platform == "linux":
		if set(("debian", "Ubuntu")) & set(platform.linux_distribution()):
			subprocess.call(['speech-dispatcher'])        #start speech dispatcher
			subprocess.call(['spd-say', '"your process has finished"'])
		else:
			#perhaps works only in ubuntu?
			a = subprocess.Popen(('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 300, 2)).split())
	elif sys.platform == "darwin":
		subprocess.call('say "your process has finished"'.split())
	else:
		import winsound
		winsound.Beep(300,2)

if __name__ == '__main__':
	fnamenorm   = "./turkcorpus/test.8turkers.tok.norm"
	fnamesimp   = "./turkcorpus/test.8turkers.tok.simp"
	fnameturk  = "./turkcorpus/test.8turkers.tok.turk."


	ssent = "About 95 species are currently accepted ."
	csent1 = "About 95 you now get in ."
	csent2 = "About 95 species are now agreed ."
	csent3 = "About 95 species are currently agreed ."
	rsents = ["About 95 species are currently known .", "About 95 species are now accepted .", "95 species are now accepted ."]

	print(SARI_score(csent1, rsents, ssent))
	print(SARI_score(csent2, rsents, ssent))
	print(SARI_score(csent3, rsents, ssent))
	main()