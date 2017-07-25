from scipy.stats import bernoulli
from scipy.interpolate import spline
from scipy.stats.stats import pearsonr
import scikits.bootstrap
import poisson_binomial as pb
from multiprocessing import Pool
from nltk import word_tokenize
from nltk import pos_tag
import cmath
import math
import random
import pandas as pd
import sys
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
sys.path.append('/home/borgr/ucca/ucca/scripts')
import pickle
sys.path.append('/home/borgr/ucca/ucca/ucca')
sys.path.append('/home/borgr/ucca/ucca')
import convert
import textutil
sys.path.append('/home/borgr/ucca/ucca/scripts/distances')
import align
from correction_quality import align_sentence_words
from correction_quality import preprocess_word

#file locations
ASSESS_LEARNER_DIR = r"/home/borgr/ucca/assess_learner_language/"
GOLD_FILE = ASSESS_LEARNER_DIR + r"data/conll14st-test-data/noalt/official-2014.combined.m2"
CORRECTIONS_DIR = ASSESS_LEARNER_DIR + r"batches/"
DATA_DIR = ASSESS_LEARNER_DIR + r"calculations_data/"
PLOTS_DIR = ASSESS_LEARNER_DIR + r"plots/corrections/"
HISTS_DIR = ASSESS_LEARNER_DIR + r"unseenEst/"
TRIALS_FILE = DATA_DIR + r"trials"
BATCH_FILES = [r"Batch_2612793_batch_results.csv", r"Batch_2626033_batch_results.csv", r"Batch_2634540_batch_results.csv"]

#batch column names
LEARNER_SENTENCES_COL = "Input.sentence"
CORRECTED_SENTENCES_COL = "Answer.WritingTexts"
INDEXES_CHANGED_COL = "IndexesChanged"

# distribution columns
VARIANTS_NUM_COL = 1
PROB_COL = 0

# ONLY_DIFFERENT_CORRECTIONS = False
INPUT_HIST_IDENTIFIER = "input_hist"
OUTPUT_HIST_IDENTIFIER = "output_hist"
EXACT_COMP = "exact"
INDEX_COMP = "index"
COVERAGE_GOAL = 0.7
REPETITIONS = 1000
COMPARISON_METHODS = [EXACT_COMP, INDEX_COMP]
COVERAGE_METHODS = [lambda results: np.mean(results, axis = 1).flatten()] # all:[lambda results: np.mean(results, axis = 1).flatten(), lambda results: np.mean(results > COVERAGE_GOAL, axis = 1).flatten()]
MEAN_MEASURE = "Mean coverage"
MEASURE_NAMES = [MEAN_MEASURE] # all:["Mean coverage", "Probabillity of more than " + str(COVERAGE_GOAL) + " coverage "]
# x
CORRECTION_NUMS = list(range(51))
ALTERNATIVE_GOLD_MS = [1,3,4,6,7,9,10]

def main():
	db = read_batches()
	create_golds(db.loc[:, LEARNER_SENTENCES_COL], db.loc[:, CORRECTED_SENTENCES_COL], GOLD_FILE, ALTERNATIVE_GOLD_MS)

	db = clean_data(db)
	learner_sentences = db[LEARNER_SENTENCES_COL].unique()
	show_correction = False
	save_correction = False
	show_coverage = False
	save_coverage = False
	show_dists = False
	save_dists = False
	show_significance = True
	save_significance = True
	compare_correction_distributions(db, EXACT_COMP, show=show_correction, save=save_correction)
	db[INDEXES_CHANGED_COL] = find_changed_indexes(learner_sentences, db.loc[:, LEARNER_SENTENCES_COL], db.loc[:, CORRECTED_SENTENCES_COL])
	compare_correction_distributions(db, INDEX_COMP, index=INDEXES_CHANGED_COL, show=show_correction, save=save_correction)
	for root, dirs, files in os.walk(HISTS_DIR):
		for filename in files:
			if INPUT_HIST_IDENTIFIER in filename:
				assess_real_distributions(root+filename, str(0))
	plot_dists(show_dists, save_dists, EXACT_COMP)
	assess_coverage(True, show=show_coverage, save=save_coverage, res_type=EXACT_COMP)
	coverage_by_corrections_num = assess_coverage(False, show=show_coverage, save=save_coverage, res_type=EXACT_COMP)
	plot_significance(show=show_significance,save=save_significance)

def create_golds(sentences, corrections, gold_file, ms):
	""" writes a m2 file and a perfect output by sampling a sentence for sentences for each ungrammatical sentence in gold_file"""
	for m in ms:
		m2file, perfectOutput = choose_corrections_for_gold(gold_file, sentences, corrections, m)
		filename = str(m) + "_sgss.m2"
		with open(DATA_DIR + filename, "w") as fl:
			fl.writelines(m2file)
		filename = "perfect_output_for_" + str(m) + "_sgss.m2"
		with open(DATA_DIR + filename, "w") as fl:
			fl.writelines(perfectOutput)

def choose_corrections_for_gold(gold_file, sentences, corrections, m):
	""" creates (source_sentences, gold_edits, system_sentences) tuple that can be passed as data for m2scorer.
		The function replaces sentences that need corrections with sentences from sentences and adds m corrections
		to the gold edits and system sentences as needed. """
	# print("_____\nsentences", sentences)
	correction4gold = []
	perfectOutput = []
	with open(gold_file, "r") as fl:
		lines = fl.readlines()
	i = 0
	while i < len(lines):
		if lines[i].startswith("S"):
			if i+1 == len(lines) or not lines[i+1].startswith("A"):
				correction4gold.append(lines[i])
				perfectOutput.append(lines[i][2:])
			else:
				chosen_index = -1
				# while chosen_index not in sentences
				chosen_index = np.random.randint(0, sentences.size - 1)
				chosen_sentence = sentences.iloc[chosen_index]
				num_chosen = 0
				corresponding_corrections = corrections[sentences == chosen_sentence]
				correction4gold.append("S " + chosen_sentence + "\n")
				while num_chosen < m:
					chosen_ind = np.random.randint(0, corresponding_corrections.size)
					chosen_correction = corresponding_corrections.iloc[chosen_ind]
					addition = convert_correction_to_m2(chosen_sentence, chosen_correction, num_chosen)
					if not addition:
						addition = ["A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(num_chosen) + "\n"]
					correction4gold += addition

					num_chosen += 1
				chosen_ind = np.random.randint(0, corresponding_corrections.size)
				chosen_correction = corresponding_corrections.iloc[chosen_ind]
				if chosen_correction.count("\n") != 0:
					chosen_correction = chosen_correction.split("\n")[-1]
				perfectOutput.append(chosen_correction + "\n")
				if perfectOutput[-1].count("\n") != 1:
					print("bad sentence",perfectOutput[-1])
					raise("?")
			correction4gold.append("\n")
		i+=1
	return correction4gold, perfectOutput


def convert_correction_to_m2(source, correct, annotator_num=0):
	lines = []
	correct = re.sub(r"(\w)([,\.\(\)])",r"\1 \2", correct)
	source = re.sub(r"(\w)([,\.\(\)])",r"\1 \2", source)
	# source = re.sub(r"(\w)\.",r"\1 .", source)
	words_align, index_align = (align_sentence_words(source, correct, True))
	s = source.split()
	c = correct.split()
	i, j = 0, 0
	while i < len(s) or j < len(c):
		error = None
		if i >= len(s):
			error = "Wform"
			# print("i length", i, j, " ".join(c[j:]))
			span = (len(s),len(s))
			correction = " ".join(c[j:])
			j = len(c)
		elif j >= len(c):
			error = "Wform"
			# print("j length", i,j)
			span = (i, len(s))
			correction = ""
			i = len(s)
		elif not is_same_words(s[i], c[j]):
			error = "Wform"
			if (i, j) in index_align:
				# print("same index", i, j)
				#aligned change
				span = (i, i+1)
				correction = c[j]
				i += 1
				j += 1
			elif (i, -1) in index_align:
				# print("i unmapped", i, j)
				# source mapped to nothing
				span = (i, i + 1)
				correction = ""
				i += 1
			# elif(-1, j):
			# 	# correction mapped to nothing
			# 	span = (i,tmpi)
			# 	correction = " ".join(c[j:tmpj])
			# 	j += 1
			elif (i, j+1) in index_align:
				# print("j+1 mapped to current i", i, j)
				span = (i, i)
				correction = c[j]
				j += 1
			elif (i+1, j) in index_align:
				# print("i+1 mapped to current j", i, j)
				span = (i, i+1)
				correction = ""
				i += 1
			elif (i, j+2) in index_align:
				# print("j+2 mapped to current i", i, j)
				span = (i, i)
				correction = " ".join(c[j:j+2])
				j += 2
			elif (i+2, j) in index_align:
				# print("i+2 mapped to current j", i, j)
				span = (i, i+2)
				correction = ""
				i += 2
			elif (i, j+3) in index_align:
				# print("j+3 mapped to current i", i, j)
				span = (i, i)
				correction = " ".join(c[j:j+3])
				j += 3
			elif (i+3, j) in index_align:
				# print("i+3 mapped to current j", i, j)
				span = (i, i+3)
				correction = ""
				i += 3
			else:
				# print("special case ", i, j)
				# i,j are not aligned and not deleted but we still got here 
				# print("test that. what happens when s longer? when c longer?")
				tmpi = i + 1
				tmpj = j + 1
				while ((tmpi < len(s) and tmpj < len(c)) and
					(not is_same_words(s[tmpi], c[tmpj])) and 
					((tmpi, -1) in index_align or (-1, tmpj) in index_align)):
					if (tmpi, -1) in index_align:
						tmpi += 1
					if (-1, tmpj) in index_align:
						tmpj += 1
				if (tmpi, tmpj) in index_align:
					span = (i, tmpi)
					correction = " ".join(c[j:tmpj])
					i = tmpi
					j = tmpj
				else:
					span = (i, i+1)
					correction = c[j]
					i += 1
					j += 1
		else:
			i += 1
			j += 1
		if error:
			# print("span:", span)
			# print("correction", correction)
			lines.append("A " + str(span[0]) + " " + str(span[1]) + "|||" + error + "|||" + correction+"|||REQUIRED|||-NONE-|||"+str(annotator_num) + "\n")
	# last_source_index = 0
	# for (w1, w2), (i, j) in zip(words_align, index_align):
	# 	if not is_same_words(w1, w2):
		# 	if i == -1:
		# 		span = (last_source_index,last_source_index)
		# 	else:
		# 		span = (i, i+1)
		# 		last_source_index = i
		# 	error = "Wform"
		# 	correction = w2
		# 	lines.append("A " + str(span[0]) + " " + str(span[1]) + "|||" + error + "|||" + correction+"|||REQUIRED|||-NONE-|||"+str(annotator_num))
		# else:
		# 	last_source_index = i
	return lines


def assess_coverage(only_different_samples, show=True, save=True, res_type=EXACT_COMP, res_measure=MEAN_MEASURE, accuracyax=None):
	""" runs all computations relevant to coverage assessments (calculations and plotting)""" 
	trial_num = get_trial_num()
	repeat = "" if only_different_samples else "_repeat"
	repeat += "_" + str(REPETITIONS)
	data_filename = DATA_DIR + str(trial_num) + "coverage_data" + repeat
	# gather coverage results
	if os.path.isfile(data_filename):
		with open(data_filename, "rb") as fl:
			all_ys = pickle.load(fl)
	else:
		all_ys = [[] for i in range(len(COMPARISON_METHODS))]
		for root, dirs, files in os.walk(HISTS_DIR):
			for filename in files:
				if OUTPUT_HIST_IDENTIFIER in filename:
					dist = read_dist_from_file(root+filename)
					i = 0
					while i < len(dist[0]):
						if dist[VARIANTS_NUM_COL][i] > 1:
							dist = np.append(dist,np.array([[dist[PROB_COL][i], dist[VARIANTS_NUM_COL][i] - 1]]).transpose(), axis=1)
							dist[VARIANTS_NUM_COL][i] = 1
						i += 1
					results = []
					for correction_num in CORRECTION_NUMS:
						results.append(compute_probability_to_account_async(dist, correction_num, REPETITIONS, only_different_samples))
					results = np.array(results)
					ys = []
					for coverage_method in COVERAGE_METHODS:
						ys.append(coverage_method(results))
					ys = np.array(ys)
					# save y
					for i, comparison_method in enumerate(COMPARISON_METHODS):
						if comparison_method in filename:
							all_ys[i].append(ys)
		print("should write to ")
		with open(data_filename, "wb+") as fl:
			print(data_filename)
			pickle.dump(all_ys, fl)
	# plot results
	#list by: comparison method->distribution->measure->correction num(Y)
	if show or save:
		xlabel = "amount of different corrections" if only_different_samples else "amount of corrections sampled"
		all_ys = np.array(all_ys)
		axis_num = len(all_ys[0][0])
		for comparison_method_key, dist in enumerate(all_ys):
			axes = [plt.subplot("1"+str(axis_num)+str(i+1)) for i in range(axis_num)]
			if save:
				fig_prefix = COMPARISON_METHODS[comparison_method_key] +"_" + repeat
				save = PLOTS_DIR + fig_prefix + r"_coverage" + ".png"
			title_addition = "using " + COMPARISON_METHODS[comparison_method_key] + " comparison"
			plot_coverage_for_each_sentence(dist, axes, title_addition, show, save, xlabel)

			if save:
				fig_prefix = COMPARISON_METHODS[comparison_method_key] +"_" + repeat
				save = PLOTS_DIR + fig_prefix + r"_accuracy" + ".png"

			plot_expected_best_coverage(dist, plt.subplot("111"), title_addition, show, save, xlabel)
			plt.cla()

			# if save:
			# 	fig_prefix = "accCI_" + COMPARISON_METHODS[comparison_method_key] +"_" + repeat
			# 	save = PLOTS_DIR + fig_prefix + r"_accuracy" + ".png"
			# plot_expected_best_coverage(dist, plt.subplot("111"), title_addition, show, save, xlabel, False)
			# plt.cla()

			if save:
				fig_prefix = COMPARISON_METHODS[comparison_method_key] +"_" + repeat
				save = PLOTS_DIR + fig_prefix + r"_covered_corrections_dist" + ".png"
			plot_covered_corrections_distribution([correction for correction in CORRECTION_NUMS if correction > 0], dist, plt.subplot("111"), title_addition, show, save, xlabel)
		if save:
			fig_prefix = repeat[1:]
			save = PLOTS_DIR + "noSig_" + fig_prefix + r"_accuracy" + ".png"
		plot_expected_best_coverages(all_ys, plt.subplot("111"), title_addition, show, save, xlabel, True, False)
		# if save:
		# 	fig_prefix = repeat[1:]
		# 	save = PLOTS_DIR + "accCI_" +fig_prefix + r"_accuracy" + ".png"
		# plot_expected_best_coverages(all_ys, plt.subplot("111"), title_addition, show, save, xlabel, False)
	# extract value for return
	res = []
	for comparison_method_key, dist in enumerate(all_ys):
		for sent_key, ys in enumerate(dist):
			for i, y in enumerate(ys):
				if COMPARISON_METHODS[comparison_method_key] == res_type and MEASURE_NAMES[i] == res_measure:
					res.append(y)
	return np.array(res)


def get_probability_with_belief(ps, n, belief=1, approximate=False, pdf=False, all=False):
	""" given probabilities of rightly identifying a good correction as such for each sentence,
		and a belief of the probability for an output to be correct, computes the probability 
		to find that n sentences are correct  """	
	new_ps = ps * belief
	if all:
		if approximate:
			return np.array([get_probability_with_belief(new_ps, n, 1, approximate, pdf, all) for n in range(len(ps)+1)])
		
		else:
			if pdf:
				return pb.poisson_binomial_PMFS_DFT(new_ps)
			else:
				return np.cumsum(pb.poisson_binomial_PMFS_DFT(new_ps))
	else:
		if pdf:
			if approximate:
				return pb.poisson_binomial_PMF_possion_approximation(new_ps, n)
			else:
				return pb.poisson_binomial_PMF_DFT(new_ps, n)
			# return pb.poisson_binomial_PMF(new_ps, n)
		else:
			if approximate:
				return pb.poisson_binomial_CDF_refined_normal_approximation(new_ps, n)
			else:
				return pb.poisson_binomial_CDF_DFT(new_ps, n)

def mass_for_poisson_binomial_probability_range(ps, range, belief=1, approximate=False):
	"""returns the sum of probability mass in the range"""
	return sum((get_probability_with_belief(ps, ind, approximate) for ind in range))


def ranges_for_poisson_binomial_probability_mass(ps, mass, belief=1, approximate=False):
	""" returns the minimum interval around the mean that coveres the given mass,
		half from each side of the mean (may cover more than the mass)
		"""
	right_mass = mass / 2
	left_mass = mass / 2
	mu = sum(ps)
	n = len(ps)
	right_ind = math.ceil(mu)
	left_ind = int(mu)
	right_covered = 0
	left_covered = 0
	if right_ind == left_ind:
		left -= 1
	while right_ind <= n and right_covered < right_mass:
		right_covered += get_probability_with_belief(ps, right_ind, belief, approximate)
		right_ind += 1
	right_ind -=1
	while left_ind >= 0 and left_covered < left_mass:
		left_covered += get_probability_with_belief(ps, left_ind, belief, approximate)
		left_ind -= 1
	left_ind += 1
	return left_ind, right_ind


def find_changed_indexes(unique, sentences, corrections):
	corrections = corrections.copy()
	for sentence in unique:
		converter = lambda x: convert_sentence_to_diff_indexes(sentence, x)
		indexing = sentences == sentence
		corrections[indexing] = corrections[indexing].apply(converter)
	return corrections

# remove_POS(sentence, ["IN", "TO"]) # remove prepositions
def remove_POS(sent, pos_tags):
	tags = pos_tag(word_tokenize(sent))
	sent = " ".join([word for word,tag in tags if tag not in pos_tags])
	return sent


def read_dist_from_file(filename):
	return np.transpose(np.loadtxt(filename, skiprows=1))


def assess_real_distributions(filename, minFrequency=0):
	outfile = filename.replace(INPUT_HIST_IDENTIFIER, OUTPUT_HIST_IDENTIFIER)
	if os.path.isfile(outfile):
		print(outfile, "already exists")
	else:
		os.system("python /home/borgr/unseenest/src/unseen_est.py " + filename + " 0 " + outfile + " -s 1 ")

def compute_coverage(cdf, p, distribution, samples, only_different_samples):
	covered = 0
	uncovered_variants_num = distribution[VARIANTS_NUM_COL].copy()
	uncovered_probes = distribution[PROB_COL].copy()
	sampled = 0
	while sampled < samples and not cmath.isclose(covered, 1.0):
		chosen = np.random.choice(np.arange(len(distribution[PROB_COL])), p=p)
		covered += min(uncovered_variants_num[chosen], samples - sampled) * distribution[PROB_COL][chosen]
		if only_different_samples:
			p = uncovered_probes/sum(uncovered_probes)
			assert(not np.isnan(sum(p)))
			uncovered_probes[chosen] = 0
			sampled += uncovered_variants_num[chosen]
		else:
			sampled += distribution[VARIANTS_NUM_COL][chosen]
		uncovered_variants_num[chosen] = 0
	return covered

def __compute_coverage(tpl):
	cdf, p, distribution, samples, only_different_samples = tpl
	return compute_coverage(cdf, p, distribution, samples, only_different_samples)

def compute_probability_to_account_async(distribution, samples, repetitions, only_different_samples):
	pool = Pool(4)
	cdf = np.cumsum(distribution[PROB_COL])
	probs = distribution[PROB_COL]/sum(distribution[PROB_COL])
	# it = pool.imap(lambda x: compute_coverage(cdf, probs, distribution, samples, only_different_samples), list(range(repetitions)))
	it = []
	it += list(pool.imap(__compute_coverage, [(cdf, probs, distribution, samples, only_different_samples)]*repetitions))
	pool.close()
	pool.join()
	return np.array(it)

def compute_probability_to_account(distribution, samples, repetitions, only_different_samples):
	covered = np.zeros([repetitions,1])
	cdf = np.cumsum(distribution[PROB_COL])

	probs = distribution[PROB_COL]/sum(distribution[PROB_COL])

	for i in range(repetitions):
		covered[i] = compute_coverage(cdf, probs, distribution, samples, only_different_samples)
	return covered

def compare_correction_distributions(db, name, index=CORRECTED_SENTENCES_COL, show=True, save=True):
	learner_sentences = db[LEARNER_SENTENCES_COL].unique()
	colors = many_colors(range(len(learner_sentences)))
	concat = []
	amount_of_models = []
	for color, sentence in zip(colors.values(),learner_sentences):
		corrected = db[index][db[LEARNER_SENTENCES_COL] == sentence]
		counts = corrected.value_counts()
		concat.append(counts)
		amount_of_models.append(len(counts))

	reverseXY = True
	fig_prefix = str(len(learner_sentences))+"_"+str(len(corrected))+"_"+name
	if show or save:
	# 	ax = plt.subplot("111")
	# 	plot_hist(learner_sentences, ax, concat, name)
	# 	plt.show()

		plt.cla()
		ax = plt.subplot("111")
		plot_differences_hist(learner_sentences, ax, concat, name)
		print("amount of models ", amount_of_models)
		if save:
			plt.savefig(PLOTS_DIR + fig_prefix + r"_hist" + ".png", bbox_inches='tight')
		if show:
			plt.show()

		plt.cla()
		ax = plt.subplot("111")
		plot_acounts_for_percentage(learner_sentences, ax, concat, name)
		if save:
			plt.savefig(PLOTS_DIR + fig_prefix + r"percentage_hist" + ".png", bbox_inches='tight')
		if show:
			plt.show()

		# ax = plt.subplot("111")
		# reverseXY = True
		# prefix_reverseXY = "_rev_" if reverseXY else ""
		# plot_acounts_for_percentage(learner_sentences, ax, concat, name, reverseXY=reverseXY)
		# if save:
		# 	plt.savefig(PLOTS_DIR + fig_prefix + prefix_reverseXY + r"percentage_hist" + ".png", bbox_inches='tight')
		# if show:
		# 	plt.show()
	export_hists(learner_sentences, concat, name, HISTS_DIR)

###################################################################################
####								plots
###################################################################################
def export_hists(l, data, comparison_by, HISTS_DIR):
	for i, name in enumerate(l):
		filename = re.sub("\W","",name)[:6]
		filename = HISTS_DIR + filename + "_"+ comparison_by + "_" + INPUT_HIST_IDENTIFIER +".txt"
		y = create_hist(data[i], bottom=1)
		y = [str(val)+"\n" for val in y]
		with open(filename, "w+") as fl:
			fl.writelines(y)

def expected_accuracy(ps):
	return pb.mu(ps)/len(ps)


def plot_covered_corrections_distribution(corrections_to_plot, dist, ax, title_addition="", show=True, save_name=None, xlabel=None):
	#corrections_num -> coverage 
	coverage_by_corrections_num = []
	for sent_key, ys in enumerate(dist):
		for i, y in enumerate(ys):
			if MEASURE_NAMES[i] == MEAN_MEASURE:
				coverage_by_corrections_num.append(y)

	ys = []
	for correction_index, correction_num in enumerate(CORRECTION_NUMS):
		if correction_num in corrections_to_plot:
			ps = np.fromiter((coverage[correction_index] for coverage in coverage_by_corrections_num), np.float)
			ys.append(get_probability_with_belief(ps, 1, pdf=True, all=True))

	x = np.arange(len(ps) + 1)
	colors = many_colors(range(len(corrections_to_plot)))
	for y, color, correction_num in zip(ys,colors.values(), corrections_to_plot):
		ax.plot(x, y, color=color, label=correction_num)

	ax.set_ylabel("probabillity")
	ax.set_xlabel("number of covered sentences")
	# ax.set_title("probabillity distribution for correct sentences covered in g.s.\n" + "out of " + str(len(x)-1) + " " + title_addition)
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True, title="corrections in g.s.")
	if save_name:
		plt.savefig(save_name, bbox_inches='tight')
	if show:
		plt.show()
	plt.cla()


def bern(p):
	if p>=1:
		return 1
	elif p<=0:
		return 0
	else:
		return bernoulli.rvs(p)


def accuracy(ps):
	try:
		res = np.mean([bern(p) for p in ps])
		return res
	except Exception as e:
		print([p for p in ps])
		return np.mean([bern(p) for p in ps])


def plot_expected_best_coverages(dists, ax, title_addition="", show=True, save_name=None, xlabel=None, sig_of_mean=True, plot_sig=True):
	""" plots a line for each sentence
		axes - a subscriptable object of axis to plot for each comparison meathod
		dist - list of lists of Ys : distribution->measure->correction num(Y)
		"""
	width = 0.1
	for comparison_method_key, dist in enumerate(dists[::-1]):
		#corrections_num -> coverage
		coverage_by_corrections_num = []
		for sent_key, ys in enumerate(dist):
			for i, y in enumerate(ys):
				if MEASURE_NAMES[i] == MEAN_MEASURE:
					coverage_by_corrections_num.append(y)

		if plot_sig:
			top = []
			bottom = []
			cis = [bottom, top]
		x = []
		y = []
		for correction_index, correction_num in enumerate(CORRECTION_NUMS):
			ps = np.fromiter((coverage[correction_index] for coverage in coverage_by_corrections_num), np.float)
			x.append(correction_num)
			y.append(expected_accuracy(ps))
			if np.all(ps == ps*0) or np.all(ps - 1 == ps*0):
				ci = [0,0]
			else:
				if sig_of_mean:
					func = expected_accuracy
				else:
					func = accuracy
				if plot_sig:
					ci = scikits.bootstrap.ci(ps, func)
					ci = [[y[-1] - float(ci[0])],[float(ci[1])] - y[-1]]
			if plot_sig:
				top.append(ci[1])
				bottom.append(ci[0])
		x = np.array(x)
		label = COMPARISON_METHODS[::-1][comparison_method_key]
		x = x + width*comparison_method_key
		if not plot_sig:
			if label == INDEX_COMP:
				print(save_name, "index_accuracy", y[:12])
				ax.plot(x, y, label=label)
			else:
				print(save_name, "accuracy", y[:22])
				ax.plot(x, y, "--",label=label)
		else:
			if label == INDEX_COMP:
				ax.errorbar(x, y, yerr=cis, label=label)
			else:
				ax.errorbar(x, y, yerr=cis, label=label, fmt="--")
		# ax.plot(np.array(CORRECTION_NUMS) + width*comparison_method_key, y, label=COMPARISON_METHODS[comparison_method_key])
	ax.set_ylabel("expected accuracy")
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True)
	if xlabel:
		ax.set_xlabel(xlabel)
	# ax.set_title("Expected accuracy for perfect corrected text by corrections number\n" + title_addition)
	if save_name:
		plt.savefig(save_name, bbox_inches='tight')
	if show:
		plt.show()
	plt.cla()


def plot_expected_best_coverage(dist, ax, title_addition="", show=True, save_name=None, xlabel=None, sig_of_mean=True):
	""" plots a line for each sentence
		axes - a subscriptable object of axis to plot for each comparison meathod
		dist - list of lists of Ys : distribution->measure->correction num(Y)
		"""
	#corrections_num -> coverage 
	coverage_by_corrections_num = []
	for sent_key, ys in enumerate(dist):
		for i, y in enumerate(ys):
			if MEASURE_NAMES[i] == MEAN_MEASURE:
				coverage_by_corrections_num.append(y)

	top = []
	bottom = []
	cis = [bottom, top]
	x = []
	y = []
	for correction_index, correction_num in enumerate(CORRECTION_NUMS):
		ps = np.fromiter((coverage[correction_index] for coverage in coverage_by_corrections_num), np.float)
		x.append(correction_num)
		y.append(expected_accuracy(ps))
		if np.all(ps == ps*0) or np.all(ps - 1 == ps*0):
			ci = [0,0]
		else:
			if sig_of_mean:
				func = expected_accuracy
			else:
				func = accuracy
			ci = scikits.bootstrap.ci(ps, func)
			ci = [[y[-1] - float(ci[0])],[float(ci[1])] - y[-1]]
		top.append(ci[1])
		bottom.append(ci[0])
	ax.errorbar(x, y, yerr=cis)
	ax.plot(CORRECTION_NUMS, y)
	if sig_of_mean:
		ax.set_ylabel("expected accuracy")
	else:
		ax.set_ylabel("accuracy")
	if xlabel:
		ax.set_xlabel(xlabel)
	# ax.set_title("Expected accuracy for perfect corrected text by corrections number\n" + title_addition)
	if save_name:
		plt.savefig(save_name, bbox_inches='tight')
	if show:
		plt.show()


def plot_significance(show=True, save=True):
	learner_file = "source"
	JMGR_file = "JMGR"
	ACL2016RozovskayaRothOutput_file = "conll14st.output.1cleaned"
	char_based_file = "filtered_test.txt"
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
	gold_file = "gold"
	files = [ACL2016RozovskayaRothOutput_file,
	JMGR,
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
	results = parse_sigfiles(files)
	print("sig results")
	for i, file in enumerate(files):
		print (file, results[i])
	names = [filename if filename != ACL2016RozovskayaRothOutput_file else "RoRo" for filename in files]
	names = [name if name != char_based_file else "char" for name in names]
	results = [[[1,0,0],[1,0,0]]] + list(results) + [[[1,1,1],[1,1,1]]]
	names = [learner_file] + names + [gold_file]
	#no gold standard
	results = results[:-1]
	names = names[:-1]
	plot_sig_bars(results, names, show, save)
	files = ["perfect_output_for_" + str(m+1) + "_sgss.m2" for m in np.arange(10)]
	results = parse_sigfiles(files)
	print("sig results")
	for i, file in enumerate(files):
		print (file, results[i])
	names = [str(m+1) for m in np.arange(10)]
	plot_sig(results, names, show, save)


def plot_sig(significances, names, show, save):
	names = np.array([0]+names)
	for measure_idx, measure in enumerate(["precision", "recall", "$F_{0.5}$"]):
		xs = [0]
		ys = [0]
		cis = [0]
		for x, significance in enumerate(significances):
			sig = [significance[0][measure_idx], significance[1][measure_idx]]
			y = np.mean(sig)
			xs.append(x + 1)
			ys.append(y)
			cis.append(y-sig[0])
		xs = np.array(xs)
		ys = np.array(ys)
		cis = np.array(cis)
		sort_idx = xs.argsort()
		labels = names[sort_idx]
		ys = ys[sort_idx]
		cis = cis[sort_idx]
		colors = many_colors(xs, cm.copper)
		colors = [colors[i] for i in xs]
		plt.errorbar(xs, ys, yerr=cis)
		plt.plot(xs, ys)
		plt.xticks(xs, labels)
		plt.ylabel(measure)
		plt.xlabel("$M$ - Number of references in gold standard")
		if save:
			plt.savefig(PLOTS_DIR + measure + "_Ms_significance" + ".png", bbox_inches='tight')
		if show:
			plt.show()
		plt.cla()


def plot_sig_bars(significances, names, show, save):
	names = np.array(names)
	for measure_idx, measure in enumerate(["precision", "recall", "$F_{0.5}$"]):
		xs = []
		ys = []
		cis = []
		for x, significance in enumerate(significances):
			sig = [significance[0][measure_idx], significance[1][measure_idx]]
			y = np.mean(sig)
			xs.append(x)
			ys.append(y)
			cis.append(y-sig[0])
		xs = np.array(xs)
		ys = np.array(ys)
		cis = np.array(cis)
		sort_idx = ys.argsort()
		labels = names[sort_idx]
		ys = ys[sort_idx]
		cis = cis[sort_idx]
		colors = many_colors(xs, cm.copper)
		colors = [colors[i] for i in xs]
		plt.bar(xs, ys, yerr=cis, align='center', label=labels, edgecolor=colors, color=colors)
		plt.xticks(xs, labels, rotation=70)
		plt.ylabel(measure)
		if save:
			plt.savefig(PLOTS_DIR + measure + "_significance" + ".png", bbox_inches='tight')
		if show:
			plt.show()
		plt.cla()


def parse_sigfiles(files):
	results = []
	for file in files:
		filename = r"/home/borgr/ucca/assess_learner_language/results/significance/1000_" + file
		with open(filename, "r") as fl:
			str_content = fl.readlines()
		content = []
		for i in range(len(str_content)):
			content.append([float(x) for x in re.split('[\[\] \t\n]+', str_content[i]) if x])
		results.append(content)
	return results


def plot_dists(show=True, save=True, dists_type=EXACT_COMP):
	if show or save:
		dists = []
		one_dist = [[],[]]
		for root, dirs, files in os.walk(HISTS_DIR):
			for filename in files:
				if OUTPUT_HIST_IDENTIFIER in filename and dists_type in filename:
					dist = read_dist_from_file(root+filename)
					dist = dist[:,dist[1] > 0]
					if isinstance(dist, np.ndarray):
						dists.append(dist)
						one_dist[0] += list(dists[-1][0])
						one_dist[1] += list(dists[-1][1])
		max_lines = 300
		# max_lines = len(dists)
		chosen_lines = set(np.random.randint(0, len(dists) - 1, max_lines))
		# print(chosen_lines)
		colors = many_colors(range(max_lines))
		pearsons = []
		# dist_key = 0
		# dist = np.array(one_dist)
		thresholds = [0, 0.001, 0.01, 0.1]
		for threshold in thresholds:
			variants_num = []
			masses = []
			for dist_key, dist in enumerate(dists):
				dist = dist[:,dist[0,:].argsort()[::-1]]
				frequents = dist[:,dist[0] > threshold]
				mass = frequents[0].dot(frequents[1])
				variants_num.append(sum(frequents[1]))
				masses.append(mass)
				# print("dist",dist)
				# print("frequents",frequents)
				# print(mass)
				# print(np.mean(frequents[0]))
				# print(variants_num)
				# raise
			print("mean number of variants with more than",threshold,"frequency:",np.mean(variants_num))
			print("mean mass of those variants",np.mean(masses))
		for dist_key, dist in enumerate(dists):
			dist = dist[:,dist[0,:].argsort()[::-1]]

			x = np.log(dist[1].cumsum())
			y = np.log(dist[0])
			# print(dist[1])
			# print(dist[0])
			if len(x) > 1:
				pearsons.append(pearsonr(x,y)[0])
			if dist_key in chosen_lines:
				x_new = x
				# x_new = np.linspace(x.min(),x.max(),300)
				# y = spline(x, np.log(y), x)
				# print("plot", dist_key)
				plt.plot(x_new, y)
				# plt.scatter(x_new, y, color="b")
				plt.ylabel("log frequency")
				plt.xlabel("log rank")
		plt.xlim(xmin=0)
		print("all pearson correlations:", pearsons)
		print("mean pearson correlation:", np.mean(pearsons))
		if save:
			plt.savefig(PLOTS_DIR + dists_type + "_dists_plot"+".png", bbox_inches='tight')
		if show:
			plt.show()
		plt.cla()


def plot_coverage_for_each_sentence(dist, axes, title_addition="", show=True, save_name=None, xlabel=None):
	""" plots expected accuracy result for each correction number
		axes - a subscriptable object of axis to plot for each comparison method
		dist - list of lists of Ys : distribution->measure->correction num(Y)
		"""
	for sent_key, ys in enumerate(dist):
		colors = many_colors(range(len(dist)))
		for i, y in enumerate(ys):
			ax = axes[i]
			ax.plot(CORRECTION_NUMS, y, color=colors[sent_key])
			ax.set_ylabel(MEASURE_NAMES[i])
			if xlabel:
				ax.set_xlabel(xlabel)
			# ax.set_title(MEASURE_NAMES[i] + " of different amount of corrections\n" + title_addition)
	if save_name:
		plt.savefig(save_name, bbox_inches='tight')
	if show:
		plt.show()
	plt.cla()


def plot_hist(l, ax, data, comparison_by, bottom=1):
	width = 1.0/len(l)
	for i, name in enumerate(l):
		y = create_hist(data[i], bottom=bottom)
		x = np.array(range(len(y)))
		print(comparison_by, " hist results \n", name,"\n",y)
		colors = many_colors(range(len(l)))
		ax.plot(x, y, color=colors[i], label=name)
	plt.ylim(ymin=0)
	plt.autoscale(enable=True, axis='x', tight=False)
	plt.ylabel("amount of corrections")
	plt.xlabel("number of times seen")
	# plt.title("hist of the number repetitions a correction was seen, using " + comparison_by + " comparison")
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True)
	# plt.tight_layout()


def plot_acounts_for_percentage(l, ax, data, comparison_by, bottom=1, reverseXY=False):
	width = 1.0/len(l)*0
	maxY=0
	for i, name in enumerate(l):
		y = create_hist(data[i], bottom=bottom)
		x = np.array(range(len(y)))
		Ysum = sum(data[i])
		maxY = max(maxY, np.max(y))
		x = x/Ysum*100
		colors = many_colors(range(len(l)))
		x = list(x)
		y = list(y)
		x = np.array([x_val for x_val, y_val in zip(x,y) if y_val])
		y = np.array([y_val for y_val in y if y_val])
		if reverseXY:
			x,y = y,x
		print("percentages", x)
		print("total number of corrections", Ysum)
		print(comparison_by, " hist results \n", name, "\n", y)
		ax.scatter(x + i*width, y, color=colors[i], label=name)
		plt.ylim(ymin=0, ymax=maxY)
		plt.xlim(xmin=0)
		# ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=name, edgecolor=colors[i])
	plt.autoscale(enable=True, axis='x', tight=False)
	xlabel = "percentage of probability each correction accounts for"
	ylabel = "amount of corrections"
	if reverseXY:
		xlabel, ylabel = ylabel, xlabel
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	# plt.title("hist of the number repetitions a correction was seen, using " + comparison_by + " comparison")
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True)
	# plt.tight_layout()


def plot_differences_hist(l, ax, data, comparison_by, bottom=1, percentage=False):
	width = 1.0/len(l)
	for i, name in enumerate(l):
		y = create_hist(data[i], bottom=bottom)
		x = np.array(range(len(y)))
		if percentage:
			Ysum = sum(data[i])
			x = [val/Ysum for val in x]
			print("total number of corrections",Ysum)
		print(comparison_by," hist results \n", name,"\n",y)
		colors = many_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=name, edgecolor=colors[i])
	plt.autoscale(enable=True, axis='x', tight=False)
	if percentage:
		plt.ylabel("percentage from total corrections")
	else:
		plt.ylabel("amount of corrections")
	plt.ylim(ymin=0)
	plt.xlabel("number of times seen")
	# plt.title("hist of the number repetitions a correction was seen, using " + comparison_by + " comparison")
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True)
	# plt.tight_layout()


def create_hist(l, top=float("inf"), bottom=0):
	""" converts a int counter to a sorted list for a histogram"""
	count = Counter(l)
	hist = [0] * (max(count.keys()) - bottom + 1)
	for key, val in count.items():
		if key <= top and key >= bottom:
			hist[key - bottom] = val
	return hist if hist else [0]


def many_colors(labels, colors=cm.rainbow):
	"""creates colors, each corresponding to a unique label"""
	cls = set(labels)
	if len(cls) == 2:
		return dict(zip(cls, ("blue", "orange")))
	return dict(zip(cls, colors(np.linspace(0, 1, len(cls)))))

###################################################################################
####							general\NLP	
###################################################################################
def read_batches():
	frames = []
	for batch_file in BATCH_FILES:
		frames.append(pd.read_csv(CORRECTIONS_DIR + batch_file))
	return pd.concat(frames)


def get_all_sentences_corrected():
	""" returns an iterable containing all the sentences that were corrected"""
	corrected = set()
	for root, dirs, files in os.walk(CORRECTIONS_DIR):
		for file in files:
			if isBatchFile(file):
				db = pd.read_csv(root+file)
				corrected.update(set(iter(db[LEARNER_SENTENCES_COL])))
	return corrected


def clean_data(db, 	max_no_correction_needed=8):
	# clean rejections
	db = db[db.AssignmentStatus != "Rejected"]
	db.loc[:,CORRECTED_SENTENCES_COL] = db[CORRECTED_SENTENCES_COL].apply(normalize_sentence)
	db.loc[:,LEARNER_SENTENCES_COL] = db[LEARNER_SENTENCES_COL].apply(normalize_sentence)
	# ignore sentences that many annotators say no corrections are needed for them
	for sentence in db[LEARNER_SENTENCES_COL].unique():
		if (len(db[(db[LEARNER_SENTENCES_COL] == db[CORRECTED_SENTENCES_COL]) &
			     (db[LEARNER_SENTENCES_COL] == sentence)])
			     >= max_no_correction_needed):
			db = db[db[LEARNER_SENTENCES_COL] != sentence]
	return db


def get_trial_num(create_if_needed=True):
	""" gets a uniqe trial number that changes with every change of COMPARISON_METHODS, MEASURE_NAMES, REPETITIONS, CORRECTION_NUMS""" 
	trial_indicators = (COMPARISON_METHODS, MEASURE_NAMES, REPETITIONS, CORRECTION_NUMS)
	trials = []
	filename = TRIALS_FILE
	if os.path.isfile(filename):
		with open(filename, "rb") as fl:
			trials = pickle.load(fl)
			if trial_indicators in trials:
				return trials.index(trial_indicators)
	else:
		print("trials file not found, creating a new one:" + filename)
	# if this is a new trial
	if create_if_needed:
		trials.append(trial_indicators)
		with open(filename, "wb+") as fl:
			pickle.dump(trials, fl)
			return trials.index(trial_indicators)
	else:
		return -1


def isBatchFile(filename):
	return "batch" in filename and filename.split(".")[-1].lower() == "csv"


def normalize_sentence(s):
	s = re.sub(r"\W+", r" ", s)
	s = re.sub(r"(\s[a-zA-Z])\s([a-zA-Z]\s)", r"\1\2", s)
	s = s.strip()
	return s


def is_same_words(w1, w2):
	return preprocess_word(w1) == preprocess_word(w2)


def convert_sentence_to_diff_indexes(original, sentence):
	""" aligns the sentences and returns an ordered tuple of the places
		where the original was changed,
		new words are considered as changed at the end of the sentence."""
	indexes = []
	words_align, index_align = (align_sentence_words(original, sentence, True))
	max_ind = max(index_align, key=lambda x:x[0])[0]
	for (w1, w2), (i, j) in zip(words_align, index_align):
		if not is_same_words(w1,w2):
			if i == -1:
				indexes.append(max_ind)
				max_ind += 1
			else:
				indexes.append(i)
	return tuple(sorted(indexes))


if __name__ == '__main__':
	main()