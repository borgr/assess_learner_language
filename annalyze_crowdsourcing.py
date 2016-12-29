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
corrections_dir = r"/home/borgr/ucca/assess_learner_language/batches/"
TRIALS_FILE = "trials"
DATA_DIR = r"/home/borgr/ucca/assess_learner_language/calculations_data/"
PLOTS_DIR = r"/home/borgr/ucca/assess_learner_language/plots/corrections/"
HISTS_DIR = r"/home/borgr/ucca/assess_learner_language/unseenEst/"
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
CORRECTION_NUMS = list(range(21))


def main():
	frames = []
	for batch_file in BATCH_FILES:
		frames.append(pd.read_csv(corrections_dir + batch_file))
	db = pd.concat(frames)
	db = clean_data(db)
	learner_sentences = db[LEARNER_SENTENCES_COL].unique()
	show_correction = False
	save_correction = False
	show_coverage = False
	save_coverage = True


	compare_correction_distributions(db, EXACT_COMP, show=show_correction, save=save_correction)

	db[INDEXES_CHANGED_COL] = find_changed_indexes(learner_sentences, db[LEARNER_SENTENCES_COL], db[CORRECTED_SENTENCES_COL])
	compare_correction_distributions(db, INDEX_COMP, index=INDEXES_CHANGED_COL, show=show_correction, save=save_correction)
	for root, dirs, files in os.walk(HISTS_DIR):
		for filename in files:
			if INPUT_HIST_IDENTIFIER in filename:
				assess_real_distributions(root+filename, str(0))

	assess_coverage(True, show=show_coverage, save=save_coverage, res_type=EXACT_COMP)
	coverage_by_corrections_num = assess_coverage(False, show=show_coverage, save=save_coverage, res_type=EXACT_COMP)

def clean_data(db):
	# clean rejections
	db = db[db.AssignmentStatus != "Rejected"]
	db.loc[:,CORRECTED_SENTENCES_COL] = db[CORRECTED_SENTENCES_COL].apply(normalize_sentence)
	db.loc[:,LEARNER_SENTENCES_COL] = db[LEARNER_SENTENCES_COL].apply(normalize_sentence)
	max_no_correction_needed = 8
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
	filename = DATA_DIR + TRIALS_FILE
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


def assess_coverage(only_different_samples, show=True, save=True, res_type=EXACT_COMP, res_measure=MEAN_MEASURE):
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
				save = PLOTS_DIR + fig_prefix + r"_coverage" + ".svg"
			title_addition = "using " + COMPARISON_METHODS[comparison_method_key] + " comparison"
			plot_coverage_for_each_sentence(dist, axes, title_addition, show, save, xlabel)

			if save:
				fig_prefix = COMPARISON_METHODS[comparison_method_key] +"_" + repeat
				save = PLOTS_DIR + fig_prefix + r"_accuracy" + ".svg"

			plot_expected_best_coverage(dist, plt.subplot("111"), title_addition, show, save, xlabel)

			if save:
				fig_prefix = COMPARISON_METHODS[comparison_method_key] +"_" + repeat
				save = PLOTS_DIR + fig_prefix + r"_covered_corrections_dist" + ".svg"
			plot_covered_corrections_distribution([correction for correction in CORRECTION_NUMS if correction > 0], dist, plt.subplot("111"), title_addition, show, save, xlabel)
	# extract value for return
	res = []
	for comparison_method_key, dist in enumerate(all_ys):
		for sent_key, ys in enumerate(dist):
			for i, y in enumerate(ys):
				if COMPARISON_METHODS[comparison_method_key] == res_type and MEASURE_NAMES[i] == res_measure:
					res.append(y)
	return np.array(res)

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
	colors = rainbow_colors(range(len(corrections_to_plot)))
	for y, color, correction_num in zip(ys,colors.values(), corrections_to_plot):
		ax.plot(x, y, color=color, label=correction_num)

	ax.set_ylabel("probabillity")
	ax.set_xlabel("number of covered sentences")
	ax.set_title("probabillity distribution for correct sentences covered in g.s.\n" + "out of " + str(len(x)-1) + " " + title_addition)
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True, title="corrections in g.s.")
	if save_name:
		plt.savefig(save_name)
	if show:
		plt.show()
	plt.cla()
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
	pool = Pool(10)
	cdf = np.cumsum(distribution[PROB_COL])
	probs = distribution[PROB_COL]/sum(distribution[PROB_COL])
	# it = pool.imap(lambda x: compute_coverage(cdf, probs, distribution, samples, only_different_samples), list(range(repetitions)))
	it = []
	it += list(pool.imap(__compute_coverage, [(cdf, probs, distribution, samples, only_different_samples)]*repetitions))
	# print(np.array(it))
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
	colors = rainbow_colors(range(len(learner_sentences)))
	plot_num = 200+(len(learner_sentences)+1)/2*10
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
			plt.savefig(PLOTS_DIR + fig_prefix + r"_hist" + ".svg")
		if show:
			plt.show()

		plt.cla()
		ax = plt.subplot("111")
		plot_acounts_for_percentage(learner_sentences, ax, concat, name)
		if save:
			plt.savefig(PLOTS_DIR + fig_prefix + r"percentage_hist" + ".svg")
		if show:
			plt.show()

		# ax = plt.subplot("111")
		# reverseXY = True
		# prefix_reverseXY = "_rev_" if reverseXY else ""
		# plot_acounts_for_percentage(learner_sentences, ax, concat, name, reverseXY=reverseXY)
		# if save:
		# 	plt.savefig(PLOTS_DIR + fig_prefix + prefix_reverseXY + r"percentage_hist" + ".svg")
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


def plot_expected_best_coverage(dist, ax, title_addition="", show=True, save_name=None, xlabel=None):
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

	x = []
	y = []
	for correction_index, correction_num in enumerate(CORRECTION_NUMS):
		ps = np.fromiter((coverage[correction_index] for coverage in coverage_by_corrections_num), np.float)
		expected_accuracy = pb.mu(ps)/len(ps)
		x.append(correction_num)
		y.append(expected_accuracy)

	ax.plot(CORRECTION_NUMS, y)
	ax.set_ylabel("Expected accuracy")
	if xlabel:
		ax.set_xlabel(xlabel)
	ax.set_title("Expected accuracy for perfect corrected text by corrections number\n" + title_addition)
	if save_name:
		plt.savefig(save_name)
	if show:
		plt.show()
	plt.cla()


def plot_coverage_for_each_sentence(dist, axes, title_addition="", show=True, save_name=None, xlabel=None):
	""" plots expected accuracy result for each correction number
		axes - a subscriptable object of axis to plot for each comparison meathod
		dist - list of lists of Ys : distribution->measure->correction num(Y)
		"""
	for sent_key, ys in enumerate(dist):
		colors = rainbow_colors(range(len(dist)))
		for i, y in enumerate(ys):
			ax = axes[i]
			ax.plot(CORRECTION_NUMS, y, color=colors[sent_key])
			ax.set_ylabel(MEASURE_NAMES[i])
			if xlabel:
				ax.set_xlabel(xlabel)
			ax.set_title(MEASURE_NAMES[i] + " of different amount of corrections\n" + title_addition)
	if save_name:
		plt.savefig(save_name)
	if show:
		plt.show()
	plt.cla()


def plot_hist(l, ax, data, comparison_by, bottom=1):
	width = 1.0/len(l)
	for i, name in enumerate(l):
		y = create_hist(data[i], bottom=bottom)
		x = np.array(range(len(y)))
		print(comparison_by, " hist results \n", name,"\n",y)
		colors = rainbow_colors(range(len(l)))
		ax.plot(x, y, color=colors[i], label=name)
	plt.ylim(ymin=0)
	plt.autoscale(enable=True, axis='x', tight=False)
	plt.ylabel("amount of corrections")
	plt.xlabel("number of times seen")
	plt.title("hist of the number repetitions a correction was seen, using " + comparison_by + " comparison")
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
		colors = rainbow_colors(range(len(l)))
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
		# ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=name)
	plt.autoscale(enable=True, axis='x', tight=False)
	xlabel = "percentage of probability each correction accounts for"
	ylabel = "amount of corrections"
	if reverseXY:
		xlabel, ylabel = ylabel, xlabel
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title("hist of the number repetitions a correction was seen, using " + comparison_by + " comparison")
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
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=name)
	plt.autoscale(enable=True, axis='x', tight=False)
	if percentage:
		plt.ylabel("percentage from total corrections")
	else:
		plt.ylabel("amount of corrections")
	plt.ylim(ymin=0)
	plt.xlabel("number of times seen")
	plt.title("hist of the number repetitions a correction was seen, using " + comparison_by + " comparison")
	plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True)
	# plt.tight_layout()

def get_all_sentences_corrected():
	""" returns an iterable containing all the sentences that were corrected"""
	corrected = set()
	for root, dirs, files in os.walk(corrections_dir):
		for file in files:
			if isBatchFile(file):
				db = pd.read_csv(root+file)
				corrected.update(set(iter(db[LEARNER_SENTENCES_COL])))

def create_hist(l, top=float("inf"), bottom=0):
	""" converts a int counter to a sorted list for a histogram"""
	count = Counter(l)
	hist = [0] * (max(count.keys()) - bottom + 1)
	for key, val in count.items():
		if key <= top and key >= bottom:
			hist[key - bottom] = val
	return hist if hist else [0]

def rainbow_colors(labels):
	"""creates colors, each corresponding to a unique label"""
	cls = set(labels)
	if len(cls) == 2:
		return dict(zip(cls, ("blue", "orange")))
	return dict(zip(cls, cm.rainbow(np.linspace(0, 1, len(cls)))))

###################################################################################
####							general\NLP	
###################################################################################

def isBatchFile(filename):
	return "batch" in filename and filename.split(".")[-1].lower() == "csv"

def normalize_sentence(s):
	s = re.sub(r"\W+", r" ", s)
	s = re.sub(r"\s([a-zA-Z]\s)", r"\1", s)
	return s


def is_same_words(w1, w2):
	return preprocess_word(w1) == preprocess_word(w2)


def convert_sentence_to_diff_indexes(original, sentence):
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