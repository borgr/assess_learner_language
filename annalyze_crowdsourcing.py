import operator
import functools
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
data_dir = r"/home/borgr/ucca/assess_learner_language/calculations_data/"
plots_dir = r"/home/borgr/ucca/assess_learner_language/plots/corrections/"
hists_dir = r"/home/borgr/ucca/assess_learner_language/unseenEst/"
batch_files = [r"Batch_2612793_batch_results.csv", r"Batch_2626033_batch_results.csv"]
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
correction_nums = list(range(20))

def main():
	frames = []
	for batch_file in batch_files:
		frames.append(pd.read_csv(corrections_dir + batch_file))
	db = pd.concat(frames)
	db[CORRECTED_SENTENCES_COL] = db[CORRECTED_SENTENCES_COL].apply(normalize_sentence)
	learner_sentences = db[LEARNER_SENTENCES_COL].unique()
	show=False
	save=False

	# print(db[LEARNER_SENTENCES_COL])
	compare_correction_distributions(db, EXACT_COMP, show=show, save=save)

		
	db[INDEXES_CHANGED_COL] = find_changed_indexes(learner_sentences, db[LEARNER_SENTENCES_COL], db[CORRECTED_SENTENCES_COL])
	compare_correction_distributions(db, INDEX_COMP, index=INDEXES_CHANGED_COL, show=show, save=save)
	for root, dirs, files in os.walk(hists_dir):
		for filename in files:
			if INPUT_HIST_IDENTIFIER in filename:
				assessed_new = assessed_new and assess_real_distributions(root+filename, str(0))

	assess_coverage(True)
	coverage_by_corrections_num = assess_coverage(False)
	print(coverage_by_corrections_num)

def assess_coverage(only_different_samples):
	all_ys = [[] for i in range(len(COMPARISON_METHODS))]
	for root, dirs, files in os.walk(hists_dir):
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
				for correction_num in correction_nums:
					results.append(compute_probabillity_to_account_async(dist, correction_num, REPETITIONS, only_different_samples))
				results = np.array(results)
				ys = []
				for coverage_method in COVERAGE_METHODS:
					ys.append(coverage_method(results))
				ys = np.array(ys)
				# save y
				for i, comparison_method in enumerate(COMPARISON_METHODS):
					if comparison_method in filename:
						all_ys[i].append(ys)
	#list by: comparison method->distribution->measure->correction num(Y)
	res = []
	all_ys = np.array(all_ys)
	axis_num = len(all_ys[0][0])
	ax = plt.subplot("1"+str(axis_num)+"1")
	colors = rainbow_colors(range(len(all_ys)))
	for comparison_method_key, dist in enumerate(all_ys):
		axes = [plt.subplot("1"+str(axis_num)+str(i+1)) for i in range(len(ys))]
		for sent_key, ys in enumerate(dist):
			for i, y in enumerate(ys):
				ax = axes[i]
				ax.plot(correction_nums, y, color=colors[sent_key]) #label=sentence
				ax.set_ylabel(MEASURE_NAMES[i])
				if only_different_samples:
					ax.set_xlabel("amount of different corrections")
				else:
					ax.set_xlabel("amount of corrections sampled")
				ax.set_title(MEASURE_NAMES[i] + " of different amount of corrections\n using " + COMPARISON_METHODS[comparison_method_key] + " comparison")
				if COMPARISON_METHODS[comparison_method_key] == EXACT_COMP and MEASURE_NAMES[i] == MEAN_MEASURE:
					res.append(y)
		repeat = "" if only_different_samples else "repeat"
		repeat += "_" + str(REPETITIONS)
		fig_prefix = COMPARISON_METHODS[comparison_method_key] +"_" + repeat
		plt.savefig(plots_dir + fig_prefix + r"_coverage" + ".svg")
		plt.show()
		plt.cla()
	return np.array(res)


def poisson_binomial_PDF_possion_approximation(ps, n):
	mu = sum(ps)
	return (mu**n)*math.exp(-mu)/math.factorial(n)


def poisson_binomial_PDF(ps, n):
	return functools.reduce(operator.mul, (1-x for x in ps), poisson_binomial_R(ps, n,range(len(ps))))


def poisson_binomial_T(ps, i, chosen):
	np.sum((ps[w]**i for w in chosen))

def poisson_binomial_R(ps, n, chosen):
	if len(chosen) < n:
		return 0
	if not n:
		return 1
	# pool = Pool(10) #TODO is pooling needed? and working depiste recursion
	a = np.empty((n,),int)
	a[::2] = 1
	a[1::2] = -1
	it = []
	# it += list(pool.imap(__compute_coverage, range(1,n+1)))
	it = np.array((poisson_binomial_T(ps,i,chosen)*poisson_binomial_R(n-i, chosen) for i in range(1,n+1)))
	# pool.close()
	# pool.join()
	return np.inner(it, a)/n

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
		return False
	else:
		os.system("python /home/borgr/unseenest/src/unseen_est.py " + filename + " 0 " + outfile + " -s 1 ")
		return True

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

def compute_probabillity_to_account_async(distribution, samples, repetitions, only_different_samples):
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

def compute_probabillity_to_account(distribution, samples, repetitions, only_different_samples):
	# print("in")
	covered = np.zeros([repetitions,1])
	cdf = np.cumsum(distribution[PROB_COL])
	# inverted_function = lambda x:np.argmax(cdf < x)
	probs = distribution[PROB_COL]/sum(distribution[PROB_COL])

	for i in range(repetitions):
		covered[i] = compute_coverage(cdf, probs, distribution, samples, only_different_samples)
		# # if i % (repetitions/10) == 0:
		# # 	print("in", i, "th repetition")
		# uncovered_variants_num = distribution[VARIANTS_NUM_COL].copy()
		# uncovered_probes = distribution[PROB_COL].copy()
		# sampled = 0
		# while sampled < samples and not cmath.isclose(covered[i], 1.0):
		# 	chosen = np.random.choice(np.arange(len(distribution[PROB_COL])), p=p)
		# 	covered[i] += min(uncovered_variants_num[chosen], samples - sampled) * distribution[PROB_COL][chosen]
		# 	if only_different_samples:
		# 		p = uncovered_probes/sum(uncovered_probes)
		# 		assert(not np.isnan(sum(p)))
		# 		uncovered_probes[chosen] = 0
		# 		sampled += uncovered_variants_num[chosen]
		# 	else:
		# 		sampled += distribution[VARIANTS_NUM_COL][chosen]
		# 	uncovered_variants_num[chosen] = 0
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
			plt.savefig(plots_dir + fig_prefix + r"_hist" + ".svg")
		if show:
			plt.show()

		plt.cla()
		ax = plt.subplot("111")
		plot_acounts_for_percentage(learner_sentences, ax, concat, name)
		if save:
			plt.savefig(plots_dir + fig_prefix + r"percentage_hist" + ".svg")
		if show:
			plt.show()

		# ax = plt.subplot("111")
		# reverseXY = True
		# prefix_reverseXY = "_rev_" if reverseXY else ""
		# plot_acounts_for_percentage(learner_sentences, ax, concat, name, reverseXY=reverseXY)
		# if save:
		# 	plt.savefig(plots_dir + fig_prefix + prefix_reverseXY + r"percentage_hist" + ".svg")
		# if show:
		# 	plt.show()
	export_hists(learner_sentences, concat, name, hists_dir)


def export_hists(l, data, comparison_by, hists_dir):
	for i, name in enumerate(l):
		filename = re.sub("\W","",name)[:6]
		filename = hists_dir + filename + "_"+ comparison_by + "_" + INPUT_HIST_IDENTIFIER +".txt"
		y = create_hist(data[i], bottom=1)
		y = [str(val)+"\n" for val in y]
		with open(filename, "w+") as fl:
			fl.writelines(y)


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
	xlabel = "percentage of probabillity each correction accounts for"
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

def isBatchFile(filename):
	return "batch" in filename and filename.split(".")[-1].lower() == "csv"

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
				# print("aligned to empty word", (w1,w2), (i,j))
			else:
				indexes.append(i)
	return tuple(sorted(indexes))

if __name__ == '__main__':
	main()