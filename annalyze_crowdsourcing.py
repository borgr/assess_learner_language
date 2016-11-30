import pandas as pd
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
corrections_dir = r"C:\Users\User\Google Drive\nlp\scripts\sent\\"
learner_sentences_col = "Input.sentence"
corrected_sentences_col = "Answer.WritingTexts"

def normalize_sentence(s):
	return re.sub(r"\W+", " ", s)

def convert_sentence_to_diff_indexes(original, sentence):
	print(align_sentence_words(original, sentence, True))

def main():
	db = pd.read_csv(corrections_dir + r"Batch_2608242_batch_results.csv")
	db[corrected_sentences_col] = db[corrected_sentences_col].apply(normalize_sentence)
	learner_sentences = db[learner_sentences_col].unique()
	colors = rainbow_colors(range(len(learner_sentences)))
	plot_num = 200+(len(learner_sentences)+1)/2*10
	convert_sentence_to_diff_indexes(db[0,learner_sentences],db[0,corrected_sentences_col])
	# width = 1/len(learner_sentences)
	concat = []
	amount_of_corrections = []
	for color, sentence in zip(colors.values(),learner_sentences):
		corrected = db[corrected_sentences_col][db[learner_sentences_col] == sentence]
		counts = corrected.value_counts()
		concat.append(counts)
		amount_of_corrections.append(len(counts))

	# 	y = create_hist(counts)
	# 	x = range(len(y))
	# 	ax = plt.subplot(str(plot_num))
	# 	plot_num += 1
	# 	ax.plot(x, y, color=color)
		# ax.ylabel("amount")
		# ax.xlabel("different corrections")
		# ax.title("sorted number of corrections per sentence")
	# 	ax.legend(loc=7, fontsize=10)
	ax = plt.subplot("111")
	plot_hist(learner_sentences, ax, concat)
	plt.show()

	ax = plt.subplot("111")
	plot_differences_hist(learner_sentences, ax, concat)
	plt.show()
	print(amount_of_corrections)

def plot_hist(l, ax, data, bottom=1):
	width = 1.0/len(l)
	print(width)
	for i, name in enumerate(l):
		y = create_hist(data[i], bottom=bottom)
		x = np.array(range(len(y)))
		print(" hist results ", name,":",y)
		colors = rainbow_colors(range(len(l)))
		ax.plot(x + i*width, y, color=colors[i], label=name)
	plt.autoscale(enable=True, axis='x', tight=False)
	plt.ylabel("amount")
	plt.xlabel("different corrections")
	plt.title("sorted number of correction")
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()

def plot_differences_hist(l, ax, data, bottom=1):
	width = 1.0/len(l)
	print(width)
	for i, name in enumerate(l):
		y = create_hist(data[i], bottom=bottom)
		x = np.array(range(len(y)))
		print(" hist results ", name,":",y)
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=name)
	plt.autoscale(enable=True, axis='x', tight=False)
	plt.ylabel("amount")
	plt.xlabel("different corrections")
	plt.title("sorted number of correction")
	plt.legend(loc=7, fontsize=10)
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
				corrected.update(set(iter(db[learner_sentences_col])))

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

if __name__ == '__main__':
	main()