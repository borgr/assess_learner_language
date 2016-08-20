# built in packages
from itertools import accumulate
import math
import re
import sys
from collections import Counter

# dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import distance
from munkres import Munkres, print_matrix

from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.stem import WordNetLemmatizer

# ucca
sys.path.append('/home/borgr/ucca/ucca/scripts/distances')
sys.path.append('/home/borgr/ucca/ucca/ucca')
sys.path.append('/home/borgr/ucca/ucca')
import align

#constants
lemmatizer = WordNetLemmatizer()
SENTENCE_END = "[\.\?\!]" #TODO add ... ??? !!! ?!
SENTENCE_ENDS_WITH_NO_SPACE_PATTERN = re.compile("(.*?\w\w" + SENTENCE_END +")(\w+[^\.].*)")
SPACE_BEFORE_SENTENCE_PATTERN = re.compile("(.*?\s" + SENTENCE_END +"(\s*\")?)(.*)")

MAX_DIST = 2
SHORT_WORD_LEN = 4
CHANGING_RATIO = 5
PATH = r"/home/borgr/ucca/data/paragraphs/"

ORDERED = "original order"
FIRST_LONGER = "sentence splitted"
SECOND_LONGER = "sentence concatenated"
ORDERED_ALIGNED = "ORDERED with align"
FIRST_LONGER_ALIGNED = "first longer with align"
SECOND_LONGER_ALIGNED = "second longer with align"
REMOVE_LAST = "remove last"
PARAGRAPH_END = "paragraph end"
NO_ALIGNED = ""

print("clean all TODO")

###########################################################
####                    GENEERAL NLP                    ###
###########################################################

def is_word(w):
	return True if w != align.EMPTY_WORD and re.search('\w', w) else False

def split_by_pattern(tokens, p, first=1, second=2):
	""" gets a list of tokens and splits tokens by a compiled regex pattern
		param:
		tokens - list of strings representing sentence or sentences
		p - compiled regex pattern or object containing method match() that returns match object
		first - the group number that represents the first token found
		second - the group number that represents the second token found"""

	res = []
	for i, token in enumerate(tokens):
		matched = p.match(token)
		while matched:
			assert(matched.group(first) + matched.group(second) == token)
			res.append(matched.group(first))
			token = matched.group(second)
			matched = p.match(token)
		if token.strip():
			res.append(token)
	return res


def sent_tokenize(s):
	"""tokenizes a text to sentences"""
	tokens = nltk_sent_tokenize(s)
	tokens = split_by_pattern(tokens, SENTENCE_ENDS_WITH_NO_SPACE_PATTERN)
	tokens = split_by_pattern(tokens, SPACE_BEFORE_SENTENCE_PATTERN, 1, 3)
	return tokens


def word_tokenize(s):
	"""tokenizes a sentence to words list"""
	return [w for w in align.word_tokenize(s) if is_word(w)]


def preprocess_paragraph(p):
	"""preprocesses a paragraph"""
	p = re.sub("\s+", " ", p)
	p = re.sub("(" + SENTENCE_END +"\s+)" + SENTENCE_END, "\1", p)
	return p


def preprocess_word(w):
	if w and not w[-1].isalnum():
		w = w[:-1]
	return align.preprocess_word(w)


def approximately_same_word(w1, w2):
	""" returns if both words are considered the same word with a small fix or not"""
	l1 = lemmatizer.lemmatize(w1)
	l2 = lemmatizer.lemmatize(w2)
	allowed_dist = MAX_DIST if len(l1) > SHORT_WORD_LEN and len(l2) > SHORT_WORD_LEN else 1 #TODO maybe this is too harsh? and letter switch is also allowed?
	if (distance.levenshtein(l1, l2) > allowed_dist or
		w1 == align.EMPTY_WORD or w2 == align.EMPTY_WORD):
		#TODO should "the" "a" etc be considered in a different way? maybe they should not but not in this function
		return False #TODO words such as in at on etc, might be considered all equal to each other and to the empty_word for our purpose
	return True


def _choose_ending_position(sentences, endings, i):
	""" i - sentence number
		sentences - list of sentences
		endings - list of sentences positions endings

		return position, last word in the i'th sentence"""
	for word in reversed(word_tokenize(sentences[i])): # maybe word tokenizer is better? for things like and also demon,2012 and demon, 2012 face-to-face and face to face? or is splitting by - too will suffice?
		word = preprocess_word(word)
		if len(word) > 1:
			return endings[i], word
	print ("sentence contains no words:\n\"", sentences[i], "\"")
	print("sentence before", sentences[i-1])
	print("sentence after", sentences[i+1])
	raise "should not happen"
	return endings[i], preprocess_word(word_tokenize(sentences[i])[-1])


def word_diff(s1, s2):
	""" counts the number of aligned words that are not considered approximately the same word in 2 sentences"""
	alignment, indexes = align.align(s1, s2, True)
	return sum(not approximately_same_word(preprocess_word(w1), preprocess_word(w2)) for i, (w1, w2) in enumerate(alignment) if is_word(w1) or is_word(w2))


def diff_words(s1, s2):
	""" returns the aproximately different words in the two sentences"""
	alignment, indexes = align.align(s1, s2, True)
	return [(w1, w2) for i, (w1, w2) in enumerate(alignment) if (is_word(w1) or is_word(w2)) and not approximately_same_word(preprocess_word(w1), preprocess_word(w2))]


def calculate_endings(sentences, paragraph):
	""" gets sentences splitted from a paragraph and returns the sentences endings positions"""
	current = 0
	endings = []
	for s in sentences:
		current += len(s)
		while current < len(paragraph) and paragraph[current] == " ":
			current += 1
		endings.append(current)
	return endings


###########################################################
####                    WORDS CHANGED                   ###
###########################################################


def aligned_ends_together(shorter, longer, reg1, reg2, addition=""):
	""" checks if two sentences, ending in two regularized words ends at the same place.
	"""
	sentence1 = shorter
	sentence2 = longer + addition
	addition_words = word_tokenize(addition) if addition else word_tokenize(longer)
	addition_words = set(preprocess_word(w) for w in addition_words)
	slen1 = len(word_tokenize(sentence1))
	slen2 = len(word_tokenize(sentence2))
	if abs(slen1 - slen2) > min(slen1, slen2) / CHANGING_RATIO:
		return False

	aligned, indexes = align.align(sentence1, sentence2, True)
	aligned = set(map(lambda x:(preprocess_word(x[0]), preprocess_word(x[1])), aligned))
	mapping = dict(aligned)
	rev = dict(align.reverse_mapping(aligned))
	empty = preprocess_word(align.EMPTY_WORD)
	# print(rev)
	# print(reg2)
	# print(aligned)
	if ((reg1, empty) in aligned):
		if reg2 in addition_words and approximately_same_word(reg2, rev[reg2]):
			return True
	if ((empty, reg2) in aligned):
		if mapping[reg1] in addition_words and approximately_same_word(reg1, mapping[reg1]):
			return True
	return False

def break2common_sentences(p1, p2):
	"""finds the positions of the common sentence ending

	Breaking is done according to the text of both passages
	returns two lists each containing positions of sentence endings
	guarentees same number of positions is acquired and the last position is the passage end
	return:
		positions1, positions2 - lists of indexes of the changed """
	aligned_by = []
	s1 = sent_tokenize(p1)
	s2 = sent_tokenize(p2)

	# calculate sentence endings positions
	endings1 = calculate_endings(s1, p1)
	endings2 = calculate_endings(s2, p2)

	# find matching endings to match
	positions1 = []
	positions2 = []
	i = 0
	j = 0
	inc = False
	while i < len(s1) and j < len(s2):
		one_after1 = "not_initialized"
		one_after2 = "not_initialized"

		# create a for loop with two pointers
		if inc:
			i += 1
			j += 1
			inc = False
			continue

		inc = True
		position1, reg1 = _choose_ending_position(s1, endings1, i)
		position2, reg2 = _choose_ending_position(s2, endings2, j)
		if approximately_same_word(reg1, reg2):
			# print(ORDERED, " ",i)
			aligned_by.append(ORDERED)
			positions1.append(position1)
			positions2.append(position2)
			continue

		#deal with addition or subtraction of a sentence ending
		slen1 = len(word_tokenize(s1[i]))
		slen2 = len(word_tokenize(s2[j]))

		if i + 1 < len(s1) and slen1 < slen2:
			pos_after1, one_after1 = _choose_ending_position(s1, endings1, i + 1)
			if approximately_same_word(one_after1, reg2):
				# print(FIRST_LONGER, " ", i)
				aligned_by.append(FIRST_LONGER)
				positions1.append(pos_after1)
				positions2.append(position2)
				i += 1
				continue
		
		if j + 1 < len(s2) and slen2 < slen1:
			pos_after2, one_after2 = _choose_ending_position(s2, endings2, j + 1)
			if approximately_same_word(reg1, one_after2):
				# print(SECOND_LONGER, " ", i)
				aligned_by.append(SECOND_LONGER)
				positions1.append(position1)
				positions2.append(pos_after2)
				j += 1
				continue

		# no alignment found with 2 sentences
		# check if a word was added to the end of one of the sentences
		if aligned_ends_together(s1[i], s2[j], reg1, reg2):
			# print(ORDERED_ALIGNED, " ",i)
			aligned_by.append(ORDERED_ALIGNED)
			positions1.append(position1)
			positions2.append(position2)
			continue

		# # if no match is found twice and we had ORDERED match, it might have been a mistake
		# if aligned_by[-1] == NO_ALIGNED and aligned_by[-2] == NO_ALIGNED:
		# 	print("using fallback")
		# 	removed_pos1 = positions1.pop()
		# 	removed_pos2 = positions2.pop()
		# 	aligned_by.append(REMOVE_LAST)
		# 	i -= 2
		# 	j -= 2
		# 	print("s1:",s1[i])
		# 	print("s2:",s2[j])
		# 	print("s1af:",s1[i+1])
		# 	print("s2af:",s2[j+1])

		# check if a word was added to the end of one of the sentences
		# Also, deal with addition or subtraction of a sentence ending
		if i + 1 < len(s1) and slen1 < slen2:
			if aligned_ends_together(s2[j], s1[i], reg2, one_after1, addition=s1[i + 1]):
				# print(FIRST_LONGER_ALIGNED, " ",i)
				aligned_by.append(FIRST_LONGER_ALIGNED)
				positions1.append(pos_after1)
				positions2.append(position2)
				i += 1
				continue

		if j + 1 < len(s2) and slen2 < slen1:
			if aligned_ends_together(s1[i], s2[j], reg1, one_after2, addition=s2[j + 1]):
				# print(SECOND_LONGER_ALIGNED, " ", i)
				aligned_by.append(SECOND_LONGER_ALIGNED)
				positions1.append(position1)
				positions2.append(pos_after2)
				j += 1
				continue

		# # removing last yielded no consequences keep in regular way
		# if aligned_by[-1] == REMOVE_LAST:
		# 	positions1.append(removed_pos1)
		# 	positions2.append(removed_pos2)
		# 	i -= 2
		# 	j -= 2

		# print (i, reg1, reg2, one_after1, one_after2)
		# print("s1:",s1[i])
		# print("s2:",s2[j])
		# print("s1af:",s1[i+1])
		# print("s2af:",s2[j+1])
		# print(i)
		# print("------------------")
		aligned_by.append(NO_ALIGNED)

	# add last sentence in case skipped
		position1, reg1 = _choose_ending_position(s1, endings1, -1)
		position2, reg2 = _choose_ending_position(s2, endings2, -1)
	if (not positions1) or (not positions2) or (
	    positions1[-1] != position1 and positions2[-1] != position2):
		positions1.append(endings1[-1])
		positions2.append(endings2[-1])
		aligned_by.append(PARAGRAPH_END)
	elif positions1[-1] != position1 and positions2[-1] == position2:
		positions1[-1] = endings1[-1]
		aligned_by.append(PARAGRAPH_END)
	elif positions1[-1] == position1 and positions2[-1] != position2:
		positions2[-1] = endings2[-1]
		aligned_by.append(PARAGRAPH_END)

	return positions1, positions2, aligned_by


def get_sentences_from_endings(paragraph, endings):
	"""a generator of sentences from a paragraph and ending positions in it"""
	last = 0
	for cur in endings:
		yield paragraph[last:cur]
		last = cur


def compare_paragraphs(origin, corrected):
	""" compares two paragraphs
		return:
		broken - the sentence endings indexes
		differences - difference measures corresponding to the indexes in broken
		aligned_by - the way the sentences were aligned"""
	print("comparing paragraphs")
	print("aligning sentences")
	broken = [None,None]
	broken[0], broken[1], aligned_by = break2common_sentences(origin, corrected)
	print("assesing differences")
	origin_sentences = get_sentences_from_endings(origin, broken[0])
	corrected_sentences = get_sentences_from_endings(corrected, broken[1])
	differences = [word_diff(orig, cor) for orig, cor in zip(origin_sentences, corrected_sentences)]
	print("comparing done")
	return broken, differences, aligned_by


def read_paragraph(filename):
	with open(PATH + filename) as fl:
		return preprocess_paragraph(fl.read())


def extract_aligned_by_dict(a):
	""" takes aligned_by list and creates a counter of ordered, first longer and second longer sentences"""
	count = Counter(a)
	res = Counter()
	res[ORDERED] = count[ORDERED] + count[ORDERED_ALIGNED]
	res[FIRST_LONGER] = count[FIRST_LONGER] + count[FIRST_LONGER_ALIGNED]
	res[SECOND_LONGER] = count[SECOND_LONGER] + count[SECOND_LONGER_ALIGNED]
	return res


###########################################################
####                    VISUALIZATION                   ###
###########################################################


def create_hist(l):
	""" converts a int counter to a sorted list for a histogram"""
	count = Counter(l)
	hist = [0] * (max(count.keys()) + 1)
	for key, val in count.items():
		hist[key] = val
	return hist


def rainbow_colors(labels):
	"""creates colors, each corresponding to a unique label"""
	cls = set(labels)
	if len(cls) == 2:
		return dict(zip(cls, ("blue", "orange")))
	return dict(zip(cls, cm.rainbow(np.linspace(0, 1, len(cls)))))


def plot_differences_hist(l, ax):
	""" gets a list of (broken, differences, aligned_by, name) tuples and plot the hists"""
	broken, differences, aligned_by, name = list(range(4)) # tuple structure
	width = 1/len(l)
	for i, tple in enumerate(l):
		y = create_hist(tple[differences])
		x = np.array(range(len(y)))
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=tple[name])
	ax.autoscale(tight=True)
	plt.ylabel("amount")
	plt.xlabel("number of words changed")
	plt.title("number of words changed by method of correction")
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()


def plot_aligned_by(l, ax):
	""" gets a list of (broken, differences, aligned_by, name) tuples and plot """
	broken, differences, aligned_by, name = list(range(4)) # tuple structure
	width = 1/len(l)
	for i, tple in enumerate(l):
		y = extract_aligned_by_dict(tple[aligned_by])
		y = [y[FIRST_LONGER], y[ORDERED], y[SECOND_LONGER]]
		x = np.array(range(len(y)))
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=tple[name])
	ax.autoscale(tight=True)
	plt.ylabel("amount")
	plt.xlabel("number of sentence changes of that sort")
	plt.title("accumulative number of sentence changes by method of correction")
	plt.xticks(x + width, (FIRST_LONGER, ORDERED, SECOND_LONGER))
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()

def plot_not_aligned(l, ax):
	""" gets a list of (broken, differences, aligned_by, name) tuples and plot the bars"""
	broken, differences, aligned_by, name = list(range(4)) # tuple structure
	width = 1/len(l)
	for i, tple in enumerate(l):
		y = extract_aligned_by_dict(tple[aligned_by])
		y = [y[FIRST_LONGER], y[SECOND_LONGER]]
		x = np.array(range(len(y)))
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=tple[name])
	ax.autoscale(tight=True)
	plt.ylabel("amount")
	plt.xlabel("number of sentence changes of that sort")
	plt.title("number of sentence changes by method of correction")
	plt.xticks(x + width, (FIRST_LONGER, SECOND_LONGER))
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()

def plot_differences(l, ax):
	""" gets a list of (broken, differences, aligned_by, name) tuples and plot the plots"""
	broken, differences, aligned_by, name = list(range(4)) # tuple structure
	width = 0.2
	ys = []
	max_len = 0
	colors = rainbow_colors(range(len(l)))

	for i, tple in enumerate(l):
		y = create_hist(tple[differences])[1:]
		ys.append((y,tple[name],colors[i]))
		max_len = max(max_len, len(y))
	
	x = np.array(range(max_len))

	for y,name, color in ys:
		y = y + [0]*(max_len-len(y))
		ax.plot(x, np.cumsum(y), color=color, label=name)
	ax.autoscale(tight=True)
	plt.ylabel("amount")
	plt.xlabel("number of words changed")
	plt.title("accumulative number of sentences by words changed per")
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()


def plot_comparison(l):
	"""gets a list of tuple parameters and plots them"""
	data = []
	ax = plt.subplot(221)
	plot_differences(l, ax)
	ax = plt.subplot(222)
	plot_differences_hist(l, ax)
	ax = plt.subplot(223)
	plot_aligned_by(l, ax)
	ax = plt.subplot(224)
	plot_not_aligned(l, ax)
	plt.show()


if __name__ == '__main__':

	ACL2016RozovskayaRothOutput_file = "conll14st.output.1cleaned"
	learner_file = "conll.tok.orig"
	gold_file = "corrected_official-2014.0.txt.comparable"

	autocorrect = read_paragraph(ACL2016RozovskayaRothOutput_file)
	origin = read_paragraph(learner_file)
	gold = read_paragraph(gold_file)

	res_list = []

	# compare origin to ACL2016RozovskayaRoth autocorrect
	broken, differences, aligned_by = compare_paragraphs(origin, autocorrect)
	comparison_sentences = list(get_sentences_from_endings(autocorrect, broken[1]))
	origin_sentences = list(get_sentences_from_endings(origin, broken[0]))
	ACL2016RozovskayaRoth_autocorrect_hist = create_hist(differences)
	res_list.append((broken, differences, aligned_by, "Rozovskaya Roth"))
	# print()
	# print("number of sentences with more than 2 changes:", sum([1 for d in differences if d > 2]))
	# print("number of sentences with more than 3 changes:", sum([1 for d in differences if d > 3]))

	# compare gold to origin
	broken, differences, aligned_by = compare_paragraphs(origin, gold)
	comparison_sentences =  list(get_sentences_from_endings(gold, broken[1]))
	origin_sentences = list(get_sentences_from_endings(origin, broken[0]))
	res_list.append((broken, differences, aligned_by, "gold standard"))

	plot_comparison(res_list)

	# prints
	for i, dif in enumerate(differences):
		if dif > 10: # or i < 3 # use i to print some, use diff to print all sentences which differ ion more than "diff" words from each other
			print("-------\nsentences:\n", comparison_sentences[i],"\norignal:\n", origin_sentences[i])
			print ("word dif:", dif)
			print("match num:", i)