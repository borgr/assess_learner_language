# built in packages
from itertools import islice
import math
import re
import sys
import os
import csv
from collections import Counter

# dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import distance
import json
from scipy.stats import spearmanr
# import scikits.statsmodels as sm
# from statsmodels.distributions.empirical_distribution import ECDF

from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.stem import WordNetLemmatizer
trial_name = ""

# ucca
sys.path.append('/home/borgr/ucca/ucca/scripts/distances')
sys.path.append('/home/borgr/ucca/ucca/ucca')
sys.path.append('/home/borgr/ucca/ucca')
import align

#constants
lemmatizer = WordNetLemmatizer()
ENDERS_DEFINITELY = r"\?\!\;" 
ENDERS = r"\." + ENDERS_DEFINITELY
SENTENCE_NOT_END = "[^" + ENDERS + "]"
SENTENCE_END = "[" + ENDERS + "]"
NOT_ABBREVIATION_PATTERN = re.compile(r"(.*?\s+\w\s*\.)(\s*\w\w.*)")
SENTENCE_DEFINITELY_PATTERN = re.compile(r"(.+\s*[" + ENDERS_DEFINITELY + r"]\s*)(.+)")
SENTENCE_ENDS_WITH_NO_SPACE_PATTERN = re.compile("(.*?\w\w" + SENTENCE_END +")(\w+[^\.].*)")
SPACE_BEFORE_SENTENCE_PATTERN = re.compile("(.*?\s" + SENTENCE_END +"(\s*\")?)(.*)")
SPECIAL_WORDS_PATTERNS = [re.compile(r"i\s*\.\s*e\s*\.", re.IGNORECASE), re.compile(r"e\s*\.\s*g\s*\.", re.IGNORECASE), re.compile(r"\s+c\s*\.\s+", re.IGNORECASE)]
SPECIAL_WORDS_REPLACEMENTS = ["ie", "eg", " c."]
MAX_SENTENCES = 1400 # accounts for the maximum number of lines to get from the database
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
COMMA_REPLACE_FIRST = ", in second sentence became the end of a new sentence (first longer)"
COMMA_REPLACE_SECOND = ", in first sentence became the end of a new sentence (second longer)"
NO_ALIGNED = ""

def main():
	global trial_name
	trial_name = "_some_competitors"
	change_date = "160111"
	filename = "results/results"+ change_date + ".json"
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
	from fce import CORRECTED_FILE as fce_gold_file
	from fce import LEARNER_FILE as fce_learner_file
	autocorrect = read_paragraph(ACL2016RozovskayaRothOutput_file)
	amu = read_paragraph(amu_file)
	camb = read_paragraph(camb_file)
	cuui = read_paragraph(cuui_file)
	iitb = read_paragraph(iitb_file)
	ipn = read_paragraph(ipn_file)
	nthu = read_paragraph(nthu_file)
	pku = read_paragraph(pku_file)
	post = read_paragraph(post_file)
	rac = read_paragraph(rac_file)
	sjtu = read_paragraph(sjtu_file)
	ufc = read_paragraph(ufc_file)
	umc = read_paragraph(umc_file)
	origin = read_paragraph(learner_file, preprocess_paragraph)
	gold = read_paragraph(gold_file, preprocess_paragraph_minimal)
	fce_gold = read_paragraph(fce_gold_file)
	fce_learner = read_paragraph(fce_learner_file)
	fce_learner_full = read_paragraph(fce_learner_file, preprocess_paragraph_minimal)
	fce_gold_full = read_paragraph(fce_gold_file, preprocess_paragraph_minimal)
	res_list = []
	old_res = read(filename) if filename else {}
	for (name, res) in old_res.items():
		res.append(name)
		dump(res_list, filename)

	# compare fce origin to fce gold without matching
	name = "fce"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(fce_learner_full, fce_gold_full, sent_token_by_char, sent_token_by_char)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)

	# compare fce origin to fce gold
	name = "fce auto"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(fce_learner, fce_gold)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)

	# compare gold to origin
	name = "gold"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, gold, sent_tokenize_default, sent_token_by_char)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		print(len(old_res[name][1]))
		broken, word_differences, index_differences, spearman_differences, aligned_by, name = old_res[name]
		origin_sentences = list(get_sentences_from_endings(origin, broken[0]))
		corrected_sentences = list(get_sentences_from_endings(gold, broken[1]))
		res_list.append(old_res[name])
		print("\nmany words changed")
		for i, dif in enumerate(word_differences):
			if dif > 10: # or i < 3 # use i to print some, use diff to print all sentences which differ ion more than "diff" words from each other
				print("-------\nsentences:\n", corrected_sentences[i],"\norignal:\n", origin_sentences[i])
				print ("word dif:", dif)
				print("match num:", i)
		print("\nmany indexes changed")
		for i, dif in enumerate(index_differences):
			if dif > 10: # or i < 3 # use i to print some, use diff to print all sentences which differ ion more than "diff" words from each other
				print("-------\nsentences:\n", corrected_sentences[i],"\norignal:\n", origin_sentences[i])
				print ("word dif:", dif)
				print("match num:", i)
		print("\nmany swaps changed (spearman)")
		for i, dif in enumerate(spearman_differences):
			if dif < 0.9: # or i < 3 # use i to print some, use diff to print all sentences which differ ion more than "diff" words from each other
				print("-------\nsentences:\n", corrected_sentences[i],"\norignal:\n", origin_sentences[i])
				print ("word dif:", dif)
				print("match num:", i)

	# compare origin to cuui #1
	name = "cuui"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, cuui)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to camb #2
	name = "camb"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, camb)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to AMU #3
	name = "AMU"	
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, amu)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))	
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to ACL2016RozovskayaRoth autocorrect
	name = "Rozovskaya Roth"
	name = "RR"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, autocorrect)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to iitb
	name = "iitb"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, iitb)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to ipn
	name = "ipn"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, ipn)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to nthu
	name = "nthu"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, nthu)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to pku
	name = "pku"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, pku)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to post
	name = "post"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, post)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to rac
	name = "rac"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, rac)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to sjtu
	name = "sjtu"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, sjtu)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to ufc
	name = "ufc"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, ufc)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])

	# compare origin to umc
	name = "umc"
	print(name)
	if name not in old_res:
		broken, words_differences, index_differences, spearman_differences, aligned_by = compare_paragraphs(origin, umc)
		res_list.append((broken, words_differences, index_differences, spearman_differences, aligned_by, name))
		dump(res_list, filename)
	else:
		res_list.append(old_res[name])


	dump(res_list, filename)
	# plot_comparison(res_list)
	convert_file_to_csv(filename)


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

def concat_empty(tokens):
	"""concatenats empty sentences or words to the one before them in the list of tokens"""
	result = []
	for token in tokens:
		if re.search(r"[A-Za-z][A-Za-z]", token) is not None:
			result.append(token)
		elif result:
			result[-1] = result[-1] + token
	return result


def sent_token_by_char(s, char="\n"):
	"""tokenizes by predefined charachter"""
	return concat_empty(s.split(char))


def sent_tokenize_default(s):
	"""tokenizes a text to a list of sentences"""
	tokens = nltk_sent_tokenize(s)
	tokens = split_by_pattern(tokens, SENTENCE_DEFINITELY_PATTERN)
	tokens = split_by_pattern(tokens, SENTENCE_ENDS_WITH_NO_SPACE_PATTERN)
	tokens = split_by_pattern(tokens, SPACE_BEFORE_SENTENCE_PATTERN, 1, 3)
	tokens = split_by_pattern(tokens, NOT_ABBREVIATION_PATTERN)

	return concat_empty(tokens)


def word_tokenize(s):
	"""tokenizes a sentence to words list"""
	res = [w for w in align.word_tokenize(s) if is_word(w)]
	return res


def preprocess_paragraph_minimal(p):
	if p[-1] == "\n":
		p = p[:-1]
	return p


def preprocess_paragraph(p):
	"""preprocesses a paragraph"""
	for i, pattern in enumerate(SPECIAL_WORDS_PATTERNS):
		p = re.sub(pattern, SPECIAL_WORDS_REPLACEMENTS[i], p)
	# p = re.sub(r"\s+\.\s+", r".", p)
	p = re.sub(r"(" + SENTENCE_NOT_END + ")(\s*\n)", r"\1.\2", p)
	p = re.sub("(\.\s*['\"])\s*\.", r"\1", p)
	p = re.sub(r"\s+", r" ", p)
	p = re.sub(r"(" + SENTENCE_END +r"\s*)" + SENTENCE_END, r"\1", p)
	return p


def preprocess_word(w):
	if w and not w[-1].isalnum():
		w = w[:-1]
	return align.preprocess_word(w)


def approximately_same_word(w1, w2):
	""" returns if both words are considered the same word with a small fix or not"""
	l1 = lemmatizer.lemmatize(w1)
	l2 = lemmatizer.lemmatize(w2)
	allowed_dist = MAX_DIST if len(l1) > SHORT_WORD_LEN and len(l2) > SHORT_WORD_LEN else 1 
	if (distance.levenshtein(l1, l2) > allowed_dist or
		w1 == align.EMPTY_WORD or w2 == align.EMPTY_WORD):
		#suggestion: should "the" "a" etc be considered in a different way? maybe they should not but not in this function
		return False #suggestion: words such as in at on etc, might be considered all equal to each other and to the empty_word for our purpose
	return True


def _choose_ending_position(sentences, endings, i):
	""" i - sentence number
		sentences - list of sentences
		endings - list of sentences positions endings

		return position, last word in the i'th sentence"""
	for word in reversed(word_tokenize(sentences[i])):
		word = preprocess_word(word)
		if len(word) > 1:
			return endings[i], word
	print ("sentence contains no words:\n\"", sentences[i], "\"")
	print("sentence before", sentences[i-1])
	print("sentence after", sentences[i+1])
	assert(False)
	return endings[i], preprocess_word(word_tokenize(sentences[i])[-1])


def index_diff(s1, s2):
	""" counts the number of not aligned words in 2 sentences"""
	alignment, indexes = align_sentence_words(s1, s2, True)
	sorted_alignment_indexes = [(w1, w2, i1, i2) for (w1, w2), (i1, i2) in zip(alignment, indexes)]
	sorted_alignment_indexes = sorted(sorted_alignment_indexes, key = lambda x: x[3])
	last = -1
	res = 0

	for w1, w2, i1, i2 in sorted_alignment_indexes:
		if is_word(w1) and is_word(w2):
			if i1 < last:
				assert (i1 != -1 and i2 != -1)
				res += 1
			last = i1
	return res


def spearman_diff(s1, s2):
	""" counts the number of not aligned words in 2 sentences"""
	alignment, indexes = align_sentence_words(s1, s2, True)
	sorted_alignment_indexes = [(w1, w2, i1, i2) for (w1, w2), (i1, i2) in zip(alignment, indexes)]
	sorted_alignment_indexes = sorted(sorted_alignment_indexes, key = lambda x: x[3])
	changes = 0
	indexes1 = []
	indexes2 = []
	for w1, w2, i1, i2 in sorted_alignment_indexes:
		if is_word(w1) and is_word(w2):
			indexes1.append(i1)
			indexes2.append(i2)
	indexes1 = np.asarray(indexes1)
	indexes2 = np.asarray(indexes2)
	return spearmanr(indexes1, indexes2)



def word_diff(s1, s2):
	""" counts the number of aligned words that are not considered approximately the same word in 2 sentences"""
	alignment, indexes = align_sentence_words(s1, s2, True)
	return sum(not approximately_same_word(preprocess_word(w1), preprocess_word(w2)) for i, (w1, w2) in enumerate(alignment) if is_word(w1) or is_word(w2))


def diff_words(s1, s2):
	""" returns the aproximately different words in the two sentences"""
	alignment, indexes = align_sentence_words(s1, s2, True)
	return [(w1, w2) for i, (w1, w2) in enumerate(alignment) if (is_word(w1) or is_word(w2)) and not approximately_same_word(preprocess_word(w1), preprocess_word(w2))]


def calculate_endings(sentences, paragraph):
	""" gets sentences splitted from a paragraph and returns the sentences endings positions"""
	current = 0
	endings = []
	for s in sentences:
		current += len(s)
		while current < len(paragraph) and not paragraph[current].isalnum():
			current += 1
		endings.append(current)
	return endings

def align_sentence_words(s1, s2, isString, empty_cache=False):
	"""aligns words from sentence s1 to s2m, allows caching
		returns arrays of word tuplds and indexes tuples"""
	if empty_cache:
		align_sentence_words.cache={}
		return
	if (s1, s2, isString) in align_sentence_words.cache:
		return align_sentence_words.cache[(s1, s2, isString)]
	elif (s2, s1, isString) in align_sentence_words.cache:
		return align_sentence_words.cache[(s2, s1, isString)]
	else:
		res = align.align(s1, s2, isString)
		align_sentence_words.cache[(s2, s1, isString)] = res
		return res
align_sentence_words.cache={}

###########################################################
####                    WORDS CHANGED                   ###
###########################################################


def aligned_ends_together(shorter, longer, reg1, reg2, addition="", force=False):
	""" checks if two sentences, ending in two regularized words ends at the same place.
	"""
	sentence1 = shorter
	sentence2 = longer + addition
	addition_words = word_tokenize(addition) if addition else word_tokenize(longer)[len(word_tokenize(shorter)):]
	addition_words = set(preprocess_word(w) for w in addition_words)
	tokens1 = [preprocess_word(w) for w in word_tokenize(sentence1)]
	tokens2 = [preprocess_word(w) for w in word_tokenize(sentence2)]
	count1 = Counter()
	# if words appear more than once make each word unique by order of appearence 
	for i, token in enumerate(tokens1):
		if count1[token] > 0:
			tokens1[i] = str(count1[token]) + token
		if is_word(token):
			count1.update(token)
	count2 = Counter()
	for i, token in enumerate(tokens2):
		if count2[token] > 0:
			tokens2[i] = str(count2[token]) + token
		if is_word(token):
			count2.update(token)
	slen1 = len(tokens1)
	slen2 = len(tokens2)
	if abs(slen1 - slen2) > min(slen1, slen2) / CHANGING_RATIO:
		return False

	aligned, indexes = align_sentence_words(sentence1, sentence2, True)
	aligned = set(map(lambda x:(preprocess_word(x[0]), preprocess_word(x[1])), aligned))
	mapping = dict(aligned)
	rev = dict(align.reverse_mapping(aligned))
	empty = preprocess_word(align.EMPTY_WORD)

	if force or ((reg1, empty) in aligned):
		if  approximately_same_word(reg2, rev[reg2]):
			return True
	if force or ((empty, reg2) in aligned):
		if approximately_same_word(reg1, mapping[reg1]):
			return True
	return False

def break2common_sentences(p1, p2, sent_tokenize1, sent_tokenize2):
	"""finds the positions of the common sentence ending

	Breaking is done according to the text of both passages
	returns two lists each containing positions of sentence endings
	guarentees same number of positions is acquired and the last position is the passage end
	return:
		positions1, positions2 - lists of indexes of the changed """
	aligned_by = []
	s1 = sent_tokenize1(p1)
	s2 = sent_tokenize2(p2)

	# calculate sentence endings positions
	endings1 = calculate_endings(s1, p1)
	endings2 = calculate_endings(s2, p2)

	# find matching endings to match
	positions1 = []
	positions2 = []
	i = 0
	j = 0
	inc = False
	force = False
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
				aligned_by.append(FIRST_LONGER)
				positions1.append(pos_after1)
				positions2.append(position2)
				i += 1
				continue
		
		if j + 1 < len(s2) and slen2 < slen1:
			pos_after2, one_after2 = _choose_ending_position(s2, endings2, j + 1)
			if approximately_same_word(reg1, one_after2):
				aligned_by.append(SECOND_LONGER)
				positions1.append(position1)
				positions2.append(pos_after2)
				j += 1
				continue

		# no alignment found with 2 sentences
		# check if a word was added to the end of one of the sentences
		if aligned_ends_together(s1[i], s2[j], reg1, reg2):
			aligned_by.append(ORDERED_ALIGNED)
			positions1.append(position1)
			positions2.append(position2)
			continue

		# if no match is found twice and we had ORDERED match, it might have been a mistake
		if (positions1 and positions2 and
		   aligned_by[-1] == NO_ALIGNED and aligned_by[-2] == NO_ALIGNED):
			removed_pos1 = positions1.pop()
			removed_pos2 = positions2.pop()
			aligned_by.append(REMOVE_LAST)
			i -= 3
			j -= 3
			position1, reg1 = _choose_ending_position(s1, endings1, i)
			position2, reg2 = _choose_ending_position(s2, endings2, j)
			pos_after1, one_after1 = _choose_ending_position(s1, endings1, i + 1)
			pos_after2, one_after2 = _choose_ending_position(s2, endings2, j + 1)
			pos_2after1, two_after1 = _choose_ending_position(s1, endings1, i + 2)
			pos_2after2, two_after2 = _choose_ending_position(s2, endings2, j + 2)
			force = True

		# check if a word was added to the end of one of the sentences
		# Also, deal with addition or subtraction of a sentence ending
		if i + 1 < len(s1) and slen1 < slen2:
			if aligned_ends_together(s2[j], s1[i], reg2, one_after1, addition=s1[i + 1], force=force):
				aligned_by.append(FIRST_LONGER_ALIGNED)
				positions1.append(pos_after1)
				positions2.append(position2)
				i += 1
				continue

		if j + 1 < len(s2) and slen2 < slen1:
			if aligned_ends_together(s1[i], s2[j], reg1, one_after2, addition=s2[j + 1], force=force):
				aligned_by.append(SECOND_LONGER_ALIGNED)
				positions1.append(position1)
				positions2.append(pos_after2)
				j += 1
				continue

		# removing last yielded no consequences keep in regular way
		if aligned_by[-1] == REMOVE_LAST:
			# try 3 distance
			if i + 2 < len(s1) and slen1 < slen2:
				if aligned_ends_together(s2[j], s1[i], reg2, two_after1, addition=s1[i + 1] + s1[i + 2], force=force):
					aligned_by.append(FIRST_LONGER_ALIGNED)
					aligned_by.append(FIRST_LONGER_ALIGNED)
					positions1.append(pos_2after1)
					positions2.append(position2)
					i += 2
					continue
			if j + 2 < len(s2) and slen2 < slen1:
				if aligned_ends_together(s1[i], s2[j], reg1, two_after2, addition=s2[j + 1] + s2[j + 2], force=force):
					aligned_by.append(SECOND_LONGER_ALIGNED)
					aligned_by.append(SECOND_LONGER_ALIGNED)
					positions1.append(position1)
					positions2.append(pos_2after2)
					j += 2
					continue
			# fallback was unnecesary
			positions1.append(removed_pos1)
			positions2.append(removed_pos2)
			i += 2
			j += 2

		# check if a , was replaced by a sentence ender
		if positions1 and slen2 < slen1:
			splitter = reg2 + ","
			comma_index = s1[i].find(splitter)
			if comma_index == -1:
				splitter = reg2 + " ,"
				comma_index = s1[i].find(splitter)
			if comma_index != -1:
				comma_index += len(splitter)
				aligned_by.append(COMMA_REPLACE_SECOND)
				positions1.append(positions1[-1] + comma_index)
				positions2.append(position2)
				s1 = s1[:i] + [s1[i][:comma_index], s1[i][comma_index:]] + s1[i+1:]
				endings1 = endings1[:i] + [endings1[i-1] + comma_index] + endings1[i:]
				continue
		if positions2 and slen1 < slen2:
			splitter = reg1 + ","
			comma_index = s2[j].find(splitter)
			if comma_index == -1:
				splitter = reg1 + " ,"
				comma_index = s2[j].find(splitter)
			if comma_index != -1:
				comma_index += len(splitter)
				aligned_by.append(COMMA_REPLACE_FIRST)
				positions2.append(positions2[-1] + comma_index)
				positions1.append(position1)
				s2 = s2[:j] + [s2[j][:comma_index], s2[j][comma_index:]] + s2[j+1:]
				endings2 = endings2[:j] + [endings2[j-1] + comma_index] + endings2[j:]
				continue

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


def compare_paragraphs(origin, corrected, break_sent1=sent_tokenize_default, break_sent2=sent_tokenize_default):
	""" compares two paragraphs
		return:
		broken - the sentence endings indexes
		differences - difference measures corresponding to the indexes in broken
		aligned_by - the way the sentences were aligned"""
	print("comparing paragraphs")
	align_sentence_words(None,None,None,True)
	print("aligning sentences")
	broken = [None,None]
	broken[0], broken[1], aligned_by = break2common_sentences(origin, corrected, break_sent1, break_sent2)
	print("assesing differences")
	origin_sentences = list(get_sentences_from_endings(origin, broken[0]))
	corrected_sentences = list(get_sentences_from_endings(corrected, broken[1]))
	# print(corrected_sentences)
	index_differences = [index_diff(orig, cor) for orig, cor in zip(origin_sentences, corrected_sentences)]
	spearman_differences = [spearman_diff(orig, cor)[0] for orig, cor in zip(origin_sentences, corrected_sentences)]
	word_differences = [word_diff(orig, cor) for orig, cor in zip(origin_sentences, corrected_sentences)]
	print("comparing done printing interesting results")
	for i, dif in enumerate(word_differences):
		if dif > 10: # or i < 3 # use i to print some, use diff to print all sentences which differ ion more than "diff" words from each other
			print("-------\nsentences:\n", corrected_sentences[i],"\norignal:\n", origin_sentences[i])
			print ("word dif:", dif)
			print("match num:", i)
	for i, dif in enumerate(index_differences):
		if dif > 10: # or i < 3 # use i to print some, use diff to print all sentences which differ ion more than "diff" words from each other
			print("-------\nsentences:\n", corrected_sentences[i],"\norignal:\n", origin_sentences[i])
			print ("word dif:", dif)
			print("match num:", i)
	return broken, word_differences, index_differences, spearman_differences, aligned_by


def read_paragraph(filename, process=preprocess_paragraph):
	with open(PATH + filename) as fl:
		return process("".join(islice(fl, MAX_SENTENCES)))


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


def create_hist(l, top=30, bottom=0):
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


def plot_differences_hist(l, ax, pivot, diff_type, bottom):
	width = 1/len(l)
	name = -1
	for i, tple in enumerate(l):
		y = create_hist(tple[pivot], bottom=bottom)
		x = np.array(range(len(y)))
		print(diff_type + " hist results ",tple[name],":",y)
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=tple[name])
	plt.autoscale(enable=True, axis='x', tight=False)
	plt.ylabel("amount")
	ply.xlim(xmin=0)
	plt.xlabel("number of " + diff_type + " changed")
	plt.title("number of " + diff_type + " changed by method of correction")
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()


def plot_words_differences_hist(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the hists"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	plot_differences_hist(l, ax, words_differences, "words",0)


def plot_index_differences_hist(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the hists"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	plot_differences_hist(l, ax, index_differences, "index",1)


def plot_spearman_differences(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the hists"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	boxplot_differences(l, ax, spearman_differences, "spearman", 1)


def plot_spearman_ecdf(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the hists"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	plot_ecdf(l, ax, spearman_differences, "spearman", 0.7, 1)


def plot_ecdf(l, ax, pivot, diff_type, bottom, top):
	ys = []
	name = -1
	colors = rainbow_colors(range(len(l)))
	for i, tple in enumerate(l):
		x = np.sort(tple[pivot])
		x = [point for point in x if point < top and point >= bottom]
		yvals = np.arange(len(x))/float(len(x))
		ys.append((x, tple[name], colors[i]))
		ax.plot(x, yvals, color=colors[i], label=tple[name])
	plt.ylim(ymax=0.6)
	# for y, name, color in ys:
		# x = np.linspace(min(sample), max(sample))
		# y = ecdf(x)
		# ax.step(x, y, olor=color, label=name)
		# ax.boxplot(x, labels=names, showmeans=True)
	plt.ylabel("probabillity")
	plt.xlabel("number of " + diff_type + " changed")
	plt.title("empirical distribution of " + diff_type + " changes")
	plt.legend(loc=6, fontsize=10)

def boxplot_differences(l, ax, pivot, diff_type, bottom):
	# ys = []
	x = []
	names = []
	name = -1
	# max_len = 0
	colors = rainbow_colors(range(len(l)))

	for i, tple in enumerate(l):
		y = tple[pivot]
		x.append(y)
		names.append(tple[name])

	plt.autoscale(enable=True, axis='x', tight=False)
	ax.boxplot(x, labels=names, showmeans=True)
	plt.title("box plot of " + diff_type + " changes")
	plt.legend(loc=7, fontsize=10)


def plot_aligned_by(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot """
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	width = 1/len(l)
	for i, tple in enumerate(l):
		y = extract_aligned_by_dict(tple[aligned_by])
		y = [y[FIRST_LONGER] + y[COMMA_REPLACE_FIRST], y[ORDERED], y[SECOND_LONGER]+ y[COMMA_REPLACE_SECOND]]
		print("first ordered and second longer",tple[name],":",y)
		x = np.array(range(len(y)))
		colors = rainbow_colors(range(len(l)))
		ax.bar(x + i*width, y, width=width, color=colors[i], align='center', label=tple[name])
	ax.autoscale(tight=True)
	plt.ylabel("amount")
	plt.xlabel("number of sentence changes of that sort")
	plt.title("number of sentence changes by method of correction")
	plt.xticks(x + width, (FIRST_LONGER, ORDERED, SECOND_LONGER))
	plt.legend(loc=7, fontsize=10)
	# plt.tight_layout()

def plot_not_aligned(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the bars"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	width = 1/len(l)
	for i, tple in enumerate(l):
		y = extract_aligned_by_dict(tple[aligned_by])
		y = y = [y[FIRST_LONGER] + y[COMMA_REPLACE_FIRST], y[SECOND_LONGER] + y[COMMA_REPLACE_SECOND]]
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

def plot_words_differences(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the hists"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	plot_differences(l, ax, words_differences, "words", 1)


def plot_index_differences(l, ax):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the hists"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	plot_differences(l, ax, index_differences, "index", 1)


def plot_differences(l, ax, pivot, diff_type, bottom):
	""" gets a list of (broken, words_differences, index_differences, spearman_differences, aligned_by, name) tuples and plot the plots"""
	broken, words_differences, index_differences, spearman_differences, aligned_by, name = list(range(6)) # tuple structure
	ys = []
	max_len = 0
	colors = rainbow_colors(range(len(l)))

	for i, tple in enumerate(l):
		y = create_hist(tple[pivot], bottom=bottom)
		ys.append((y, tple[name], colors[i]))
		max_len = max(max_len, len(y))
	
	x = np.array(range(bottom, max_len+bottom))

	for y, name, color in ys:
		y = y + [0]*(max_len-len(y))
		ax.plot(x, np.cumsum(y), color=color, label=name)
	plt.autoscale(enable=True, axis='x', tight=False)
	plt.ylabel("amount")
	plt.xlabel("number of " + diff_type + " changed")
	plt.title("accumulative number of sentences by " + diff_type + " changed per")
	plt.legend(loc=7, fontsize=10)


def plot_comparison(l):
	"""gets a list of tuple parameters and plots them"""
	data = []
	ax = plt.subplot(221)
	plot_spearman_differences(l, ax)
	ax = plt.subplot(222)
	plot_spearman_ecdf(l, ax)
	ax = plt.subplot(223)
	plot_aligned_by(l, ax)
	ax = plt.subplot(224)
	plot_not_aligned(l, ax)
	plt.show()
	plt.clf()

	data = []
	dirname = "./plots/"
	ax = plt.subplot(111)
	plot_spearman_differences(l, ax)
	plt.savefig(dirname + r"spearman_differences" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_spearman_ecdf(l, ax)
	plt.savefig(dirname + r"spearman_ecdf" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_words_differences(l, ax)
	plt.savefig(dirname + r"words_differences" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_words_differences_hist(l, ax)
	plt.savefig(dirname + r"words_differences_hist" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_index_differences(l, ax)
	plt.savefig(dirname + r"index_differences" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_index_differences_hist(l, ax)
	plt.savefig(dirname + r"index_differences_hist" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_aligned_by(l, ax)
	plt.savefig(dirname + r"aligned_all" + trial_name + ".png", bbox_inches='tight')
	plt.clf()
	ax = plt.subplot(111)
	plot_not_aligned(l, ax)
	plt.savefig(dirname + r"aligned" + trial_name + ".png", bbox_inches='tight')


###########################################################
####                        UTIL                        ###
###########################################################
def convert_file_to_csv(filename):
	l = read(filename)
	filename = os.path.splitext(filename)[0]+".csv"
	col_names = ["words_differences", "index_differences", "spearman_differences", "aligned_by"]
	names = l.keys()
	names_row = []
	spacing_left = int(len(col_names)/2)*[""]
	spacing_right = (int((len(col_names) + 1)/2) - 1)*[""]
	for name in names:
		names_row += spacing_left + [name] + spacing_right
	col_names = col_names*len(names_row)
	max_len = 0
	for value in l.values():
		for lst in value:
			lst = lst[1:] # remove sentence breaks
			max_len = max(max_len, len(lst))
	with open(filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(names_row)
		writer.writerow(col_names)
		for i in range(max_len):
			row = []
			for value in l.values():
				value = value[1:]
				for lst in value:
					if len(lst) > i:
						row.append(lst[i])
					else:
						row.append("")
			writer.writerow(row)

	    
def read(filename):
	try:
		with open(filename, "r+") as fl:
			return json.load(fl)
	except FileNotFoundError as e:
		print("file not found:", e)
		return dict()
	except json.decoder.JSONDecodeError as e:
		print("json decoder error:", e)
		return dict()


def dump(l, filename):
	out = read(filename)
	for obj in l:
		name = obj[-1]
		obj = obj[:-1]
		if name not in out:
			print(name, " name")
			out[name] = obj
	with open(filename, "w+") as fl:
		json.dump(out, fl)


if __name__ == '__main__':
	main()