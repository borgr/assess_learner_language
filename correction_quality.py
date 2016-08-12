from itertools import accumulate
import math

from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.stem import WordNetLemmatizer
import sys
sys.path.append('/home/borgr/ucca/ucca/scripts/distances')
sys.path.append('/home/borgr/ucca/ucca/ucca')
sys.path.append('/home/borgr/ucca/ucca')
import align
import distance
from munkres import Munkres, print_matrix
import re
lemmatizer = WordNetLemmatizer()
sentence_ends_with_no_space_pattern = re.compile("(.*?\w\w\.)(\w+[^\.].*)")
space_before_sentence_pattern = re.compile("(.*?\s\.(\s*\")?)(.*)")

MAX_DIST = 2
PATH = r"/home/borgr/ucca/data/paragraphs/"

print("bugs are still seen when running on gold, autocorrect shouldn't make such errors")
print("remmember to clean prints after fixing bugs and before committing")
print("clean all TODO")

def is_word(w):
	return True if w != align.EMPTY_WORD and re.search('[a-zA-Z]', w) else False

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
	tokens = split_by_pattern(tokens, sentence_ends_with_no_space_pattern)
	tokens = split_by_pattern(tokens, space_before_sentence_pattern, 1, 3)
	return tokens


def preprocess_paragraph(p):
	"""preprocesses a paragraph"""
	p = re.sub("\s+", " ", p)
	p = re.sub("(\.\s+)\.", "\1", p)
	return p


def preprocess_word(w):
	if not w[-1].isalnum():
		w = w[:-1]
	return align.preprocess_word(w)


def approximately_same_word(w1, w2):
	""" returns if both words are considered the same word with a small fix or not"""
	l1 = lemmatizer.lemmatize(w1)
	l2 = lemmatizer.lemmatize(w2)
	if (distance.levenshtein(l1, l2) > MAX_DIST or
		w1 == align.EMPTY_WORD or w2 == align.EMPTY_WORD):
		#TODO should "the" "a" etc be considered in a different way? maybe they should not but not in this function
		return False #TODO words such as in at on etc, might be considered all equal to each other and to the empty_word for our purpose
	return True


def _choose_ending_position(sentences, endings, i):
	""" i - sentence number
		sentences - list of sentences
		endings - list of sentences positions endings

		return position, last word in the i'th sentence"""
	for word in reversed(sentences[i].split()): # maybe word tokenizer is better? for things like face-to-face and face to face? or is splitting by - too will suffice?
		word = preprocess_word(word)
		if len(word) > 1:
			return endings[i], word
	print ("sentence contains no words:\n\"", sentences[i], "\"")
	print("sentence before", sentences[i-1])
	print("sentence after", sentences[i+1])
	raise "should not happen"
	return endings[i], preprocess_word(sentences[i].split()[-1])


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

def aligned_ends_together(sentence1, sentence2, reg1, reg2):
	""" checks if two sentences, ending in two regularized words ends at the same place.
	"""
	aligned, indexes = align.align(sentence1, sentence2, True)
	mapping = dict(map(lambda x:(preprocess_word(x[0]), preprocess_word(x[1])), aligned))
	rev = dict(align.reverse_mapping(aligned))
	# print(rev)
	# print(reg2)
	# print(aligned)
	if (reg1 in mapping and mapping[reg1] == align.EMPTY_WORD):
		# print(reg2, rev[reg2])
		if approximately_same_word(reg2, rev[reg2]):
			return True
	if (reg2 in rev and rev[reg2] == align.EMPTY_WORD):
		# print(reg1, mapping[reg1])
		if approximately_same_word(reg1, mapping[reg1]):
			return True
	return False

def break2common_sentences(p1, p2):
	"""finds the positions of the common sentence ending

	Breaking is done according to the text of both passages
	returns two lists each containing positions of sentence endings
	guarentees same number of positions is acquired and the last position is the passage end"""
	
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
			j +=1
			inc = False
			continue

		inc = True
		position1, reg1 = _choose_ending_position(s1, endings1, i)
		position2, reg2 = _choose_ending_position(s2, endings2, j) # the problem arises because I do not use alignment, what if the last word was deleted or added? (also aligning very different sentences and checking wheter the last word is aligned to approximately the same word is fallable (they are aligned by how similar they are, not place in sentence))
		if approximately_same_word(reg1, reg2):
			print("ordered ",i)
			positions1.append(position1)
			positions2.append(position2)
			continue

		#deal with addition or subtraction of a sentence ending
		slen1 = len(s1[i].split())
		slen2 = len(s2[j].split())
		# print(slen1," ", slen2)
		if i + 1 < len(s1) and slen1 < slen2:
			pos_after1, one_after1 = _choose_ending_position(s1, endings1, i + 1)
			if approximately_same_word(one_after1, reg2):
				print("first longer ",i)
				positions1.append(pos_after1)
				positions2.append(position2)
				i += 1
				continue
		
		if j + 1 < len(s2) and slen2 < slen1:
			pos_after2, one_after2 = _choose_ending_position(s2, endings2, j + 1)
			if approximately_same_word(reg1, one_after2):
				print("second longer ", i)
				positions1.append(position1)
				positions2.append(pos_after2)
				j += 1
				continue
		# print("not missing a sentence")
		# no alignment found with 2 sentences
		# check if a word was added to the end of one of the sentences
		if aligned_ends_together(s1[i], s2[j], reg1, reg2):
				positions1.append(position1)
				positions2.append(position2)
				continue

		# check if a word was added to the end of one of the sentences
		# Also, deal with addition or subtraction of a sentence ending
		if i + 1 < len(s1) and slen1 < slen2:
			if aligned_ends_together(s1[i] + s1[i + 1], s2[j], one_after1, reg2):
					positions1.append(pos_after1)
					positions2.append(position2)
					i += 1
					continue

		if j + 1 < len(s2) and slen2 < slen1:
			if aligned_ends_together(s1[i], s2[j]  + s2[j + 1], reg1, one_after2):
					positions1.append(position1)
					positions2.append(pos_after2)
					j += 1
					continue
		if i>339:
			print("s1:",s1[i])
			print("s2:",s2[j])
			print("s1af:",s1[i+1])
			print("s2af:",s2[j+1])

		print (i, reg1, reg2, one_after1, one_after2)
		print("------------------")
		# print("s1:",s1[i])
		# print("s2:",s2[j])
		# print("s1af:",s1[i+1])
		# print("s2af:",s2[j+1])

	# add last sentence in case skipped
		position1, reg1 = _choose_ending_position(s1, endings1, -1)
		position2, reg2 = _choose_ending_position(s2, endings2, -1)
	if (not positions1) or (not positions2) or (
	    positions1[-1] != position1 and positions2[-1] != position2):
		positions1.append(endings1[-1])
		positions2.append(endings2[-1])
	elif positions1[-1] != position1 and positions2[-1] == position2:
		positions1[-1] = endings1[-1]
	elif positions1[-1] == position1 and positions2[-1] != position2:
		positions2[-1] = endings2[-1]

	return positions1, positions2


def get_sentences_from_endings(paragraph, endings):
	"""a generator of sentences from a paragraph and ending positions in it"""
	last = 0
	for cur in endings:
		yield paragraph[last:cur]
		last = cur


def compare_paragraphs(origin, corrected):
	""" compares two paragraphs, returns the sentence endings and their corresponding difference measures"""
	print("comparing paragraphs")
	print("aligning sentences")
	broken = break2common_sentences(origin, corrected)
	# print(list(get_sentences_from_endings(origin, broken[0]))[:5])
	# print(list(get_sentences_from_endings(corrected, broken[1]))[-10:])
	print("assesing differences")
	origin_sentences = get_sentences_from_endings(origin, broken[0])
	corrected_sentences = get_sentences_from_endings(corrected, broken[1])
	differences = [word_diff(orig,cor) for orig, cor in zip(origin_sentences,corrected_sentences)]
	return broken, differences


def read_paragraph(filename):
	with open(PATH + filename) as fl:
		return preprocess_paragraph(fl.read())


if __name__ == '__main__':
	# origin = """genetic risk refers more to your chance of inheriting a disorder or disease . 
	# how much a genetic change tells us about your chances of developing a disorder is not always clelar . 
	# when we are diagnosed out with a certain genetic diseases , are we supposed to disclose this result to our relatives ? 
	# on one hand , we do not want this potential danger causing frightening affect our families ' later lives . 
	# when people around us know that we got certain disease , their altitudes will be easily changed , whether caring us too much or keeping away from us . 
	# surrounded by such concerns , it is very likely that we are distracted to worry about these problems . 
	# it is a concern that will be with us during our whole life , because we will never know when the ''potential bomb' ' will explode . """
	# autocorrect = """genetic risk refers to your chance of inheriting a disorder or disase.
	# how much a genetic change tells us about your chance of developing a disorder is not always clear.
	# when we are diagnosed with certain genetic diseases, are we suppose to disclose this result to our relatives?
	# on one hand, we do not want this potential danger having frightening effects in our families' later lives.
	# when people around us know that we have certain diseases, their attitude will easily change, whether caring for us too much or keeping away from us.
	# surrounded by such concerns, it is very likely that we are distracted and worry about these problems.
	# it is a concern that will be with us during our whole life, because we never know when the ''potential bomb'' will explode."""
	# origin = preprocess_paragraph(origin)
	# autocorrect = preprocess_paragraph(autocorrect)
	# a = break2common_sentences(origin, autocorrect)

	# print(diff_words("""it is common that almost everyone in a public transport is watching phone or wearing a pair ear piece in this modern society today .  ""","""It is a common scene that almost everyone in a public transport is watching at phone or wearing a pair ear piece in this modern society today ."""))
	# raise

	ACL2016RozovskayaRothOutput_file = "conll14st.output.1cleaned"
	learner_file = "conll.tok.orig"
	gold_file = "corrected_official-2014.0.txt.comparable"

	autocorrect = read_paragraph(ACL2016RozovskayaRothOutput_file)
	origin = read_paragraph(learner_file)
	gold = read_paragraph(gold_file)
	# # compare origin to autocorrect
	# broken, differences = compare_paragraphs(origin, autocorrect)
	# autocorrect_sentences = list(get_sentences_from_endings(autocorrect, broken[1]))

	# compare gold to autocorrect
	broken, differences = compare_paragraphs(origin, gold)######################TODO not working, why?
	gold_sentences =  list(get_sentences_from_endings(gold, broken[1]))
	origin_sentences = list(get_sentences_from_endings(origin, broken[0]))
	for i, dif in enumerate(differences):
		if dif > 2: # or i < 3
			print("-------\nsentences:\n",autocorrect_sentences[i],"\n", origin_sentences[i])
			print ("dif:", dif)
			print("match num:", i)

	# print(list(get_sentences_from_endings(origin, broken[0]))[:5])
	# print(list(get_sentences_from_endings(autocorrect, broken[1]))[:5])