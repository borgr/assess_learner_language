import os
import re
import pandas as pd
import numpy as np
from rank import sentence_m2, grammaticality_score, semantics_score, glue_score, ucca_parse_files
from m2scorer import m2scorer

SYSTEM1RANK = "system1rank"
SENTENCE_ID = "segmentId"
SYSTEM1NAME = "system1Id"
ANNOTATOR = "judgeID"

ASSESS_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
# ASSESS_DIR = '/home/borgr/ucca/assess_learner_language/'
# ASSESS_DIR = '/cs/labs/oabend/borgr/assess_learner_language/'
DATA_DIR = ASSESS_DIR + "data/"
HUMAN_JUDGMENTS_DIR = DATA_DIR + "human_judgements/"
PARAGRAPHS_DIR = DATA_DIR + "paragraphs/"
REFERENCE_DIR = DATA_DIR + "references/"
cALCULATIONS_DIR = "calculations_data/uccasim_rerank/"
PARSE_DIR = cALCULATIONS_DIR + "/ucca_parse/"


def parse_xml(judgments_file):
	judgments = []
	with open(judgments_file) as fl:
		for line in fl:
			if "<ranking-item" in line:
				match = re.search('src-id="(\d*)"', line)
				if match:
					sent_id = match.group(1)
				else:
					match = re.search('id="(\d*)"', line)
					sent_id = match.group(1)
				annotator = re.search('user="([\w\-_]*)"', line).group(1)
			if "translation" in line:
				system_rank = re.search('rank="(\d*)"', line).group(1)
				system_names = re.search('system="([\w\s,_-]*)"', line).group(1)
				for system_name in re.split("\W", system_names):
					judgments.append((sent_id, system_rank, system_name, annotator))

	db = pd.DataFrame(judgments, columns=[SENTENCE_ID, SYSTEM1RANK, SYSTEM1NAME, ANNOTATOR])
	return db


def get_lines_from_file(file, lines):
	lines = np.array(lines)
	with open(file) as fl:
		text = np.array(fl.readlines())
		return (line.replace("\n","") for line in text[lines])


def get_edits_from_file(file, lines):
	lines = np.array(lines)
	source_sentences, gold_edits = m2scorer.load_annotation(file)
	return np.array(gold_edits)[lines]


def main():
	# parse xmls
	judgments_file = HUMAN_JUDGMENTS_DIR + "all_judgments.xml" 
	judgments_file = HUMAN_JUDGMENTS_DIR + "8judgments.xml" 
	db = parse_xml(judgments_file)
	print("number of judgments:", len(db.index))
	sentence_ids = db[SENTENCE_ID].map(lambda x: int(x))
	# read relevant lines note that id135 = line 136
	learner_file = PARAGRAPHS_DIR + "conll.tok.orig"
	origin_lines = list(get_lines_from_file(learner_file, sentence_ids))
	first_nucle =  REFERENCE_DIR + "NUCLEA"
	second_nucle =  REFERENCE_DIR + "NUCLEB"
	# combined_nucle = REFERENCE_DIR + "NUCLE.m2"
	# BN = REFERENCE_DIR + "BN.m2"
	# ALL =  REFERENCE_DIR + "ALL.m2"
	references_files = [first_nucle]
	edits_files = [first_nucle + ".m2"]

	references_edits = list(zip(*[list(get_edits_from_file(fl, sentence_ids)) for fl in edits_files]))
	references_lines = list(zip(*[list(get_lines_from_file(fl, sentence_ids)) for fl in references_files]))

	# print("are the edits good?", references_edits[-10], origin_lines[-10], references_lines[-10])
	JMGR_file = PARAGRAPHS_DIR + "JMGR"
	amu_file = PARAGRAPHS_DIR + "AMU"
	cuui_file = PARAGRAPHS_DIR + "CUUI"
	iitb_file = PARAGRAPHS_DIR + "IITB"
	ipn_file = PARAGRAPHS_DIR + "IPN"
	nthu_file = PARAGRAPHS_DIR + "NTHU"
	pku_file = PARAGRAPHS_DIR + "PKU"
	post_file = PARAGRAPHS_DIR + "POST"
	rac_file = PARAGRAPHS_DIR + "RAC"
	sjtu_file = PARAGRAPHS_DIR + "SJTU"
	ufc_file = PARAGRAPHS_DIR + "UFC"
	umc_file = PARAGRAPHS_DIR + "UMC"
	camb_file = PARAGRAPHS_DIR + "CAMB"
	system_files = [JMGR_file, amu_file, cuui_file, iitb_file,
				    ipn_file, nthu_file, pku_file, post_file,
				    rac_file, sjtu_file, ufc_file, umc_file, camb_file, second_nucle]
	# ucca_parse_files(system_files + references_files, PARSE_DIR)
	print("enable other similarities")
	score_db = []
	system_sentences_calculated = set()
	for system_file in system_files:
		system_lines = get_lines_from_file(system_file, sentence_ids)
		for source, references, edits, system in zip(origin_lines, references_lines, references_edits, system_lines):
			if system not in system_sentences_calculated:
				glue = 0
				# glue = glue_score(source, references, system)
				grammar = 0
				# grammar = grammaticality_score(source, system, PARSE_DIR)
				uccaSim = 0
				# uccaSim = semantics_score(source, system, PARSE_DIR)
				edits = list(edits)
				m2 = m2scorer.get_score([system], [source], edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)
				score_db.append((m2, glue, grammar, uccaSim, system, source))
				system_sentences_calculated.add(system)

			if len(score_db) % 100 == 99:
				print(len(score_db))
	score_db = pd.DataFrame(score_db, columns=["m2", "glue", "grammar", "uccaSim", "sentence", "source"])
	print(score_db)
	# rank outputs on each sentence that has human judgment using ucca and lt (and glue and m2 later)

	# think what to calculate with it (percentage of agreeing comparisons \ truekill \ expected wins both found in Human Evaluation of Grammatical Error Correction Systems)

	# rank using GLUE?


if __name__ == '__main__':
	main()