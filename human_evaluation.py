import re
import pandas as pd
import numpy as np

SYSTEM1RANK = "system1rank"
SENTENCE_ID = "segmentId"
SYSTEM1NAME = "system1Id"
ANNOTATOR = "judgeID"

ASSESS_DIR = '/home/borgr/ucca/assess_learner_language/'
# ASSESS_DIR = '/cs/labs/oabend/borgr/assess_learner_language/'
DATA_DIR = ASSESS_DIR + "data/"
HUMAN_JUDGMENTS_DIR = DATA_DIR + "human_judgements/"
PARAGRAPHS_DIR = DATA_DIR + "paragraphs/"


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
		return text[lines]

def main():
	# parse xmls
	judgments_file = HUMAN_JUDGMENTS_DIR + "all_judgments.xml" 
	judgments_file = HUMAN_JUDGMENTS_DIR + "8judgments.xml" 
	db = parse_xml(judgments_file)

	# read relevant lines note that id135 = line 136
	learner_file = PARAGRAPHS_DIR + "conll.tok.orig"
	lines = get_lines_from_file(learner_file, db[SENTENCE_ID].map(lambda x: int(x)))
	print(lines)
	print(make sure it returns the right lines!!!!)

	# rank outputs on each sentence that has human judgment using ucca and lt (and glue and m2 later)

	# think what to calculate with it (percentage of agreeing comparisons \ truekill \ expected wins both found in Human Evaluation of Grammatical Error Correction Systems)

	# rank using GLUE?


if __name__ == '__main__':
	main()