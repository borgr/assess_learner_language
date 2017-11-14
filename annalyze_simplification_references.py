import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import annalyze_crowdsourcing as an
import create_confirmation_batch as ccb
import os

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep + "/simplification/data/"

DIFFERENT_REFS = "ref_num"
ORIGIN = "origin"


def extract_short(db, max_len):
	length = "len"
	db[length] = db[ORIGIN].apply(lambda x: x.count(" "))
	short_sentences = db[ORIGIN][db[length] <= max_len]
	choice = np.random.randint(0,len(short_sentences.values), 50)
	print(short_sentences.values[choice])
	# short_sentences = db.iloc[:,-2][db[length] <= max_len]
	# print(short_sentences.values[choice])
	

	lengths = db[ORIGIN].apply(lambda x: x.count(" ")).value_counts().sort_index()
	# lengths = lengths[lengths.index.values <= max_len]
	lengths = lengths[lengths.index.values]
	lengths.plot.bar()
	plt.show()


def remove_non_simplified(db):
	keep = []
	for i,row in db.iterrows():
		keep.append(not row[ORIGIN] in row[-8:].values)
	keep = np.array(keep)
	db = db.loc[keep,:]
	return db

def main():
	db = an.read_simplification()

	db = remove_non_simplified(db)

	# references = db.iloc[:, -8:].values

	extract_short(db, 15)

	db[DIFFERENT_REFS] = db.iloc[:,-8:].apply(lambda row: len(row.unique()), axis=1)
	# print(db.ix[:,DIFFERENT_REFS].value_counts())
	# print(db[db[DIFFERENT_REFS] == 3])


if __name__ == '__main__':
	main()