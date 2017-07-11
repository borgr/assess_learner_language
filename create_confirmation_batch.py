import pandas as pd
import numpy as np
import annalyze_crowdsourcing as an

TRIAL = "no_originals_"
CORRECTION_FREQUENCY_COL = "correction_frequency"
CONFIRMATION_BATCH_SOURCE = an.DATA_DIR + TRIAL + "confirmation_batch_to_send.csv"
FREQ_LEARNER_COL = "Original"
FREQ_CORRECTION_COL = "Correction"

def main():
	db = get_db()
	freq_db = db[an.CORRECTED_SENTENCES_COL].value_counts()
	print("number of sentences found", len(freq_db))
	# print(type(freq_db))
	# freq_db.plot()
	# plt.show()
	freq_db = freq_db.to_frame(name=CORRECTION_FREQUENCY_COL)
	freq_db[FREQ_CORRECTION_COL] = freq_db.index
	freq_db.index = pd.Series(range(len(freq_db.index)))
	freq_db[FREQ_LEARNER_COL] = freq_db[FREQ_CORRECTION_COL]
	# freq_db = freq_db.merge(db)
	testing_sentence = freq_db[FREQ_CORRECTION_COL].iloc[10]
	testing_sentence = "People can promote their friendships and relationships through social media "
	print(testing_sentence)
	print( find_learner(testing_sentence))
	freq_db[FREQ_LEARNER_COL] = [find_learner(freq_db[FREQ_CORRECTION_COL].iloc[x]) for x in range(len(freq_db[FREQ_CORRECTION_COL]))]
	freq_db.to_csv(CONFIRMATION_BATCH_SOURCE)
	# db[CORRECTION_FREQUENCY_COL] = 

def find_learner(x):
	db = get_db()
	# print(db[db[an.CORRECTED_SENTENCES_COL] == x].index[0])
	return db[an.LEARNER_SENTENCES_COL][db[db[an.CORRECTED_SENTENCES_COL] == x].index[0]]

_db = None
def get_db():
	global _db
	if _db is None:
		_db = an.clean_data(an.read_batches())
		# remove sentences that were not changed
		_db = _db[_db[an.CORRECTED_SENTENCES_COL] != _db[an.LEARNER_SENTENCES_COL]]
		_db.index = pd.Series(range(len(_db.index)))
	return _db

if __name__ == '__main__':
	main()