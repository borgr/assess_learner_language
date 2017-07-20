import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import annalyze_crowdsourcing as an
import create_confirmation_batch as ccb

CONFIRMATION_BATCH_RESULTS = an.CORRECTIONS_DIR + "confirmation_batches/" + "confirmation_batch.csv" 
CORRECTION_COL = "Input.Correction"
ORIGINAL_COL = "Input.Original"
ANSWER_COL = "Answer.Q1Answer"
ANSWER_FREQ_COL = "num"

CONFIRMED = "Yes"

def main():
	save_name = "confirmation_frequency.png"
	save_dir = an.PLOTS_DIR
	# save_name = ""
	show = True
	freq_db = pd.read_csv(ccb.CONFIRMATION_BATCH_SOURCE)
	res_db = pd.read_csv(CONFIRMATION_BATCH_RESULTS)
	freq_db.loc[:,ccb.FREQ_CORRECTION_COL] = freq_db[ccb.FREQ_CORRECTION_COL].apply(an.normalize_sentence)
	freq_db.loc[:,ccb.FREQ_LEARNER_COL] = freq_db[ccb.FREQ_LEARNER_COL].apply(an.normalize_sentence)
	res_db.loc[:,CORRECTION_COL] = res_db[CORRECTION_COL].apply(an.normalize_sentence)
	confirmations_count = res_db.groupby([CORRECTION_COL, ANSWER_COL]).size().reset_index(name=ANSWER_FREQ_COL)
	# print(res_db.groupby([CORRECTION_COL, ANSWER_COL]).size())
	# print(res_db.groupby(["WorkerId", ANSWER_COL]).size().reset_index(name=ANSWER_FREQ_COL).sort_values([ANSWER_COL, ANSWER_FREQ_COL], ascending=[True,False]))

	x = []
	y = []
	for i in range(len(freq_db[ccb.FREQ_CORRECTION_COL])):
		sentence = freq_db[ccb.FREQ_CORRECTION_COL].iloc[i]
		# if freq_db[ccb.FREQ_LEARNER_COL].iloc[i] not in freq_db[ccb.FREQ_CORRECTION_COL].iloc[i]:
		x.append(freq_db[ccb.CORRECTION_FREQUENCY_COL].iloc[i])
		num_no = confirmations_count[ANSWER_FREQ_COL][(confirmations_count[CORRECTION_COL] == sentence) & 
									 (confirmations_count[ANSWER_COL] != CONFIRMED)]
		y.append(0 if num_no.size == 0 else int(num_no)/3)
		# else:
		# 	print(sentence)
		# print(sentence, confirmations_count[(confirmations_count[CORRECTION_COL] == sentence) & 
		# 							 (confirmations_count[ANSWER_COL] != CONFIRMED)].size + confirmations_count[(confirmations_count[CORRECTION_COL] == sentence) & 
		# 							 (confirmations_count[ANSWER_COL] == CONFIRMED)].size )
		assert(confirmations_count[(confirmations_count[CORRECTION_COL] == sentence) & 
									 (confirmations_count[ANSWER_COL] != CONFIRMED)].size + confirmations_count[(confirmations_count[CORRECTION_COL] == sentence) & 
									 (confirmations_count[ANSWER_COL] == CONFIRMED)].size >= 3)
	for mean_by_corrections_rejected in [True, False]: 
		y_mean_workers_rejcted = []
		y_mean_corrections_rejcted = []
		y_mean = []
		x_4means = []
		y = np.array(y)
		x = np.array(x)
		for i in np.unique(x):
			if i < 10:
				x_4means.append(i)
				y_mean_workers_rejcted.append(np.mean(1 - y[np.where(x == i)])) 
				y_mean_corrections_rejcted.append(np.mean(y[np.where(x == i)] > 1/3))
		if mean_by_corrections_rejected:
			y_mean = y_mean_corrections_rejcted if mean_by_corrections_rejected else y_mean_workers_rejcted
			y_mean.append(np.mean(y[np.where(x > 10)] > 1))
		else:
			y_mean = y_mean_workers_rejcted
			y_mean.append(np.mean(1 - y[np.where(x > 10)]))
		ticks = [str(i) for i in x_4means] + ["10+"]
		x_4means.append(10)
		x_4means = np.array(x_4means)
		plt.xticks(x_4means, ticks)
		print(x_4means,y_mean)
		plt.scatter(np.unique(x_4means), y_mean)
		# fit with np.polyfit
		m, b = np.polyfit(x_4means, y_mean, 1)
		print(m,b)
		plt.plot(x_4means, m*x_4means + b, '-', label="Regression Line")
		plt.ylabel("Percentage of rejected corrections" if mean_by_corrections_rejected else "Mean validity")
		plt.xlabel("Correction Frequency")
		plt.ylim(ymax=1, ymin=0)
		plt.legend(loc=7, fontsize=10, fancybox=True, shadow=True)
		mean_prefix = "rejected_" if mean_by_corrections_rejected else "IAA_"
		if save_name:
			plt.savefig(save_dir + mean_prefix + save_name, bbox_inches='tight')
		if show:
			plt.show()
		plt.cla()


if __name__ == '__main__':
	main()