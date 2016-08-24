import re
from correction_quality import PATH
FCE_DATA_FILE = "en_esl-ud-all.conllu"
CORRECTED_FILE = "corrected." + FCE_DATA_FILE
LEARNER_FILE = "learner." + FCE_DATA_FILE

def strip(line):
	return line.replace("#SENT=","")

def to_learner(xml_text):
	sentence = re.sub("<c>.*?</c>", "", xml_text)
	sentence = re.sub("<.*?>", "", sentence)
	return sentence

def to_corrected(xml_text):
	sentence = re.sub("<i>.*?</i>", "", xml_text)
	sentence = re.sub("<.*?>", "", sentence)
	return sentence

def main():
	corrected = []
	learner = []
	with open(PATH + FCE_DATA_FILE) as fl:
		for line in fl:
			if line.startswith("#SENT="):
				stripped = strip(line)
				learner.append(to_learner(stripped))
				corrected.append(to_corrected(stripped))

	with open(PATH + CORRECTED_FILE,"w") as fl:
		fl.writelines(corrected)
	with open(PATH + LEARNER_FILE,"w") as fl:
		fl.writelines(learner)

if __name__ == '__main__':
	main()