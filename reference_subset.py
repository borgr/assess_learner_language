import os
import numpy as np
# in all.m2 file:
# NUCLEA - 14
# NUCLEB - 15
# BN1-10 - 0-9
all_references = list(range(10))+[14, 15]
subset_sizes = range(3, len(all_references))

input_dir = r"data/references/"
input_file = r"ALL.m2"
output_dir = input_dir
output_file = r"subset.m2"
def main():
	# read all references
	with open(input_dir + input_file) as fl:
		lines = fl.readlines()

	# sample
	for i in subset_sizes:
		references = list(np.random.choice(all_references, i, False))
		references.sort()
		mapping = dict(((x, i) for i, x in enumerate(references)))
		# print(mapping)
		res = []
		for line in lines:
			if not line.startswith("A"):
				res.append(line)
			else:
				# assumes no more than 100 references exists otherwise need regex |\s*(d+)\s*\n
				# print(line)
				if "|" == line[-3]:
					origin_ref_num = int(line[-2])
					# print(line[:-1])
					line = line[:-2]
				else:
					origin_ref_num = int(line[-3:-1])
					line = line[:-3]
				if origin_ref_num in mapping:
					line += str(mapping[origin_ref_num]) + "\n"
					res.append(line)
					# print(line)
					# return
				

		#assumes each subset size exists only once
		with open(output_dir + str(i) + output_file, "w+") as fl:
			fl.writelines(res)

def basename(name):
	return name.split("\\")[-1].split("/")[-1]

def name_extension(name):
	return basename(name).split(".")

if __name__ == '__main__':
	main()