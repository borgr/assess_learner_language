import os
import platform
import traceback
from multiprocessing import Pool
import csv
import sys
sys.path.append('/home/borgr/ucca/ucca/scripts')
import pickle
sys.path.append('/home/borgr/ucca/ucca/ucca')
sys.path.append('/home/borgr/ucca/ucca')
from ucca import convert
from ucca import textutil
from ucca.ioutil import file2passage
sys.path.append('/home/borgr/ucca/ucca/scripts/distances')
import align
from ucca import layer0, layer1
POOL_SIZE = 4
PATH = r"/home/borgr/ucca/assess_learner_language/data/xmls/"
trial_name = "parser_r2s"
UNCOMBINED_DIR = "uncombined/"
corrected_stamp = "_corrected"

filenames = []
parsed_paragraphs = [2, 3, 5, 6, 7, 8, 10]
passage_filenames = []
for x in parsed_paragraphs:
	passage_filenames.append(str(x))
	passage_filenames.append(str(x) + corrected_stamp)
sys.setrecursionlimit(10000000)

# # combined passage names
# passage_filenames = [x + ".xml" for x in passage_filenames]

# sentence splitted xmls
passage_filenames  = []
for root, dirs, files in os.walk(PATH + UNCOMBINED_DIR):
	for filename in files:
		if filename.endswith(".xml"):
			print(filename)
			if corrected_stamp not in filename:
				passage_filenames.append(UNCOMBINED_DIR + filename[:-7] + corrected_stamp + filename[-7:])
				passage_filenames.append(UNCOMBINED_DIR + filename)
print(passage_filenames)

# borgr = list(("tree1197", "tree1297", "tree1198", "tree1298", "tree1200", "tree1300", "tree1202", "tree1302")) # "tree1299",  "tree1301"
# amittaic = ["amittaic1197", "amittaic1297", "amittaic1200", "amittaic1300", "amittaic1198", "amittaic1298", "amittaic1205", "amittaic1305", "amittaic1203", "amittaic1303"] #, "amittaic1301"]
# filenames = borgr + amittaic #same annotators
# filenames = ["tree1197", "amittaic1297","amittaic1197", "tree1297", "tree1198", "amittaic1298", "amittaic1198", "tree1298",  "tree1200", "amittaic1300", "amittaic1200", "tree1300"] # different annotators
# filenames = ["amittaic1197", "tree1197", "amittaic1297", "tree1297", "amittaic1200", "tree1200", 
# 			 "amittaic1300", "tree1300", "amittaic1198", "tree1198", "amittaic1298", "tree1298",
# 			 "amittaic1301",  "tree1301"]#inter annotator
# filenames = ["tree1297", "amittaic1297", "tree1298", "amittaic1298", "amittaic1300", "tree1300", "amittaic1301",  "tree1301"]
# implemented = [align.fully_aligned_distance]
# print("should flatten centers?")

#used functions
funcs = [align.aligned_edit_distance, align.fully_aligned_distance, align.aligned_top_down_distance,
		 align.token_distance, 
		 lambda x,y :align.token_distance(x,y,align.top_down_align),
		 lambda x,y :align.token_distance(x,y,align.fully_align)]
complex_func = align.token_level_similarity


def test(func, p, maximum=1, sym=True):
	print("testing "+ str(func.__name__))
	passed = True
	# if func(p3, p2) > 0.5: # not very informative and hard computationally because of word alignment
	# 	print("random passages have high similarity")
	p1 = p[0]
	p2 = p[1]
	base = func(p1, p2)
	if sym and base != func(p2, p1):
		print("not symmetrical")
		passed = False
	if func(p1, p1) != maximum:
		print("passage does not have the maximum value with itself")
		passed = False
	if base != func(p1, p2):
		print("function not deterministic")
		passed = False
	print("passed" if passed else "failed")
	return passed

def main():
	print (align.align("what has is by the meaning of the word is", "what is the men for the wk is are be"))

	# read xml files
	print("reading db xmls")
	p = []
	for filename in filenames:
		with open(PATH + filename, "rb") as fl:
			p += pickle.load(fl)[0]
		print("read ",filename," it starts with ", tuple(term.text for term in textutil.extract_terminals(convert.from_site(p[-1]))[:6]))
	#convert xml to passages
	p = list(map(convert.from_site,p))

	print("reading passage xmls")
	# read passage files
	for filename in passage_filenames:
		print("reading" + filename)
		if os.path.isfile(PATH + os.path.splitext(filename)[0] + ".pkl"):
			with open(PATH + os.path.splitext(filename)[0] + ".pkl", "rb") as fl:
				p.append(pickle.load(fl))
		else:
			p.append(file2passage(PATH + filename))
			with open(PATH + os.path.splitext(filename)[0] + ".pkl", "wb") as fl:
				pickle.dump(p[-1], fl)
				print("dumping", PATH + os.path.splitext(filename)[0] + ".pkl")

	all_filenames = filenames + passage_filenames
	print("read ", all_filenames)
	word2word = align.align_yields(p[0], p[1])
	assert align.reverse_mapping(word2word) == align.align_yields(p[1], p[0]), "align_yields asymmetrical"

	# create symmilarity matrix
	sources = []
	goals = []
	names = []
	i = 0
	while i < len(p):
		names.append(all_filenames[i])
		sources.append(p[i])
		i += 1
		goals.append(p[i])
		i += 1
	print("multithreading")
	pool = Pool(POOL_SIZE)
	results = pool.starmap(distances, zip(sources, goals, names))
	print(results)
	pool.close()
	pool.join()
	sym_mat = []
	keys = []
	for row, key in results:
		keys.append(key)
		sym_mat.append(row)
	print("functions and matrix")
	print(funcs+keys)
	for item in sym_mat:
		print(item)
	print("overall token analysis")
	print(align.token_level_analysis(p))
	with open(trial_name + "output.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(sym_mat)
	return

def distances(p1, p2, name):
	try:
		print(p1, p2, name)
		res = [func(p1, p2) for func in funcs]
		dic = complex_func(p1, p2)
		keys = sorted(dic.keys())
		# print("funcs", funcs, file = sys.stderr)
		# print("complex and keys", dic, keys, file = sys.stderr)
		res.insert(0, name)
		for key in keys:
			res.append(dic[key])

		print(res)
		return res, keys
	except Exception as e:
		print("in", name)
		traceback.print_exc()
		raise e

	# # tests
	# print("ordered trees:\n")
	# print(align.aligned_edit_distance(p[0], p[1]))
	# raise
	# print(align.token_level_analysis(p))
	# # print(align.token_level_similarity(p[0], p[1]))
	# print("token compare checks")
	# print("with top down")
	# test(lambda x,y :align.token_distance(x,y,align.top_down_align), p)
	# print("with fully align")
	# test(lambda x,y :align.token_distance(x,y,align.fully_align), p)
	# print("with buttom up")
	# test(align.token_distance, p)
	# test(align.aligned_top_down_distance, p, sym=False)
	# test(align.fully_aligned_distance, p) # heavy computations

def anounce_finish():
	if sys.platform == "linux":
		if set(("debian", "Ubuntu")) & set(platform.linux_distribution()):
			subprocess.call(['speech-dispatcher'])        #start speech dispatcher
			subprocess.call(['spd-say', '"your process has finished"'])
		else:
			#perhaps works only in ubuntu?
			a = subprocess.Popen(('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 300, 2)).split())
	elif sys.platform == "darwin":
		subprocess.call('say "your program has finished"'.split())
	else:
		import winsound
		winsound.Beep(300,2)



if __name__ == '__main__':
	main()


# old code
# print("first calculation")
# print(align.fully_aligned_distance(p[0], p[1]))
# print("symmetric calculation, should be the same")
# print(align.fully_aligned_distance(p[1], p[0]))
# print("paragraph with itself should be 1")
# print (align.fully_aligned_distance(p[1], p[1]))
# print (align.fully_aligned_distance(p[0], p[0]))

# for func in implemented:
# 	print("testing" + str(func)) 
# 	if test(func):
# 		print(func, "passed")
# 	else:
# 		print(func, "failed")

# print(node1)
# print(node1.outgoing)
# print([node.children[0] for node in node1 if len(node.children) > 1])
# print(node1.terminals)
# print(node1.parents[0])
# print(node1.tag)
# print(node2)
# for node1, node2 in align.align_nodes(set(node1), set(node2), word2word).items():
# 	print("1",node1)
# 	print("2",node2)
# 	pass

# for (i,j),(k,l) in zip(align.align_nodes(set(node1), set(node2), word2word).items(), align.align_nodes(set(node2), set(node1), align.reverse_mapping(word2word)).items()):
# 	print(i)
# 	if i == l and k != j:
# 		print(k)
# 		print(j)

# print()
# print (align.align("what is by the meaning of the word is", "what is the men for the wk is are be"))
# align.fully_aligned_distance(p[0], p[1])
# print([terminal.text for terminal in p[0].layer(layer0.LAYER_ID).all])
# print ([p[1].layer('0').by_position(terminal).text for terminal in textutil.break2sentences(p[1])])
