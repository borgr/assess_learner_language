import csv
import sys
sys.path.append('/home/borgr/ucca/ucca/scripts')
import pickle
sys.path.append('/home/borgr/ucca/ucca/ucca')
sys.path.append('/home/borgr/ucca/ucca')
import convert
import textutil
sys.path.append('/home/borgr/ucca/ucca/scripts/distances')
import align
from ucca import layer0, layer1
PATH = r"/home/borgr/ucca/data/annotaions/"

borgr = list(("tree1197", "tree1297", "tree1198", "tree1298", "tree1200", "tree1300", "tree1202", "tree1302")) # "tree1299",  "tree1301"
amittaic = ["amittaic1197", "amittaic1297", "amittaic1200", "amittaic1300", "amittaic1198", "amittaic1298", "amittaic1205", "amittaic1305", "amittaic1203", "amittaic1303"] #, "amittaic1301"]
# filenames = borgr + amittaic
# filenames = ["tree1197", "amittaic1297","amittaic1197", "tree1297", "tree1198", "amittaic1298", "amittaic1198", "tree1298",  "tree1200", "amittaic1300", "amittaic1200", "tree1300"] # different annotators
filenames = ["amittaic1197", "tree1197", "amittaic1297", "tree1297", "tree1200", "amittaic1200",
			 "amittaic1300", "tree1300", "amittaic1198", "tree1198", "amittaic1298", "tree1298",
			 "amittaic1301",  "tree1301"]#inter annotator
implemented = [align.fully_aligned_distance]
print("should flatten centers?")

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

	# read files
	p = []
	for filename in filenames:
		with open(PATH + filename, "rb") as fl:
			p += pickle.load(fl)[0]
		print("read ",filename," it starts with ", tuple(term.text for term in textutil.extract_terminals(convert.from_site(p[-1]))[:6]))
	#convert xml to passages
	p = list(map(convert.from_site,p))
	word2word = align.align_yields(p[0], p[1])
	assert align.reverse_mapping(word2word) == align.align_yields(p[1], p[0]), "align_yields asymmetrical"

	# create symmilarity matrix
	funcs = [align.aligned_edit_distance, align.fully_aligned_distance, align.aligned_top_down_distance,
			 align.token_distance, 
			 lambda x,y :align.token_distance(x,y,align.top_down_align),
			 lambda x,y :align.token_distance(x,y,align.fully_align)]
	complex_func = align.token_level_similarity
	sym_mat = []
	i = 0
	while i < len(p):
		filename = filenames[i]
		first = p[i]
		i += 1
		second = p[i]
		i += 1
		sym_mat.append([func(first, second) for func in funcs])
		dic = complex_func(first, second)
		keys = sorted(dic.keys())
		for key in keys:
			sym_mat[-1].append(dic[key])
		sym_mat[-1].insert(0, filename)

		print(sym_mat[-1])
	print("functions and matrix")
	print(funcs+keys)
	for item in sym_mat:
		print(item)
	print("overall token analysis")
	print(align.token_level_analysis(p))
	with open("output.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(sym_mat)
	return
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
