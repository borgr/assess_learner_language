import sys
import os
sys.path.append(os.path.abspath("./m2scorer/scripts/"))

import subprocess
import scikits.bootstrap
from m2scorer.scripts import m2scorer
import numpy as np
from multiprocessing import Pool
from rank import SARI_score
import annalyze_crowdsourcing as an
POOL_SIZE = 4
ALTERNATIVE_GOLD_MS = an.ALTERNATIVE_GOLD_MS
# ALTERNATIVE_GOLD_MS = np.arange(10) + 1

def main():
    ACL2016RozovskayaRothOutput_file = "conll14st.output.1cleaned"
    char_based_file = "filtered_test.txt"
    learner_file = "conll.tok.orig"
    JMGR_file = "JMGR"
    amu_file = "AMU"
    cuui_file = "CUUI"
    iitb_file = "IITB"
    ipn_file = "IPN"
    nthu_file = "NTHU"
    pku_file = "PKU"
    post_file = "POST"
    rac_file = "RAC"
    sjtu_file = "SJTU"
    ufc_file = "UFC"
    umc_file = "UMC"
    camb_file = "CAMB"
    gold_file = "corrected_official-2014.0.txt.comparable"
    files = [ACL2016RozovskayaRothOutput_file,
    char_based_file,
    JMGR_file,
    amu_file,
    cuui_file,
    iitb_file,
    ipn_file,
    nthu_file,
    pku_file,
    post_file,
    rac_file,
    sjtu_file,
    ufc_file,
    umc_file,
    camb_file]
    last = False
    correction = 0
    count = 0

    # # calculate the number of sentences with corrections and ratio
    # with open(r"/home/borgr/ucca/data/conll14st-test-data/noalt/official-2014.combined.m2", "r") as fl:
    #   for line in fl:
    #       if last and line.startswith("A"):
    #           correction += 1
    #       last = False
    #       if line.startswith("S"):
    #           count +=1
    #           last = True
    # print(count, correction, 1.0*correction/count)
    # return

    # # systems' significance
    # pool = Pool(POOL_SIZE)
    # results = pool.imap_unordered(m2score_sig, files)
    # pool.close()
    # pool.join()
    # results = [[1, 0, 0], [1, 0, 0]] + list(results) + [[1, 1, 1], [1, 1, 1]]
    # print(results)

    # # perfect annotator significance
    # perfect_dir = "/home/borgr/ucca/assess_learner_language/calculations_data/"
    # pool = Pool(POOL_SIZE)
    # files = ["perfect_output_for_" + str(m) + "_sgss.m2" for m in ALTERNATIVE_GOLD_MS]
    # gold_files = [perfect_dir + str(m) + "_sgss.m2" for m in ALTERNATIVE_GOLD_MS]
    # input_dirs = [perfect_dir] * len(files)
    # results = pool.imap_unordered(m2score_sig_in_one,list(zip(files, gold_files, input_dirs)))
    # pool.close()
    # pool.join()
    # results = list(results)
    # print(results)

    # perfect annotator sari score

    pool = Pool(POOL_SIZE)
    results = pool.imap_unordered(sari_score, [ALTERNATIVE_GOLD_MS])
    pool.close()
    pool.join()
    results = list(results)
    print(results)


def m2score_sig_in_one(tpl):
    if len(tpl) == 2:
        return m2score_sig(tpl[0], tpl[1])
    if len(tpl) == 3:
        return m2score_sig(tpl[0], tpl[1], tpl[2])
    return m2score_sig(tpl[0], tpl[1], tpl[2], tpl[3])


def read_system(file):
    # load system hypotheses
    fin = m2scorer.smart_open(file, 'r')
    if sys.version_info < (3, 0):
        system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
    else:
        system_sentences = [line.strip() for line in fin.readlines()]
    fin.close()

    return system_sentences


def m2score(m=None, system_file=None, gold_file=r"./data/conll14st-test-data/noalt/official-2014.combined.m2"):
    directory = r"./calculations_data/"
    system_file = system_file if system_file else directory + \
        "perfect_output_for_" + str(m) + "_sgss.m2"
    gold_file = gold_file if gold_file else directory + str(m) + "_sgss.m2"
    print("testing score of " + system_file)
    # load source sentences and gold edits
    source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

    system_sentences = read_system(system_file)

    assert(len(system_sentences) == len(source_sentences)
           and len(system_sentences) == len(gold_edits))
    return m2scorer.get_score(system_sentences, source_sentences, gold_edits, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False)


def m2score_sig(filename, gold_file=r"./data/conll14st-test-data/noalt/official-2014.combined.m2", input_dir=r"./data/paragraphs/", output_dir=r"./results/significance/"):
    system_file = input_dir + filename
    n_samples = 1000
    print("testing significance of " + filename)
    # load source sentences and gold edits
    source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

    system_sentences = read_system(system_file)

    statfunction = lambda source, gold, system: m2scorer.get_score(
        system, source, gold, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False)
    data = (source_sentences, gold_edits, system_sentences)
    test_significance(statfunction, data, output_dir +
                      str(n_samples) + "_" + filename, n_samples=n_samples)


def sari_sig(m, output_dir=r"./results/significance/"):
    n_samples = 1000
    print("testing significance of sari with m=" + str(m))
    gold_db = read_simplification()
    idx = gold_db[ORIGIN].apply(normalize_sentence).isin(
        sentences.apply(normalize_sentence))
    sentences = pd.concat(
        [sentences] + [gold_db.loc[idx, ORIGIN]] * len(gold_files), ignore_index=True)
    simplifications = pd.concat(
        [simplifications] + [gold_db.loc[idx, i] for i in range(1, 9)], ignore_index=True)
    statfunction = lambda x: sari_score(m, sentences, simplifications)
    test_significance(statfunction, None, output_dir +
                      str(n_samples) + "_sari" + str(m), n_samples=n_samples)


def sari_score(m, sentences, simplifications):
    corpus_size = 2000
    unique_sentences = sentences.unique()
    res = []
    for i in range(corpus_size):
        chosen_index = np.random.randint(0, unique_sentences.size - 1)
        chosen_sentence = unique_sentences.iloc[chosen_index]
        corresponding_simplifications = simplifications[
            sentences == chosen_sentence]
        chosen_simplifications = []
        for i in range(m + 1):
            chosen_ind = np.random.randint(
                0, corresponding_simplifications.size)
            chosen_simplifications.append(corresponding_simplifications.iloc[
                chosen_ind])
        res.append(SARI_score(chosen_sentence, chosen_simplifications[:-1], chosen_simplifications[-1]))
    return np.mean(res)

def test_significance(statfunction, data, filename=None, alpha=0.05, n_samples=100, method='bca', output='lowhigh', epsilon=0.001, multi=True):
    """ checks the confidence rate of alpha over n_samples based on the empirical distribution data writes to file the results.
        if filename already exists its content is considered to be the results of the function
        if filename is None the results are not save to any file"""
    if filename == None:
        res = scikits.bootstrap.ci(data, statfunction, alpha, n_samples, method, output, epsilon, multi)
    elif not os.path.isfile(filename):
        print("calculating for " + str(filename))
        res = scikits.bootstrap.ci(data, statfunction, alpha, n_samples, method, output, epsilon, multi)
        with open(filename, "w") as fl:
            fl.write(str(res))
    else:
        with open(filename, "r") as fl:
            res = fl.readlines()
    print(filename, "results:", res)
    return res


if __name__ == '__main__':
    main()
