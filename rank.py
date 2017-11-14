import time
import os
import sys
import re
import scipy
import numpy as np
import pandas as pd
from multiprocessing import Pool
from subprocess import call
import pickle
import json
from functools import reduce
import operator
import platform
import random
import six
ASSESS_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
# UCCA_DIR = '/home/borgr/ucca/ucca'
# ASSESS_DIR = '/home/borgr/ucca/assess_learner_language'
# ASSESS_DIR = '/cs/labs/oabend/borgr/assess_learner_language'
TUPA_DIR = '/cs/labs/oabend/borgr/tupa/'
UCCA_DIR = TUPA_DIR + 'ucca'
sys.path.append(ASSESS_DIR + '/m2scorer/scripts')
sys.path.append(UCCA_DIR)
sys.path.append(UCCA_DIR + '/scripts/distances')
sys.path.append(UCCA_DIR + '/ucca')
# sys.path.append(TUPA_DIR)

from ucca.ioutil import file2passage
import subprocess
import codecs
from m2scorer import m2scorer
from gec_ranking.scripts.gleu import GLEU
import align
# from significance_testing import m2score
from ucca.ioutil import passage2file
from ucca.convert import from_text
from correction_quality import word_diff

from simplification import SARI
import annalyze_crowdsourcing as an

POOL_SIZE = 7
full_rerank = True

from tupa.parse import Parser
model_path = "/cs/labs/oabend/borgr/tupa/models/bilstm"
parser = Parser(model_path, "bilstm")


def main():
    # parse_JFLEG()
    # # rerank_by_m2()
    # for gamma in np.linspace(0,1,11):
    #   print(m2score(system_file="calculations_data/uccasim_rerank/" + str(gamma) + "_" + "uccasim_rank_results",
    #                 gold_file=r"/home/borgr/ucca/assess_learner_language/data/references/ALL.m2"))
    #   # rerank_by_uccasim(gamma)
    #   rerank_by_uccasim(gamma)
    # print(m2score(system_file=r"/home/borgr/ucca/assess_learner_language/data/paragraphs/conll14st.output.1cleaned",
    #             gold_file=r"/home/borgr/ucca/assess_learner_language/data/references/ALL.m2"))
    # reduce_k_best(100, 10, filename)
    # rerank_by_wordist()
    # rerank_by_SARI()
    rerank_by_SARI("moses")
    # anounce_finish()


def parse_JFLEG():
    JFLEG_dir = ASSESS_DIR + "/data/jfleg/dev"
    (path, dirs, files) = next(os.walk(JFLEG_dir))
    filenames = [path + os.sep + fl for fl in files]
    ucca_parse_files(filenames, JFLEG_dir + os.sep + "xmls")


def rerank_by_uccasim(gamma=0.27):
    data_dir = "data/"
    # only used to extract source sentences
    first_nucle = data_dir + "references/" + "NUCLEA.m2"
    k_best_dir = data_dir + "K-best/"
    system_file = k_best_dir + "conll14st.output.1.best100"
    calculations_dir = "calculations_data/uccasim_rerank/"
    ucca_parse_dir = calculations_dir + "/ucca_parse/"
    full = "full" if full_rerank else ""
    output_file = full + str(gamma) + "_" + "uccasim_rank_results"
    out_text_file = calculations_dir + output_file
    out_res_file = calculations_dir + "score_" + output_file

    if not os.path.isfile(out_text_file):
        gold_file = first_nucle  # only used to extract source sentences
        print("acquiring source")
        source_sentences, _ = m2scorer.load_annotation(gold_file)

        source_sentences = source_sentences
        # load system hypotheses
        fin = m2scorer.smart_open(system_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        packed_system_sentences = get_roro_packed(system_sentences)

        print("parsing")
        # print(reduce(operator.add, packed_system_sentences))
        ucca_parse(reduce(operator.add, packed_system_sentences) +
                   source_sentences, ucca_parse_dir)

        print("reranking")
        # find top ranking
        pool = Pool(POOL_SIZE)
        assert(len(packed_system_sentences) == len(source_sentences))
        if full_rerank:
            results = pool.starmap(referece_less_full_rerank, zip(source_sentences, packed_system_sentences, [
                                   ucca_parse_dir] * len(packed_system_sentences), [gamma] * len(packed_system_sentences)))
        else:
            results = pool.starmap(referece_less_oracle, zip(source_sentences, packed_system_sentences, [
                                   ucca_parse_dir] * len(packed_system_sentences), [gamma] * len(packed_system_sentences)))
        pool.close()
        pool.join()
        results = list(results)
        if full_rerank:
            results = [x for y in results for x in y]
        sentences = "\n".join(list(zip(*results))[0])
        results = list(zip(*results))[1]
        results = "\n".join([str(x) for x in results])

        print("writing to " + out_text_file)
        with codecs.open(out_text_file, "w+", "utf-8") as fl:
            fl.write(sentences)
        with open(out_res_file, "w+") as fl:
            fl.write(results)


def rerank_by_wordist():
    data_dir = "data/"
    # only used to extract source sentences
    first_nucle = data_dir + "references/" + "NUCLEA.m2"
    k_best_dir = data_dir + "K-best/"
    system_file = k_best_dir + "conll14st.output.1.best100"
    calculations_dir = "calculations_data/uccasim_rerank/"
    # ucca_parse_dir = calculations_dir + "/ucca_parse/"
    min_change = 2
    output_file = str(min_change) + "wordist_rank_results"
    out_text_file = calculations_dir + output_file
    out_res_file = calculations_dir + "score_" + output_file
    out_source_file = calculations_dir + "source" + output_file
    if not os.path.isfile(out_text_file):
        gold_file = first_nucle  # only used to extract source sentences
        print("acquiring source")
        source_sentences, _ = m2scorer.load_annotation(gold_file)

        source_sentences = source_sentences
        # load system hypotheses
        fin = m2scorer.smart_open(system_file, 'r')
        system_sentences = [line.strip() for line in fin.readlines()]
        fin.close()

        packed_system_sentences = get_roro_packed(system_sentences)

        # print("parsing")
        # print(reduce(operator.add, packed_system_sentences))
        # ucca_parse(reduce(operator.add, packed_system_sentences) + source_sentences, ucca_parse_dir)

        print("reranking")
        # find top ranking
        pool = Pool(POOL_SIZE)
        assert(len(packed_system_sentences) == len(source_sentences))
        results = pool.starmap(wordist_oracle, zip(
            source_sentences, packed_system_sentences))
        pool.close()
        pool.join()
        results = list(results)
        tmp = []
        out_sentences = []
        for (k, n), sent in zip(results, source_sentences):
            if n > min_change:
                tmp.append((k, n))
                out_sentences.append(sent)
        results = tmp

        sentences = "\n".join(list(zip(*results))[0])
        results = list(zip(*results))[1]
        results = "\n".join([str(x) for x in results])
        out_sentences = "\n".join([str(x) for x in out_sentences])

        print("writing to " + out_text_file)
        with codecs.open(out_text_file, "w+", "utf-8") as fl:
            fl.write(sentences)
        with codecs.open(out_source_file, "w+", "utf-8") as fl:
            fl.write(out_sentences)
        with open(out_res_file, "w+") as fl:
            fl.write(results)


def rerank_by_m2():
    data_dir = "data/"
    k_best_dir = data_dir + "K-best/"
    system_file = k_best_dir + "conll14st.output.1.best100"

    reference_dir = data_dir + "references/"
    first_nucle = reference_dir + "NUCLEA.m2"
    combined_nucle = reference_dir + "NUCLE.m2"
    BN = reference_dir + "BN.m2"
    ALL = reference_dir + "ALL.m2"
    gold_files = [first_nucle, combined_nucle, BN, ALL]

    (path, dirs, files) = next(os.walk(reference_dir))
    for fl in files:
        if "subset" in fl:
            gold_files.append(path + fl)

    calculations_dir = "calculations_data/"
    output_file = "first_rank_results"
    for gold_file in gold_files:
        out_text_file = calculations_dir + \
            output_file + name_extension(gold_file)[0]
        out_res_file = calculations_dir + "prf_" + \
            output_file + name_extension(gold_file)[0]
        if not os.path.isfile(out_text_file):
            print("processing " + gold_file)
            source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

            # load system hypotheses
            fin = m2scorer.smart_open(system_file, 'r')
            system_sentences = [line.decode("utf8").strip()
                                for line in fin.readlines()]
            fin.close()

            # pack and parse RoRo's k-best
            packed_system_sentences = get_roro_packed(source_sentences)
            # candidate_num = 0
            # for sentence_num, (source, this_edits) in enumerate(zip(source_sentences, gold_edits)):
            #   curr_sentences = []
            #   # keep packing until reached another sentence, assumes k-best are consequetive
            #   while (candidate_num < len(system_sentences) and
            #         system_sentences[candidate_num].split()[0] == str(sentence_num)):
            #       sentence = re.sub("\|\d+-\d+\| ","",system_sentences[candidate_num].split("|||")[1][1:])
            #       candidate_num += 1
            #       curr_sentences.append(sentence)
            #   packed_system_sentences.append(curr_sentences)

            # find top ranking
            pool = Pool(POOL_SIZE)
            assert(len(packed_system_sentences) == len(gold_edits)
                   and len(gold_edits) == len(source_sentences))
            results = pool.imap(RBM_oracle, zip(
                source_sentences, packed_system_sentences))
            pool.close()
            pool.join()
            results = list(results)
            sentences = "\n".join(list(zip(*results))[0])
            results = list(zip(*results))[1]
            results = "\n".join([str(x) for x in results])

            print("writing to " + out_text_file)
            with codecs.open(out_text_file, "w+", "utf-8") as fl:
                fl.write(sentences)
            with open(out_res_file, "w+") as fl:
                fl.write(results)


def load_nisioi_k_best(k_best_dir):
    system_dir = k_best_dir + "NTS_beam12_12hyp"
    for root, dirs, files in os.walk(system_dir):
        all_lines = []
        for filename in files:
            if re.search("h\d+$", filename):
                with open(root + os.sep + filename) as fl:
                    all_lines.append([line.replace("\n", "")
                                      for line in fl.readlines()])
    all_lines = list(zip(*all_lines))
    return all_lines


def load_moses_k_best(k_best_dir):
    system_file = k_best_dir + "Moses_based"
    # load system hypotheses
    with open(system_file, "r") as fl:
        system_sentences = []
        cur = "0"
        for line in fl:
            splitted = line.split("|||")
            if cur != splitted[0]:
                system_sentences.append([])
                cur = splitted[0]
            system_sentences[-1].append(splitted[1])
    return system_sentences


def rerank_by_SARI(k_best="nisioi"):
    data_dir = "data/simplification/"
    k_best_dir = data_dir + "K-best/"

    DATA_DIR = os.path.dirname(os.path.realpath(
        __file__)) + os.sep + "/simplification/data/"
    TURK_CORPUS_DIR = DATA_DIR + "turkcorpus/"
    TURKERS_DIR = TURK_CORPUS_DIR + "truecased/"

    ORIGIN = "origin"

    db = []
    # for root, dirs, files in os.walk(TURKERS_DIR):
    #   for filename in files:
    #       cur_db = pd.read_table(TURKERS_DIR + filename, names=["index", ORIGIN, 1, 2, 3, 4, 5, 6, 7, 8])
    #       db.append(cur_db)
    # db = pd.concat(db, ignore_index=True)
    filename = "test.8turkers.organized.tsv"
    db = pd.read_table(TURKERS_DIR + filename,
                       names=["index", ORIGIN, 1, 2, 3, 4, 5, 6, 7, 8])
    db.drop("index", inplace=True, axis=1)
    db.dropna(inplace=True, axis=0)
    db.applymap(an.normalize_sentence)

    with open(TURK_CORPUS_DIR + "test.8turkers.tok.turk.0") as fl:
        gold = fl.readlines()

    keep = []
    for i, row in db.iloc[:, -8:].iterrows():
        keep.append(db.ix[i, ORIGIN] in row.values)
    keep = np.array(keep)
    db = db.iloc[keep, :]
    source_sentences = db[ORIGIN].tolist()
    references = db.iloc[:, -8:].values

    if "nisioi":
        system_sentences = np.array(load_nisioi_k_best(k_best_dir))[keep]
    else:
        system_sentences = np.array(load_moses_k_best(k_best_dir))[keep]
    gold = np.array(gold)[keep]

    calculations_dir = "calculations_data/"
    output_file = "simplification_rank_results_" + k_best

    out_text_file = calculations_dir + output_file + "_origin"
    with codecs.open(out_text_file, "w+", "utf-8") as fl:
        fl.write("\n".join(source_sentences))

    out_text_file = calculations_dir + output_file + "_gold"
    with codecs.open(out_text_file, "w+", "utf-8") as fl:
        fl.write("\n".join(gold).replace("\n\n", "\n"))

    for ref_num in range(8, 0, -1):
        out_text_file = calculations_dir + output_file + str(ref_num) + "refs"
        out_res_file = calculations_dir + "SARI_" + \
            output_file + str(ref_num) + "refs"
        if not os.path.isfile(out_text_file):
            print("ranking with", ref_num, "refs")

            # pack k-best
            packed_system_sentences = []
            for source, refs, system in zip(source_sentences, references, system_sentences):
                packed_system_sentences.append(
                    (source, refs[np.random.randint(0, 8, ref_num)].tolist(), system))

            # find top ranking
            pool = Pool(POOL_SIZE)
            assert(len(packed_system_sentences) == len(references)
                   and len(references) == len(source_sentences))
            results = pool.imap(SARI_oracle, packed_system_sentences)
            pool.close()
            pool.join()
            results = list(results)
            sentences = "\n".join(list(zip(*results))[0])
            results = list(zip(*results))[1]
            results = "\n".join([str(x) for x in results])

            print("writing to " + os.path.realpath(out_text_file))
            with codecs.open(out_text_file, "w+", "utf-8") as fl:
                fl.write(sentences)
            with open(out_res_file, "w+") as fl:
                fl.write(results)
        else:
            print("skipped calculating with", ref_num,
                  " references, file already exists.")


def reduce_k_best(big_k, small_k, filename, outfile=None):
    if outfile is None:
        outfile = os.path.normpath(filename)
        outfile = os.path.split(outfile)
        outfile[1] = str(small_k) + "_" + outfile[1]
        outfile = "".join(outfile)
    output = []
    with open(outfile) as fl:
        for i, line in enumerate(fl):
            if i % big_k < small_k:
                output.append(line)
    # finish that
    raise


def referece_less_full_rerank(source, system_sentences, parse_dir, gamma):
    combined_scores = []
    for sentence in set(system_sentences):
        combined_scores.append(
            (sentence, reference_less_score(source, sentence, parse_dir, gamma)))

    return sorted(combined_scores, key=lambda x: x[1])


def wordist_oracle(source, system_sentences):
    maximum = 0
    for sentence in set(system_sentences):
        combined_score = word_diff(source, sentence)
        if maximum <= combined_score:
            maximum = combined_score
            chosen = sentence, combined_score
    return chosen


def referece_less_oracle(source, system_sentences, parse_dir, gamma):
    maximum = 0
    for sentence in set(system_sentences):
        combined_scores = reference_less_score(
            source, sentence, parse_dir, gamma)
        if maximum <= combined_score:
            maximum = combined_score
            chosen = sentence, combined_score
    # print(chosen)
    return chosen


def RBM_oracle(tple):
    maximum = 0
    source, this_edits, system_sentences = tple
    for sentence in system_sentences:
        p, r, f = score(source, this_edits, sentence)
        if maximum <= f:
            maximum = f
            chosen = sentence, (p, r, f)
    return chosen


def SARI_oracle(tple):
    maximum = 0
    source, references, system_sentences = tple
    for sentence in system_sentences:
        score = SARI_score(source, references, sentence)
        if maximum <= score:
            maximum = score
            chosen = sentence, score
    return chosen


def parse_location(output_dir, filename, sentence_num=None):
    filename = os.path.basename(filename)
    cur_dir = os.path.join(output_dir, filename)
    if sentence_num is None:
        return cur_dir
    return os.path.join(cur_dir, filename + str(sentence_num) + ".xml")


def ucca_parse_files(filenames, output_dir, clean=False):
    # parse_command = "python ../tupa/tupa/parse.py -c bilstm -m ../tupa/models/bilstm -o "+ output_dir +" "
    # print("parsing with:", parse_command)

    if filenames:
        for filename in filenames:
            cur_output_dir = parse_location(output_dir, filename)
            if os.path.isdir(cur_output_dir):
                print("Skipping parsing, file already parsed in", cur_output_dir)
            else:
                os.mkdir(cur_output_dir)
                print("parsing " + filename)
                with open(filename, "r") as fl:
                    text = fl.readlines()
                text = from_text(text, split=True, one_per_line=True)
                text = list(text)
                # text = [item for line in text for item in from_text(line, split=True)]
                for i, passage in enumerate(parser.parse(text)):
                    passage2file(passage, parse_location(
                        output_dir, filename, i))
                print("printed all xmls from " + cur_output_dir)
                if clean:
                    filenames = os.listdir(cur_output_dir)
                    for filename in filenames:
                        if filename.endswith(".txt"):
                            os.remove(os.path.join(cur_output_dir, item))
        # res = subprocess.run(parse_command.split() + list(files), stdout=subprocess.PIPE)


def create_one_sentence_files(sentences, output_dir):
    count = 0
    for sentence in list(set(sentences)):
        filename = str(get_sentence_id(sentence, output_dir))
        txt_file = filename + ".txt"
        filepath = os.path.join(output_dir, txt_file)
        if not os.path.isfile(filepath):
            with open(filepath, "w+") as fl:
                fl.write(sentence)


#! deprecated
def ucca_parse(sentences, output_dir):
    parse_command = "python ../tupa/tupa/parse.py -c bilstm -m ../tupa/models/bilstm -o " + output_dir + " "
    # print("parsing with:", parse_command)
    filenames = []
    count = 0
    for sentence in list(set(sentences)):
        # print("parsing" + sentence[:20])
        filename = str(get_sentence_id(sentence, output_dir))
        txt_file = filename + ".txt"
        xml_file = filename + ".xml"
        if not os.path.isfile(output_dir + txt_file):
            with open(output_dir + txt_file, "w+") as fl:
                fl.write(sentence)
        if not os.path.isfile(output_dir + xml_file):
            filenames.append(output_dir + txt_file)

    # check = []
    # for sentence in list(set(filenames)):
    #   if sentence not in check:
    #       check.append(sentence)
    #   else:
    #       print("repeats")
    #       return
    # print(sorted(filenames))
    if filenames:
        print("parsing sentences")
        res = subprocess.run(parse_command.split() +
                             filenames, stdout=subprocess.PIPE)
    # print(res)
    # call(parse_command.split() + filenames)


def get_roro_packed(system_sentences):
    """ pack and parse RoRo's k-best"""
    candidate_num = 0
    packed_system_sentences = []
    for sentence_num, source in enumerate(system_sentences):
        curr_sentences = []
        # keep packing until reached another sentence, assumes k-best are
        # consequetive
        while (candidate_num < len(system_sentences) and
               system_sentences[candidate_num].split()[0] == str(sentence_num)):
            sentence = re.sub(
                "\|\d+-\d+\| ", "", system_sentences[candidate_num].split("|||")[1][1:])
            candidate_num += 1
            curr_sentences.append(sentence)
        if curr_sentences:
            packed_system_sentences.append(curr_sentences)
    return packed_system_sentences


_id_dics = {}


def get_sentence_id(sentence, parse_dir, graceful=True):
    """ returns the sentence id in the parse_dir, 
        if graceful is true adds a new sentence id 
        if the sentence does not exist in the ids list,
        otherwise throws exception"""
    filename = "sentenceIds.pkl"
    max_id = "max"
    if parse_dir in _id_dics:
        id_dic = _id_dics[parse_dir]
    elif not os.path.isfile(parse_dir + os.sep + filename):
        print("creating a new id list")
        id_dic = {max_id: -1}
        _id_dics[parse_dir] = id_dic
    else:
        with open(parse_dir + os.sep + filename, "rb") as fl:
            id_dic = pickle.load(fl)
            _id_dics[parse_dir] = id_dic
    if not graceful:
        # print(id_dic)
        pass
    if graceful and not sentence in id_dic:
        # print("dumping" + sentence + "\n")
        id_dic[max_id] += 1
        id_dic[sentence] = id_dic[max_id]
        with open(parse_dir + os.sep + filename, "wb+") as fl:
            pickle.dump(id_dic, fl)
    # print(sentence)
    return id_dic[sentence]


def reference_less_score(source, sentence, parse_dir, gamma):
    return gamma * grammaticality_score(source, sentence, parse_dir) + (1 - gamma) * semantics_score(source, sentence, parse_dir)


def score(source, gold_edits, system):
    return sentence_m2(source, gold_edits, system)


def SARI_score(source, references, system):
    return SARI.SARIsent(system, source, references)


def semantics_score(source, sentence, parse_dir, source_sentence_id=None, sentence_id=None):
    """ accepts filename instead of sentence\source and a sentence id\source_sentence id for locating the file"""
    if align.regularize_word(source) == "":
        if align.regularize_word(sentence) == "":
            return 1
        else:
            return 0
    elif align.regularize_word(sentence) == "":
        return 0

    if source_sentence_id is None:
        source_xml = file2passage(
            parse_dir + str(get_sentence_id(source, parse_dir, False)) + ".xml")
    else:
        source_xml = file2passage(parse_location(
            parse_dir, source, source_sentence_id))
    if sentence_id is None:
        sentence_xml = file2passage(
            parse_dir + str(get_sentence_id(sentence, parse_dir, False)) + ".xml")
    else:
        sentence_xml = file2passage(
            parse_location(parse_dir, sentence, sentence_id))

    return align.fully_aligned_distance(source_xml, sentence_xml)


def grammaticality_score(source, sentence, parse_dir):
    command = "java -jar ../softwares/LanguageTool-3.7/languagetool-commandline.jar --json -l en-US"
    filename = str(get_sentence_id(sentence, parse_dir, False)) + ".txt"
    with open(os.devnull, 'wb') as devnull:
        res = subprocess.run(
            command.split() + [parse_dir + filename], stdout=subprocess.PIPE, stderr=devnull)
    out = res.stdout.decode("utf-8")
    out = re.sub(r"\\'", "'", out)
    res = json.loads(out)
    return 1 / (1 + len(res["matches"]))


def sentence_m2(source, gold_edits, system):
    return m2scorer.get_score([system], [source], [gold_edits], max_unchanged_words=2, beta=0.5, ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)


def get_gleu_stats(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    ci = scipy.stats.norm.interval(0.95, loc=mean, scale=std)
    return ['%f' % mean,
            '%f' % std,
            '(%.3f,%.3f)' % (ci[0], ci[1])]


def gleu_scores(source, references, systems, ngrams_len=4, num_iterations=500, debug=False):
    # if there is only one reference, just do one iteration
    if len(references) == 1:
        num_iterations = 1

    gleu_calculator = GLEU(ngrams_len)

    if isinstance(source, six.string_types):
        gleu_calculator.load_sources(source)
    else:
        gleu_calculator.set_sources(source)

    if isinstance(references[0], six.string_types):
        gleu_calculator.load_references(reference)
    else:
        gleu_calculator.set_references(references)

    total = []
    per_sentence = []
    for hpath in systems:
        if isinstance(hpath, six.string_types):
            with open(hpath) as instream:
                hyp = [line.split() for line in instream]
            if not debug:
                print(os.path.basename(hpath),)
        else:
            instream = hpath
            hyp = [line.split() for line in instream]

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            indices.append([random.randint(0, len(references) - 1)
                            for i in range(len(hyp))])

        if debug:
            print()
            print('===== Sentence-level scores =====')
            print('SID Mean Stdev 95%CI GLEU')

        iter_stats = [[0 for i in range(2 * ngrams_len + 2)]
                      for j in range(num_iterations)]

        for i, h in enumerate(hyp):

            gleu_calculator.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [None for r in range(len(references))]

            for j in range(num_iterations):
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None:
                    this_stats = [s for s in gleu_calculator.gleu_stats(
                        i, r_ind=ref)]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [sum(scores)
                                 for scores in zip(iter_stats[j], this_stats)]

            per_sentence.append(get_gleu_stats([gleu_calculator.gleu(stats, smooth=True)
                                                for stats in stats_by_ref]))
            if debug:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(len(references)):
                    if stats_by_ref[r] is None:
                        stats_by_ref[r] = [s for s in gleu_calculator.gleu_stats(
                            i, r_ind=r)]

                print(i,)
                print(' '.join(results[-1]))
        total.append(get_gleu_stats([gleu_calculator.gleu(stats)
                                     for stats in iter_stats]))
        if debug:
            print('\n==== Overall score =====')
            print('Mean Stdev 95%CI GLEU')
            print(' '.join(total[-1]))
        # else:
        #     print("total", total[-1][0])
    return total, per_sentence


def gleu_score(source, references, system):
    return None


def basename(name):
    return name.split("\\")[-1].split("/")[-1]


def name_extension(name):
    return basename(name).split(".")


def anounce_finish():
    if sys.platform == "linux":
        if set(("debian", "Ubuntu")) & set(platform.linux_distribution()):
            subprocess.call(['speech-dispatcher'])  # start speech dispatcher
            subprocess.call(['spd-say', '"your process has finished"'])
        else:
            # perhaps works only in ubuntu?
            a = subprocess.Popen(
                ('play --no-show-progress --null --channels 1 synth %s sine %f' % (300, 2)).split())
    elif sys.platform == "darwin":
        subprocess.call('say "your process has finished"'.split())
    else:
        import winsound
        winsound.Beep(300, 2)

if __name__ == '__main__':
    # fnamenorm   = "./turkcorpus/test.8turkers.tok.norm"
    # fnamesimp   = "./turkcorpus/test.8turkers.tok.simp"
    # fnameturk  = "./turkcorpus/test.8turkers.tok.turk."

    # ssent = "About 95 species are currently accepted ."
    # csent1 = "About 95 you now get in ."
    # csent2 = "About 95 species are now agreed ."
    # csent3 = "About 95 species are currently agreed ."
    # rsents = ["About 95 species are currently known .", "About 95 species are now accepted .", "95 species are now accepted ."]

    # print(SARI_score(csent1, rsents, ssent))
    # print(SARI_score(csent2, rsents, ssent))
    # print(SARI_score(csent3, rsents, ssent))
    main()
