import os
import re
import pandas as pd
import numpy as np

from rank import ucca_parse_files
from rank import semantics_score

from rank import gleu_scores

from rank import grammaticality_score
from rank import create_one_sentence_files
from rank import parse_location

from rank import sentence_m2
from m2scorer import m2scorer

DEBUG = False
# DEBUG = True
CACHE_EVERY = 100

SYSTEM1RANK = "system1rank"
SENTENCE_ID = "segmentId"
SYSTEM1NAME = "system1Id"
ANNOTATOR = "judgeID"

ASSESS_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
# ASSESS_DIR = '/home/borgr/ucca/assess_learner_language/'
# ASSESS_DIR = '/cs/labs/oabend/borgr/assess_learner_language/'
DATA_DIR = ASSESS_DIR + "data/"
HUMAN_JUDGMENTS_DIR = DATA_DIR + "human_judgements/"
PARAGRAPHS_DIR = DATA_DIR + "paragraphs/"
REFERENCE_DIR = DATA_DIR + "references/"
CALCULATIONS_DIR = "calculations_data/human_judgements/"
PARSE_DIR = CALCULATIONS_DIR + "/ucca_parse/"
ONE_SENTENCE_DIR = CALCULATIONS_DIR + "/one_sentence_files/"
CACHE_FILE = CALCULATIONS_DIR + "cache.pkl"
RESULTS_DIR = ASSESS_DIR + "/results/"
TRUE_KILL_DIR = RESULTS_DIR + "/HumanEvaluation/"
M2, GLEU, GRAMMAR, UCCA_SIM, SENTENCE, SOURCE, SYSTEM_ID = "m2", "gleu", "grammar", "uccaSim", "sentence", "source", "systemId"
SRC_LNG, TRG_LANG, SRC_ID, DOC_ID, SEG_ID, MEASURE_ID = "srclang", "trglang", "srcIndex", "documentId", "segmentId", "judgeId"


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
                system_ids = re.search(
                    'system="([\w\s,_-]*)"', line).group(1)
                for system_id in re.split("\W", system_ids):
                    judgments.append(
                        (sent_id, system_rank, system_id, annotator))

    db = pd.DataFrame(judgments, columns=[
                      SENTENCE_ID, SYSTEM1RANK, SYSTEM1NAME, ANNOTATOR])
    return db


def get_lines_from_file(file, lines):
    lines = np.array(lines)
    with open(file) as fl:
        text = np.array(fl.readlines())
        return (line.replace("\n", "") for line in text[lines])


def get_edits_from_file(file, lines):
    lines = np.array(lines)
    source_sentences, gold_edits = m2scorer.load_annotation(file)
    return np.array(gold_edits)[lines]


def systemXNumber(x):
    return -1


SYSTEMS = ["POST", "JMGR", "AMU", "CUUI", "IITB", "IPN", "NTHU",
           "PKU", "RAC", "SJTU", "UFC", "UMC", "CAMB", "NUCLEA"]
if DEBUG:
    SYSTEMS = SYSTEMS[-3:]


def systemXId(x):
    return SYSTEMS[x]


def system_by_id(system):
    return SYSTEMS.index(system)


def create_measure_db(score_db, measure_name, measure_from_row):
    all_rank_db = []
    for idx in score_db.loc[:, SENTENCE_ID].unique():
        cur_db = score_db.loc[score_db.loc[:, SENTENCE_ID] == idx, :]
        sentence_rank = ["Complex", "simple", idx, -1, idx, measure_name]
        argsort = np.flipud(np.argsort(
            cur_db.apply(measure_from_row, axis=1).values))
        print(measure_name, "vals", cur_db.apply(
            measure_from_row, axis=1).values)
        cur_db = cur_db.iloc[argsort, :]
        ranks = pd.Index(cur_db.loc[:, SYSTEM_ID])
        for i in range(len(SYSTEMS)):
            sentence_rank.append(systemXNumber(i))
            system_id = systemXId(i)
            sentence_rank.append(system_id)
            sentence_rank.append(ranks.get_loc(system_id) + 1)

        all_rank_db.append(sentence_rank)
    columns = [SRC_LNG, TRG_LANG, SRC_ID, DOC_ID, SEG_ID, MEASURE_ID] + \
        ["system" + str(1 + x) + header
         for x in range(len(SYSTEMS)) for header in ["Number", "id", "rank"]]
    all_rank_db = pd.DataFrame(all_rank_db, columns=columns)
    return all_rank_db


def load_cache(cache_file):
    if not os.path.isfile(cache_file):
        print("Cache was not found, creating a new cache file", cache_file)
        return pd.DataFrame(
            [], columns=[M2, GLEU, GRAMMAR, UCCA_SIM, SENTENCE, SOURCE, SENTENCE_ID])
    else:
        print("reading cache")
        return pd.read_pickle(cache_file)


def save_cache(cache, cache_file, verbose=True):
    cache.to_pickle(cache_file)
    if verbose:
        print("Cached with db with size", cache.shape)


def save_for_Truekill(db, name, dr=TRUE_KILL_DIR):
    db.to_csv(os.path.join(dr, name + ".csv"))


def main():
    if DEBUG:
        print("***********************")
        print("DEBUGGING!")
        print("***********************")

    # parse xmls
    # judgments_file = HUMAN_JUDGMENTS_DIR + "8judgments.xml"
    judgments_file = HUMAN_JUDGMENTS_DIR + "all_judgments.xml"
    db = parse_xml(judgments_file)
    sentence_ids = db[SENTENCE_ID].map(lambda x: int(x)).unique()
    print("number of judgments:", len(sentence_ids))

    cached = load_cache(CACHE_FILE)
    # cached_ids = cached.loc[:, SENTENCE_ID].unique()
    # sentence_ids = [
    #     sentence_id for sentence_id in sentence_ids if sentence_id not in cached_ids]
    # print("From which not cached", len(sentence_ids))
    # read relevant lines note that id135 = line 136
    learner_file = PARAGRAPHS_DIR + "conll.tok.orig"
    source_lines = list(get_lines_from_file(learner_file, sentence_ids))
    first_nucle = REFERENCE_DIR + "NUCLEA"
    second_nucle = REFERENCE_DIR + "NUCLEB"
    # combined_nucle = REFERENCE_DIR + "NUCLE.m2"
    # BN = REFERENCE_DIR + "BN.m2"
    # ALL =  REFERENCE_DIR + "ALL.m2"
    references_files = [second_nucle]
    edits_files = [second_nucle + ".m2"]

    score_db = []
    # load files
    references_edits = zip(
        *[get_edits_from_file(fl, sentence_ids) for fl in edits_files])
    references_edits = [list(x) for x in references_edits]
    references_lines = zip(
        *[get_lines_from_file(fl, sentence_ids) for fl in references_files])
    references_lines = [list(x) for x in references_lines]

    lines_references = [list(get_lines_from_file(fl, sentence_ids))
                        for fl in references_files]

    # JMGR_file = PARAGRAPHS_DIR + "JMGR"
    # amu_file = PARAGRAPHS_DIR + "AMU"
    # cuui_file = PARAGRAPHS_DIR + "CUUI"
    # iitb_file = PARAGRAPHS_DIR + "IITB"
    # ipn_file = PARAGRAPHS_DIR + "IPN"
    # nthu_file = PARAGRAPHS_DIR + "NTHU"
    # pku_file = PARAGRAPHS_DIR + "PKU"
    # post_file = PARAGRAPHS_DIR + "POST"
    # rac_file = PARAGRAPHS_DIR + "RAC"
    # sjtu_file = PARAGRAPHS_DIR + "SJTU"
    # ufc_file = PARAGRAPHS_DIR + "UFC"
    # umc_file = PARAGRAPHS_DIR + "UMC"
    # camb_file = PARAGRAPHS_DIR + "CAMB"
    # system_files = [JMGR_file, amu_file, cuui_file, iitb_file,
    #                 ipn_file, nthu_file, pku_file, post_file,
    # rac_file, sjtu_file, ufc_file, umc_file, camb_file, first_nucle]
    system_files = [
        PARAGRAPHS_DIR + systemXId(x) for x in range(len(SYSTEMS) - 1)] + [REFERENCE_DIR + systemXId(len(SYSTEMS) - 1)]
    ucca_parse_files(system_files + references_files +
                     [learner_file], PARSE_DIR)
    create_one_sentence_files([x for lines in references_lines for x in lines] +
                              source_lines, ONE_SENTENCE_DIR)
    # return

    system_sentences_calculated = set()
    for x, system_file in enumerate(system_files):
        system_id = systemXId(x)
        print("calculating for", system_id)
        system_lines = list(get_lines_from_file(system_file, sentence_ids))
        create_one_sentence_files(system_lines, ONE_SENTENCE_DIR)
        gleu_sentence_scores = gleu_scores(
            source_lines, lines_references, [system_lines])[1]
        gleu_sentence_scores = [s[0] for s in gleu_sentence_scores]

        for i, (sent_id, source, references, edits, system, gleu) in enumerate(zip(
                sentence_ids, source_lines, references_lines, references_edits, system_lines, gleu_sentence_scores)):
            cache = cached[cached[SENTENCE] == system]
            if not cache.empty:
                m2 = cache[M2].values[0]
                gleu = cache[GLEU].values[0]
                grammar = cache[GRAMMAR].values[0]
                uccaSim = cache[UCCA_SIM].values[0]
                score_db.append(
                    (m2, gleu, grammar, uccaSim, system, source, sent_id, system_id))
                system_sentences_calculated.add(system)
            elif system not in system_sentences_calculated:
                uccaSim = semantics_score(
                    learner_file, system_file, PARSE_DIR, i, i)
                grammar = grammaticality_score(
                    source, system, ONE_SENTENCE_DIR)
                edits = list(edits)
                m2 = m2scorer.get_score([system], [source], edits, max_unchanged_words=2, beta=0.5,
                                        ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)
                score_db.append(
                    (m2, gleu, grammar, uccaSim, system, source, sent_id, system_id))
                system_sentences_calculated.add(system)
            else:
                for row in score_db:
                    if row[4] == system:
                        row = list(row[:])
                        row[-1] = system_id
                        score_db.append(row)
                        break
                assert score_db[-1][4] == system

            if (not DEBUG) and (len(score_db) % CACHE_EVERY == 0) and (len(score_db) != 0):
                cur_scores = pd.DataFrame(
                    score_db, columns=[M2, GLEU, GRAMMAR, UCCA_SIM, SENTENCE, SOURCE, SENTENCE_ID, SYSTEM_ID])
                cached = cur_scores.append(cached, ignore_index=True)
                cached.drop_duplicates(inplace=True)
                save_cache(cached, CACHE_FILE)
                print("calculated", len(score_db),
                      "sentences overall", i + 1, "for", system_file)
                # print(Cached)
                # return
            if DEBUG and len(score_db) % 2 == 0 and len(score_db) != 0:
                print("fast run for DEBUG")
                break

    score_db = pd.DataFrame(
        score_db, columns=[M2, GLEU, GRAMMAR, UCCA_SIM, SENTENCE, SOURCE, SENTENCE_ID, SYSTEM_ID])

    if not cached.empty:
        score_db = score_db.append(cached, ignore_index=True)

    if not DEBUG:
        save_cache(score_db, CACHE_FILE)

    name = "glue"
    gleu_db = save_for_Truekill(create_measure_db(
        score_db, name, lambda row: float(row[GLEU])), name)
    name = "m2"
    m2_db = save_for_Truekill(create_measure_db(
        score_db, name, lambda row: float(row[M2][2])), name)
    name = "uccaSim"
    uccaSim_db = save_for_Truekill(create_measure_db(
        score_db, name, lambda row: float(row[UCCA_SIM])), name)
    name = "grammar"
    Grammatical = save_for_Truekill(create_measure_db(
        score_db, name, lambda row: float(row[GRAMMAR])), name)
    for alpha in np.linspace(0, 1, 101):
        name = "combined" + str(alpha)
        combined = save_for_Truekill(create_measure_db(
            score_db, name, lambda row: float(
                float(row[UCCA_SIM]) * alpha + (1 - alpha) * float(row[GRAMMAR]))), name)
    # rank outputs on each sentence that has human judgment using ucca and lt
    # (and gleu and m2 later)

    # think what to calculate with it (percentage of agreeing comparisons \
    # truekill \ expected wins both found in Human Evaluation of Grammatical
    # Error Correction Systems)

if __name__ == '__main__':
    main()
