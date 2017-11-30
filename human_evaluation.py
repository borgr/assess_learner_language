import os
import subprocess
import shlex
import re
import json
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import distance

import annalyze_crowdsourcing as an

from rank import ucca_parse_files
from rank import semantics_score
from rank import gleu_scores
from rank import grammaticality_score
from rank import create_one_sentence_files
from rank import parse_location
from rank import sentence_m2

from m2scorer import m2scorer
import pathlib

DEBUG = False
# DEBUG = True
CACHE_EVERY = 100

SYSTEM1RANK = "system1rank"
SENTENCE_ID = "segmentId"
SYSTEM1NAME = "system1Id"
ANNOTATOR = "judgeID"

ASSESS_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
DATA_DIR = ASSESS_DIR + "data/"
HUMAN_JUDGMENTS_DIR = DATA_DIR + "human_judgements/"
PARAGRAPHS_DIR = DATA_DIR + "paragraphs/"
REFERENCE_DIR = DATA_DIR + "references/"
CALCULATIONS_DIR = "calculations_data/human_judgements/"
PARSE_DIR = CALCULATIONS_DIR + "/ucca_parse/"
ONE_SENTENCE_DIR = CALCULATIONS_DIR + "/one_sentence_files/"
CACHE_FILE = CALCULATIONS_DIR + "cache.pkl"
SCORE_FILE = CALCULATIONS_DIR + "score_db.pkl"
RESULTS_DIR = ASSESS_DIR + "/results/"
TRUE_KILL_DIR = RESULTS_DIR + "/HumanEvaluation/"
TS_DIR = TRUE_KILL_DIR + "TS/"
LEVENSHTEIN = "levenshtein"
M2, GLEU, GRAMMAR, UCCA_SIM, SENTENCE, SOURCE, SYSTEM_ID = "m2", "gleu", "grammar", "uccaSim", "sentence", "source", "systemId"
SRC_LNG, TRG_LANG, SRC_ID, DOC_ID, SEG_ID, MEASURE_ID = "srclang", "trglang", "srcIndex", "documentId", "segmentId", "judgeId"
DB_COLS = [LEVENSHTEIN, M2, GLEU, GRAMMAR, UCCA_SIM,
           SENTENCE, SOURCE, SENTENCE_ID, SYSTEM_ID]

pathlib.Path(PARSE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ONE_SENTENCE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(TRUE_KILL_DIR).mkdir(parents=True, exist_ok=True)

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


def get_lines_from_file(filename, lines):
    lines = np.array(lines)
    with open(filename) as fl:
        text = np.array(fl.readlines())
        return (line.replace("\n", "") for line in text[lines])


def get_edits_from_file(filename, lines):
    lines = np.array(lines)
    source_sentences, gold_edits = m2scorer.load_annotation(filename)
    return np.array(gold_edits)[lines]


def systemXNumber(x):
    return -1


SYSTEMS = ["POST", "JMGR", "AMU", "CUUI", "IITB", "IPN", "NTHU",
           "PKU", "RAC", "SJTU", "UFC", "UMC", "CAMB", "src", "NUCLEA"]
if DEBUG:
    SYSTEMS = SYSTEMS[-3:]


def systemXId(x):
    return SYSTEMS[x]


def system_by_id(system):
    return SYSTEMS.index(system)


def create_measure_db(score_db, measure_name, measure_from_row):
    ranked_db = []
    for idx in score_db.loc[:, SENTENCE_ID].unique():
        cur_db = score_db.loc[score_db.loc[
            :, SENTENCE_ID] == idx, :].drop_duplicates()
        sentence_ranked_row = ["Complex", "simple",
                               int(idx), -1, int(idx), measure_name]
        scores = cur_db.apply(measure_from_row, axis=1).values
        argsort = np.flipud(np.argsort(scores))
        scores = scores[argsort]
        # print(measure_name, "vals", scores)
        cur_db = cur_db.iloc[argsort, :]
        ranked_systems = pd.Index(cur_db.loc[:, SYSTEM_ID])
        ranks = []
        for i, score in enumerate(scores):
            if i != 0 and scores[i] == scores[i - 1]:
                ranks.append(ranks[-1])
            else:
                ranks.append(i + 1)
        # print(cur_db, "cur")
        # print(ranked_systems, "ranked_systems")
        for i in range(len(SYSTEMS)):
            sentence_ranked_row.append(systemXNumber(i))
            system_id = systemXId(i)
            sentence_ranked_row.append(system_id)
            sentence_ranked_row.append(
                ranks[ranked_systems.get_loc(system_id)])
            # print(sentence_ranked_row)
        ranked_db.append(sentence_ranked_row)
    columns = [SRC_LNG, TRG_LANG, SRC_ID, DOC_ID, SEG_ID, MEASURE_ID] + \
        ["system" + str(1 + x) + header
         for x in range(len(SYSTEMS)) for header in ["Number", "Id", "rank"]]
    ranked_db = pd.DataFrame(ranked_db, columns=columns)
    return ranked_db


def load_cache(cache_file, force=False):
    if force:
        return pd.DataFrame(
            [], columns=DB_COLS)
    elif not os.path.isfile(cache_file):
        print("Cache was not found, creating a new cache results_file", cache_file)
        return pd.DataFrame(
            [], columns=DB_COLS)
    else:
        print("reading cache")
        return pd.read_pickle(cache_file)


def save_cache(cache, cache_file, verbose=True):
    cache.drop_duplicates(inplace=True)
    cache.to_pickle(cache_file)
    if verbose:
        print("Cached with db with size", cache.shape)


def save_for_Truekill(score_db, name, measure_from_row, dr=TRUE_KILL_DIR, force=False):
    filename = os.path.join(dr, name + ".csv")
    if force or not os.path.isfile(filename):
        db = create_measure_db(
            score_db, name, measure_from_row)

        db.to_csv(filename, index=False)
        return db
    else:
        return pd.read_csv(filename)

def normalize_sentence(s):
    s = an.normalize_sentence(s)
    if len(s):
        s = s[0].upper() + s[1:] + "."
    return s

def create_score_db(cache_file, judgment_file, references_files, edits_files, learner_file, system_files, one_sentence_dir, ucca_parse_dir, cache_every, results_file="", force=False):
    if (not force) and os.path.isfile(results_file):
        print("reading scores from", results_file)
        return pd.read_pickle(results_file)

    # parse xmls
    db = parse_xml(judgment_file)
    sentence_ids = db[SENTENCE_ID].map(lambda x: int(x)).unique()
    print("number of judgments:", len(sentence_ids))

    cached = load_cache(cache_file, force=force)

    # read relevant lines note that id135 = line 136
    source_lines = list(get_lines_from_file(learner_file, sentence_ids))
    source_lines = [normalize_sentence(x) for x in source_lines]
    score_db = []
    # load files
    references_edits = zip(
        *[get_edits_from_file(fl, sentence_ids) for fl in edits_files])
    references_edits = [list(x) for x in references_edits]
    references_lines = zip(
        *[get_lines_from_file(fl, sentence_ids) for fl in references_files])
    references_lines = [list(x) for x in references_lines]
    references_lines = [[normalize_sentence(sent) for sent in ref] for ref in references_lines]

    lines_references = [list(get_lines_from_file(fl, sentence_ids))
                        for fl in references_files]
    lines_references = [[normalize_sentence(sent) for sent in ref] for ref in lines_references]
    ucca_parse_files(system_files + references_files +
                     [learner_file], ucca_parse_dir, normalize_sentence=normalize_sentence)
    create_one_sentence_files([x for lines in references_lines for x in lines] +
                              source_lines, one_sentence_dir)

    system_sentences_calculated = set()
    for x, system_file in enumerate(system_files):
        system_id = systemXId(x)
        print("calculating for", system_id)
        system_lines = list(get_lines_from_file(system_file, sentence_ids))
        system_lines = [normalize_sentence(x) for x in system_lines]
        create_one_sentence_files(system_lines, one_sentence_dir)
        gleu_sentence_scores = gleu_scores(
            source_lines, lines_references, [system_lines])[1]
        gleu_sentence_scores = [s[0] for s in gleu_sentence_scores]
        uncached = 0
        for i, (sent_id, source, references, edits, system, gleu) in enumerate(zip(
                sentence_ids, source_lines, references_lines, references_edits, system_lines, gleu_sentence_scores)):
            cache = cached[cached[SENTENCE] == system]
            if not cache.empty:
                m2 = cache[M2].values[0]
                leven = cache[LEVENSHTEIN].values[0]
                gleu = cache[GLEU].values[0]
                grammar = cache[GRAMMAR].values[0]
                uccaSim = cache[UCCA_SIM].values[0]
                score_db.append(
                    (leven, m2, gleu, grammar, uccaSim, system, source, sent_id, system_id))
                system_sentences_calculated.add(system)
            elif system not in system_sentences_calculated:
                leven = distance.levenshtein(source, system)
                uccaSim = semantics_score(
                    learner_file, system_file, PARSE_DIR, i, i)
                if uccaSim < 0.5:
                    print("score with system is", uccaSim, "source",
                        source, "system", system)
                grammar = grammaticality_score(
                    source, system, one_sentence_dir)
                edits = list(edits)
                m2 = m2scorer.get_score([system], [source], edits, max_unchanged_words=2, beta=0.5,
                                        ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)
                score_db.append(
                    (leven, m2, gleu, grammar, uccaSim, system, source, sent_id, system_id))
                system_sentences_calculated.add(system)
                uncached += 1
            else:
                for row in score_db:
                    if row[4] == system:
                        row = list(row[:])
                        row[-1] = system_id
                        score_db.append(row)
                        uncached += 1
                        break
                assert score_db[-1][4] == system

            if (not DEBUG) and (uncached == cache_every):
                uncached = 0
                cur_scores = pd.DataFrame(
                    score_db, columns=DB_COLS)
                cached = cur_scores.append(cached, ignore_index=True)

                save_cache(cached, CACHE_FILE)
                print("calculated", len(score_db),
                      "sentences overall", i + 1, "for", system_file)
                # print(Cached)
                # return
            if DEBUG and len(score_db) % 2 == 0 and len(score_db) != 0:
                print("fast run for DEBUG")
                break

    score_db = pd.DataFrame(
        score_db, columns=DB_COLS)

    if not cached.empty:
        score_db = score_db.append(cached, ignore_index=True)

    if not DEBUG and uncached != 0:
        save_cache(score_db, CACHE_FILE)

    if results_file:
        save_cache(score_db, results_file)
    return score_db

def combined_score(uccaSim, grammar, alpha):
    score = float(uccaSim) * alpha + (1 - alpha) * float(grammar)
    print("uccaSim", uccaSim)
    print("grammar", grammar)
    print("alpha", alpha)
    print("score", score)
    return score

def main():
    if DEBUG:
        print("***********************")
        print("DEBUGGING!")
        print("***********************")

    # calculate

    learner_file = PARAGRAPHS_DIR + "conll.tok.orig"
    first_nucle = REFERENCE_DIR + "NUCLEA"
    second_nucle = REFERENCE_DIR + "NUCLEB"
    # combined_nucle = REFERENCE_DIR + "NUCLE.m2"
    # BN = REFERENCE_DIR + "BN.m2"
    # ALL =  REFERENCE_DIR + "ALL.m2"
    references_files = [second_nucle]
    edits_files = [second_nucle + ".m2"]
    system_files = [
        PARAGRAPHS_DIR + systemXId(x) for x in range(len(SYSTEMS) - 2)] + [learner_file] + [REFERENCE_DIR + systemXId(len(SYSTEMS) - 1)]

    # TODO perhaps combine them?
    judgments_file = HUMAN_JUDGMENTS_DIR + "all_judgments"
    judgments_file = HUMAN_JUDGMENTS_DIR + "8judgments"
    # augmented_judgments_file = HUMAN_JUDGMENTS_DIR + "augmented_all_judgments"
    force = False
    score_db = create_score_db(CACHE_FILE, judgments_file + ".xml", references_files, edits_files,
                               learner_file, system_files, ONE_SENTENCE_DIR, PARSE_DIR, CACHE_EVERY, SCORE_FILE, force=force)
    # print(score_db)
    # print(score_db[SYSTEM_ID].unique(), len(score_db[SYSTEM_ID].unique()))
    # return
    # format for truekill
    force = False
    names = []
    names.append("glue")
    gleu_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[GLEU]), force=force)
    names.append("uccaSim")
    uccaSim_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[UCCA_SIM]), force=force)
    names.append("m2")
    m2_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[M2][2]), force=force)
    names.append("grammar")
    Grammatical = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[GRAMMAR]), force=force)
    names.append("levenshtein")
    Grammatical = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[LEVENSHTEIN]), force=force)
    for alpha in np.linspace(0, 1, 101):
        names.append("combined" + str(alpha))
        combined = save_for_Truekill(score_db, names[-1], lambda row: combined_score(row[UCCA_SIM], row[GRAMMAR], alpha), force=force)

    # run truekills
    # run human judgments trueskill
    judgment_name = os.path.basename(judgments_file)
    judgment_rank = trueskill_rank(2, judgments_file + ".csv", judgment_name)

    judgment_rank = ["NUCLEA" if x == "refmix1" else x for x in judgment_rank]
    # augmented_judgments_name = os.path.basename(augmented_judgments_file)
    # augmented_judgment_rank = trueskill_rank(3, augmented_judgments_file + ".csv", augmented_judgments_name)
    ranks = []
    for name in names:
        measure_path = os.path.join(TRUE_KILL_DIR, name + ".csv")
        rank = trueskill_rank(len(SYSTEMS), measure_path, name, True)

        id_rank = [judgment_rank.index(x) for x in rank if x in judgment_rank]
        print(name, pearsonr(list(range(len(judgment_rank))), id_rank))
        ranks.append(rank)


def trueskill_rank(system_num, measure_db_path, measure_name, verbose=False):
    trueskill_dir = os.path.join(TS_DIR + measure_name) + os.sep
    trueskill_file = os.path.join(trueskill_dir, "_mu_sigma.json")
    if verbose:
        print("ranking by", measure_name)
    if not os.path.isfile(trueskill_file):
        ps = subprocess.Popen(("cat", measure_db_path), stdout=subprocess.PIPE)
        subprocess.check_call(shlex.split("python " + ASSESS_DIR + "wmt-trueskill/src/infer_TS.py -e -s " +
                                          str(system_num) + " -d 0 -n 2 " + trueskill_dir), stdin=ps.stdout)
        # subprocess.Popen(r"cat " + measure_db_path +
        #                  " | python " + ASSESS_DIR + "wmt-trueskill/src/infer_TS.py -e -s " + str(system_num) + " -d 0 -n 2 " + output_file)
    with open(trueskill_file) as fl:
        mu_sig = json.load(fl)
    mu_sig.pop("data_points")
    rank = sorted(mu_sig.keys(), key=lambda x: -mu_sig[x][0])
    if verbose:
        print(rank)
    return rank


if __name__ == '__main__':
    main()
