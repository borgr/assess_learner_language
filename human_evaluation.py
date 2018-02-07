import traceback
import os
import subprocess
import re
from itertools import repeat
import json
import pandas as pd
import shlex
import multiprocessing
from multiprocessing import Pool
import argparse
POOL_SIZE = multiprocessing.cpu_count()
POOL_SIZE = 4

import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
import distance
from nltk.translate import bleu_score
import annalyze_crowdsourcing as an

from rank import ucca_parse_files
from rank import semantics_score
from rank import gleu_scores
from rank import SARI_score, SARI_max_score
from rank import BLEU_score
from rank import grammaticality_score
from rank import create_one_sentence_files
from rank import parse_location
from rank import sentence_m2

from m2scorer import m2scorer
import pathlib

# constants
DEBUG = False
SYSTEM1RANK = "system1rank"
SENTENCE_ID = "segmentId"
SYSTEM1NAME = "system1Id"
ANNOTATOR = "judgeID"
BN = "BN"
REFS = ""  # NUCLEB only
PAPER = "referencless_paper"
OLD_PAPER = "judgments_paper"
BEST = "bpractice"
SHORT_PAPER = "p"
SHORT_OLD = "o"
SHORT_BEST = "b"

SOURCE_BLEU = "source_bleu"
BLEU = "bleu"
R_LEV, LEVENSHTEIN = "reference levenshtein", "levenshtein"
M2, GLEU, SARI, MAX_SARI, GRAMMAR, UCCA_SIM, SENTENCE, SOURCE, SYSTEM_ID = "m2", "gleu", "sari", "max_sari", "grammar", "uccaSim", "sentence", "source", "systemId"
SRC_LNG, TRG_LANG, SRC_ID, DOC_ID, SEG_ID, MEASURE_ID = "srclang", "trglang", "srcIndex", "documentId", "segmentId", "judgeId"
DB_COLS = [SOURCE_BLEU, BLEU, R_LEV, LEVENSHTEIN, M2, GLEU, SARI, MAX_SARI, GRAMMAR, UCCA_SIM,
           SENTENCE, SOURCE, SENTENCE_ID, SYSTEM_ID]

CACHE_EVERY = 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate evaluation for GEC.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-t", "--judgment_type",
                        help="Which type of judgment to do, like " +
                        PAPER + " - Courtney Napoles, Keisuke Sakaguchi, Joel Tetreault 2016 replication (short:" + SHORT_PAPER + "), " +
                        OLD_PAPER + " - use annotation from Grundkiewicz et al. 2015 (short:" + SHORT_OLD + " )" +
                        BEST + " - for best practice. (short:" + SHORT_BEST + ") ", type=str, default=BEST)
    parser.add_argument("-c", "--cache_every",
                        help="Cache results every X sentences.", type=int, default=100)
    parser.add_argument("-d", "--debug",
                        help="Debug mode doesn't use or save cache, also it runs only on few sentences.", action="store_true")
    parser.add_argument("-f", "--force",
                        help="Force recalculation, ignore caches even if exists.", action="store_true")
    args, _ = parser.parse_known_args()
    HUMAN_JUDGMENT_TYPE = args.judgment_type
    if HUMAN_JUDGMENT_TYPE == SHORT_OLD:
        HUMAN_JUDGMENT_TYPE = OLD_PAPER
    elif HUMAN_JUDGMENT_TYPE == SHORT_PAPER:
        HUMAN_JUDGMENT_TYPE = PAPER
    elif HUMAN_JUDGMENT_TYPE == SHORT_BEST:
        HUMAN_JUDGMENT_TYPE = BEST
    if args.debug:
        DEBUG = args.debug
    CACHE_EVERY = args.cache_every

# globals
REFS = [BN]
HUMAN_JUDGMENT_SYMB = HUMAN_JUDGMENT_TYPE + \
    "_" if HUMAN_JUDGMENT_TYPE != PAPER else "" # acts as a version of the 
ASSESS_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
DATA_DIR = ASSESS_DIR + "data/"
HUMAN_JUDGMENTS_DIR = DATA_DIR + "human_judgements/"
PARAGRAPHS_DIR = DATA_DIR + "paragraphs/"
REFERENCE_DIR = DATA_DIR + "references/"
CALCULATIONS_DIR = "calculations_data/human_judgements/"
PARSE_DIR = CALCULATIONS_DIR + "/ucca_parse/"
ONE_SENTENCE_DIR = CALCULATIONS_DIR + "/one_sentence_files/"
CACHE_FILE = CALCULATIONS_DIR + \
    "_".join(REFS) + HUMAN_JUDGMENT_SYMB + "cache.pkl"
SCORE_FILE = CALCULATIONS_DIR + \
    "_".join(REFS) + HUMAN_JUDGMENT_SYMB + "score_db.pkl"
RESULTS_DIR = ASSESS_DIR + "/results/"
TRUE_KILL_DIR = RESULTS_DIR + "/HumanEvaluation/"
TS_DIR = TRUE_KILL_DIR + "TS/"


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


def get_lines_from_file(filename, lines, normalize=lambda x: x):
    with open(filename) as fl:
        text = np.array(fl.readlines())
        if lines is not None:
            lines = np.array(lines)
            text = text[lines]
        return (normalize(line.replace("\n", "")) for line in text)


def get_edits_from_file(filename, lines):
    lines = np.array(lines)
    source_sentences, gold_edits = m2scorer.load_annotation(filename)
    return np.array(gold_edits)[lines]


def systemXNumber(x):
    return -1


SYSTEMS = ["POST", "NTHU", "AMU", "CUUI", "IITB", "IPN",
           "PKU", "RAC", "SJTU", "UFC", "UMC", "CAMB", "SOURCE", "NUCLEA"]
if DEBUG:
    SYSTEMS = SYSTEMS[-2:]


def systemXId(x):
    return SYSTEMS[x]


def system_by_id(system):
    return SYSTEMS.index(system)


def create_measure_db(score_db, measure_name, measure_from_row):
    count = 0
    ranked_db = []
    for idx in score_db.loc[:, SENTENCE_ID].unique():
        cur_db = score_db.loc[score_db.loc[
            :, SENTENCE_ID] == idx, :].drop_duplicates()
        # print(cur_db)
        sentence_ranked_row = ["Complex", "simple",
                               int(idx), -1, int(idx), measure_name]
        scores = cur_db.apply(measure_from_row, axis=1).values
        argsort = np.flipud(np.argsort(scores))
        scores = scores[argsort]
        cur_db = cur_db.iloc[argsort, :]
        ranked_systems = pd.Index(cur_db.loc[:, SYSTEM_ID])
        if len(ranked_systems) != len(SYSTEMS):
            print("system lengths in the db and experiment do not match",
                  len(ranked_systems), len(SYSTEMS), ranked_systems, SYSTEMS)
            continue
        count += 1
        ranks = []
        for i, score in enumerate(scores):
            if i != 0 and scores[i] == scores[i - 1]:
                ranks.append(ranks[-1])
            else:
                ranks.append(i + 1)
        for i in range(len(SYSTEMS)):
            sentence_ranked_row.append(systemXNumber(i))
            system_id = systemXId(i)
            sentence_ranked_row.append(system_id)
            sentence_ranked_row.append(
                ranks[ranked_systems.get_loc(system_id)])
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
        print("Cached db with size", cache.shape)
        print("Cached systems", cache[SYSTEM_ID].unique())


def save_for_Truekill(score_db, name, measure_from_row, dr=TRUE_KILL_DIR, force=False):
    filename = os.path.join(dr, name + ".csv")
    if force or not os.path.isfile(filename):
        db = create_measure_db(
            score_db, name, measure_from_row)

        print("saving to", filename, "with", len(db), "rows")
        db.to_csv(filename, index=False)
        return db
    else:
        return pd.read_csv(filename)


def normalize_sentence(s):
    s = an.normalize_sentence(s)
    if len(s):
        s = s[0].upper() + s[1:] + "."
    return s


def convert_edit_reflist2dic(edits):
    edits = list(edits)
    edits, rest = edits[0], edits[1:]
    for edit in rest:
        i = max(edits.keys())
        for key, val in edit.items():
            i += 1
            edits[i] = edit[key]
    return [edits]


def calculate_all_scores(sent_id, source, references, edits, system, gleu, cached, one_sentence_dir, learner_file, system_file, system_id):
    cache = cached[cached[SENTENCE] == system]
    calculated = cache.columns.values
    # if system not in system_sentences_calculated:
    if not cache.empty and M2 in calculated:
        m2 = cache[M2].values[0]
    else:
        edits = convert_edit_reflist2dic(edits)
        m2 = m2scorer.get_score([system], [source], edits, max_unchanged_words=2, beta=0.5,
                                ignore_whitespace_casing=True, verbose=False, very_verbose=False, should_cache=False)
    if not cache.empty and LEVENSHTEIN in calculated:
        leven = cache[LEVENSHTEIN].values[0]
    else:
        leven = distance.levenshtein(source, system)
        leven = 1 if len(source) == 0 else 1 - leven / len(source)
    if not cache.empty and R_LEV in calculated:
        r_leven = cache[R_LEV].values[0]
    else:
        r_leven = 1
        for ref in references:
            cur = distance.levenshtein(ref, system)
            cur = 1 if len(ref) == 0 else 1 - cur / len(ref)
            if cur < r_leven:
                r_leven = cur
    if not cache.empty and GLEU in calculated:
        # print(cache)
        # assert gleu == cache[GLEU].values[0], "cached glue is " + str(
        # cache[GLEU].values[0]) + " while calculated gleu is " + str(gleu) +
        # str((source, references, system))
        gleu = cache[GLEU].values[0]
    if not cache.empty and SARI in calculated:
        sari = cache[SARI].values[0]
    else:
        sari = SARI_score(source, references, system)
    if not cache.empty and MAX_SARI in calculated:
        max_sari = cache[MAX_SARI].values[0]
    else:
        max_sari = SARI_max_score(source, references, system)
    if not cache.empty and GRAMMAR in calculated:
        grammar = cache[GRAMMAR].values[0]
    else:
        grammar = grammaticality_score(
            source, system, one_sentence_dir)
    if not cache.empty and UCCA_SIM in calculated:
        uccaSim = cache[UCCA_SIM].values[0]
    else:
        uccaSim = semantics_score(
            learner_file, system_file, PARSE_DIR, sent_id, sent_id)
    if not cache.empty and BLEU in calculated:
        bleu = cache[BLEU].values[0]
    else:
        bleu = BLEU_score(source, references, system, smoothing=bleu_score.SmoothingFunction(
        ).method3)  # method3 = NIST geometric sequence smoothing
    if not cache.empty and SOURCE_BLEU in calculated:
        s_bleu = cache[SOURCE_BLEU].values[0]
    else:
        s_bleu = BLEU_score(source, [source], system, smoothing=bleu_score.SmoothingFunction(
        ).method3)  # method3 = NIST geometric sequence smoothing
    return (s_bleu, bleu, r_leven, leven, m2, gleu, sari, max_sari, grammar, uccaSim, system, source, sent_id, system_id)


def create_score_db(cache_file, judgment_files, references_files, edits_files, learner_file, system_files, one_sentence_dir, ucca_parse_dir, cache_every, results_file="", force=False, clean_cache=False, use_all=False):
    """ when using force, and clean_cache relevant caches will be overwritten"""
    if (not force) and os.path.isfile(results_file):
        print("reading scores from", results_file)
        return pd.read_pickle(results_file)

    # parse xmls
    sentence_ids = []
    for judgment_file in judgment_files:
        db = parse_xml(judgment_file)
        # print(type(db[SENTENCE_ID].map(lambda x: int(x)).unique()), db[SENTENCE_ID].map(lambda x: int(x)).unique())
        sentence_ids.append(db[SENTENCE_ID].map(lambda x: int(x)).unique())
    sentence_ids = np.unique(np.concatenate(sentence_ids))
    if use_all:
        print("using all sentences, not only human judgment ones")
        sentence_ids = list(range(1312))
    print("number of judgments:", len(sentence_ids))

    cached = load_cache(cache_file, force=clean_cache)

    # read relevant lines note that id135 = line 136
    source_lines = list(get_lines_from_file(
        learner_file, sentence_ids, normalize_sentence))
    score_db = []
    # load files
    references_edits = zip(
        *[get_edits_from_file(fl, sentence_ids) for fl in edits_files])
    references_edits = [list(x) for x in references_edits]
    references_lines = zip(
        *[get_lines_from_file(fl, sentence_ids, normalize_sentence) for fl in references_files])
    references_lines = [list(x) for x in references_lines]

    lines_references = [list(get_lines_from_file(fl, sentence_ids, normalize_sentence))
                        for fl in references_files]
    ucca_parse_files(system_files + [learner_file],
                     ucca_parse_dir, normalize_sentence=normalize_sentence)
    create_one_sentence_files([x for lines in references_lines for x in lines] +
                              source_lines, one_sentence_dir)
    scores = []
    for x, system_file in enumerate(system_files):
        system_id = systemXId(x)
        print("calculating for", system_id)
        system_lines = list(get_lines_from_file(system_file, sentence_ids))
        system_lines = [normalize_sentence(x) for x in system_lines]
        assert(len(system_lines) == len(sentence_ids))
        create_one_sentence_files(system_lines, one_sentence_dir)
        gleu_sentence_scores = gleu_scores(
            source_lines, lines_references, [system_lines])[1]
        gleu_sentence_scores = [s[0] for s in gleu_sentence_scores]
        pool = Pool(POOL_SIZE)
        score_db = pool.starmap(calculate_all_scores, zip(
            sentence_ids, source_lines, references_lines, references_edits, system_lines, gleu_sentence_scores, repeat(cached), repeat(one_sentence_dir), repeat(learner_file), repeat(system_file), repeat(system_id)))
        pool.close()
        pool.join()
        scores.append(pd.DataFrame(score_db, columns=DB_COLS))
        if (not DEBUG):
            cur_scores = pd.DataFrame(
                score_db, columns=DB_COLS)
            cached = cur_scores.append(cached, ignore_index=True)
            print("new cache", cached.shape)
            save_cache(cached, CACHE_FILE)

    score_db = pd.concat(
        scores)
    print(score_db[SYSTEM_ID].unique())
    # if not cached.empty:
    #     score_db = score_db.append(cached, ignore_index=True)
    score_db[SYSTEM_ID] = score_db[SYSTEM_ID].apply(normalize_system_name)

    if not DEBUG:
        save_cache(score_db, CACHE_FILE)

    if results_file:
        save_cache(score_db, results_file)
    return score_db


def normalize_system_name(name):
    return "SOURCE" if name.lower() in ["source", "input", "src"] else name


def combined_score(uccaSim, grammar, alpha):
    score = float(uccaSim) * alpha + (1 - alpha) * float(grammar)
    return score


def main(args):
    if DEBUG:
        print("***********************")
        print("DEBUGGING!")
        print("***********************")

    # Human Judgments
    all_judgments = "all_judgments"
    judgments8 = "8judgments"
    combined_judgments = "combined_judgments"

    if HUMAN_JUDGMENT_TYPE == BEST:
        judgments_files = [HUMAN_JUDGMENTS_DIR + judgments8 + ".xml",
                           HUMAN_JUDGMENTS_DIR + all_judgments + ".xml"]
        judgment_name = combined_judgments
    elif HUMAN_JUDGMENT_TYPE == PAPER:
        judgment_name = judgments8
        judgments_files = [HUMAN_JUDGMENTS_DIR + judgments8 + ".xml"]
    elif HUMAN_JUDGMENT_TYPE == OLD_PAPER:
        judgment_name = all_judgments
        judgments_files = [HUMAN_JUDGMENTS_DIR + all_judgments + ".xml"]
    else:
        raise NotImplementedError(
            "Unknown type", HUMAN_JUDGMENT_TYPE, "as a human judgment type")
    # measure judgments
    learner_file = PARAGRAPHS_DIR + "conll.tok.orig"
    first_nucle = REFERENCE_DIR + "NUCLEA"
    second_nucle = REFERENCE_DIR + "NUCLEB"
    # combined_nucle = REFERENCE_DIR + "NUCLE.m2"
    bn = REFERENCE_DIR + "BN"
    # ALL =  REFERENCE_DIR + "ALL.m2"

    references_files = [second_nucle]
    edits_files = [second_nucle]

    if BN in REFS:
        edits_files += [bn]
        references_files += [bn + str(i) for i in range(1, 11)]

    edits_files = [x + ".m2" for x in edits_files]

    system_files = [
        PARAGRAPHS_DIR + systemXId(x) for x in range(len(SYSTEMS) - 2)] + [learner_file] + [REFERENCE_DIR + systemXId(len(SYSTEMS) - 1)]

    clean_cache = args.force
    if not clean_cache:
        force = False
    else:
        force = True
    score_db = create_score_db(CACHE_FILE, judgments_files, references_files, edits_files,
                               learner_file, system_files, ONE_SENTENCE_DIR, PARSE_DIR, CACHE_EVERY, SCORE_FILE, force=force, clean_cache=clean_cache, use_all=(HUMAN_JUDGMENT_TYPE != BEST))

    # format for truekill
    if not force:
        force = False
    names = []
    names.append("BLEU")
    bleu = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[BLEU]), force=force)
    names.append("iBLEU")
    alpha=0.8
    ibleu = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[BLEU]) * alpha + (1 - alpha) * float(row[SOURCE_BLEU]), force=force)
    names.append("$LD_{S\\rightarrow O}$")
    Grammatical = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[LEVENSHTEIN]), force=force)
    names.append("$LD_{O\\rightarrow R}$")
    Grammatical = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[R_LEV]), force=force)
    names.append("GLEU")
    gleu_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[GLEU]), force=force)
    names.append("SARI")
    sari_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[SARI]), force=force)
    names.append("MAX-SARI")
    max_sari_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[MAX_SARI]), force=force)
    names.append("USim")
    uccaSim_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[UCCA_SIM]), force=force)
    names.append("$M^2$")
    m2_db = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[M2][2]), force=force)
    names.append("LT")
    Grammatical = save_for_Truekill(
        score_db, names[-1], lambda row: float(row[GRAMMAR]), force=force)

    # names.append("levenshtein")
    # leven = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[LEVENSHTEIN]), force=force)
    # names.append("glue")
    # gleu_db = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[GLEU]), force=force)
    # names.append("sari")
    # sari_db = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[SARI]), force=force)
    # names.append("max-sari")
    # max_sari_db = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[MAX_SARI]), force=force)
    # names.append("uccaSim")
    # uccaSim_db = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[UCCA_SIM]), force=force)
    # names.append("m2")
    # m2_db = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[M2][2]), force=force)
    # names.append("grammar")
    # Grammatical = save_for_Truekill(
    #     score_db, names[-1], lambda row: float(row[GRAMMAR]), force=force)
    # for alpha in np.linspace(0, 1, 11):
    #     names.append("combined" + "{0:.2f}".format(alpha))
    #     combined = save_for_Truekill(
    #         score_db, names[-1], lambda row: combined_score(row[UCCA_SIM], row[GRAMMAR], alpha), force=force)
    # for alpha in np.linspace(0, 1, 11):
    #     names.append("combined_minus_UCCASim" + "{0:.1f}".format(alpha))
    #     combined = save_for_Truekill(
    #         score_db, names[-1], lambda row: combined_score(-row[UCCA_SIM], row[GRAMMAR], alpha), force=force)
    # for alpha in np.linspace(0, 1, 11):
    #     names.append("combined_levenshtein" + "{0:.1f}".format(alpha))
    #     combined = save_for_Truekill(
    #         score_db, names[-1], lambda row: combined_score(-row[LEVENSHTEIN], row[GRAMMAR], alpha), force=force)

    if not force:
        force = False
    # run trueskill human judgments
    judgment_csv = os.path.join(
        HUMAN_JUDGMENTS_DIR, "corrected_" + judgment_name + ".csv")
    if force or not os.path.isfile(judgment_csv):
        correct_wmt_csv(os.path.join(
            HUMAN_JUDGMENTS_DIR, judgment_name + ".csv"))
    judgment_rank = trueskill_rank(2, judgment_csv, judgment_name, force=force)
    judgment_rank = ["NUCLEA" if x == "refmix1" else x for x in judgment_rank]
    judgment_rank = ["SOURCE" if x.lower() in ["source", "input", "src"]
                     else x for x in judgment_rank]
    print("Human judgment rank")
    print("rank", judgment_rank)

    # run trueskill systems
    ranks = []
    human_ranks = list(range(len(judgment_rank)))
    correlations_db = []
    assert ([judgment_rank.index(x) for x in judgment_rank] == human_ranks)
    for name in names:
        measure_path = os.path.join(TRUE_KILL_DIR, name + ".csv")
        rank = trueskill_rank(len(SYSTEMS), measure_path,
                              name, False, force=force)
        id_rank = [judgment_rank.index(x) for x in rank if x in judgment_rank]
        print(name)
        pearson = pearsonr(human_ranks, id_rank)
        print("Pearson (val, P-val):", pearson[0], pearson[1])
        spearman = spearmanr(human_ranks, id_rank)
        print("Spearman (val, P-val):", spearman[0], spearman[1])
        correlations_db.append(
            [name, pearson[0], pearson[1], spearman[0], spearman[1]])
        print("rank", rank)
        ranks.append(rank)
    correlations_file = os.path.join(CALCULATIONS_DIR, HUMAN_JUDGMENT_SYMB + "correlations.csv")
    print("writing to ", correlations_file)
    correlations_db = pd.DataFrame(np.array(correlations_db), columns=["Name", "Spearman", "Spearman P", "Pearson", "Pearson P"]).to_csv(correlations_file, index=False)


def correct_wmt_csv(filename):
    """ converts places where is contains several systems to different rows, 
        NOTE: currently assumes exactly 2 systems in each row """
    lines = []
    with open(filename) as fl:
        for i, line in enumerate(fl.readlines()):
            parts = line.split(",")
            if i == 0:
                first_system_col = parts.index("system1Id")
                second_system_col = parts.index("system2Id")
                if first_system_col > second_system_col:
                    first_system_col, second_system_col = second_system_col, first_system_col
            systems = parts[first_system_col].split()
            second_systems = parts[second_system_col].split()
            for system in systems:
                system = normalize_system_name(system)
                for second_system in second_systems:
                    second_system = normalize_system_name(second_system)
                    lines.append(
                        ",".join(parts[0:first_system_col] + [system] + parts[first_system_col + 1:second_system_col] + [second_system] + parts[second_system_col + 1:]))
    outfile = os.path.join(os.path.dirname(filename),
                           "corrected_" + os.path.basename(filename))
    with open(outfile, "w") as fl:
        print("wrote corrected csv to", outfile)
        fl.writelines(lines)


def trueskill_rank(system_num, measure_db_path, measure_name, verbose=False, force=False):
    trueskill_dir = os.path.join(TS_DIR, measure_name)
    trueskill_file = os.path.normpath(os.path.join(trueskill_dir, "_mu_sigma.json"))
    if not os.path.isdir(trueskill_dir):
        os.makedirs(trueskill_dir)
        print("trueskill_dir:", trueskill_dir)
    if verbose:
        print("ranking by", measure_name)
    if force or not os.path.isfile(trueskill_file):
        verbose_symb = "-v" if verbose else ""
        ps = subprocess.Popen(("cat", measure_db_path), stdout=subprocess.PIPE)
        command = "python " + ASSESS_DIR + "wmt-trueskill/src/infer_TS.py -e -s " + \
            str(system_num) + verbose_symb + \
            ' -d 0 -n 2 "' + trueskill_dir + os.sep + '"'
        try:
            p = subprocess.Popen(shlex.split(command), stdin=ps.stdout)
            p.communicate()
        except CalledProcessError:
            traceback.print_exc()

    with open(trueskill_file) as fl:
        mu_sig = json.load(fl)
    mu_sig.pop("data_points")
    rank = sorted(mu_sig.keys(), key=lambda x: -mu_sig[x][0])
    if verbose:
        print("rank", rank)
    return rank


if __name__ == '__main__':
    main(args)
