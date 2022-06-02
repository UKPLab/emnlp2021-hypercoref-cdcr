#!/usr/bin/env python3
from tabulate import tabulate

import sys
from coval.conll import util, reader
from coval.eval import evaluator
from pathlib import Path
import pandas as pd

def score_all():
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
            ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
            ('lea', evaluator.lea)]

    key_file = sys.argv[1]

    NP_only = 'NP_only' in sys.argv
    remove_nested = 'remove_nested' in sys.argv
    keep_singletons = ('remove_singletons' not in sys.argv
            and 'removIe_singleton' not in sys.argv)
    min_span = False
    if ('min_span' in sys.argv
        or 'min_spans' in sys.argv
        or 'min' in sys.argv):
        min_span = True
        has_gold_parse = util.check_gold_parse_annotation(key_file)
        if not has_gold_parse:
                util.parse_key_file(key_file)
                key_file = key_file + ".parsed"


    if 'all' in sys.argv:
        metrics = allmetrics
    else:
        metrics = [(name, metric) for name, metric in allmetrics
                if name in sys.argv]
        if not metrics:
            metrics = allmetrics

    # -------------- score all sys files in a folder hierarchy which is /<seed 0>/sys.conll , /<seed 1>/sys.conll, ... -----------------

    sys_base_dir = Path(sys.argv[2])
    assert sys_base_dir.exists()
    all_scores = []
    for seed_dir in sys_base_dir.iterdir():
        try:
            seed = int(seed_dir.name)
        except ValueError:
            print(f"{seed_dir} is not a seed directory")
            continue
        for sys_file in seed_dir.iterdir():
            if sys_file.suffix == ".conll":
                scores = evaluate(key_file, sys_file.absolute(), metrics, NP_only, remove_nested, keep_singletons, min_span)
                scores["seed"] = seed
                scores = scores.set_index("seed", append=True)
                all_scores.append(scores)
    all_scores = pd.concat(all_scores)
    all_scores = all_scores.unstack("measure").dropna(axis="columns")
    scores_aggregated = all_scores.describe().loc[["mean", "std"]]

    all_scores.to_pickle(sys_base_dir / "scores.pkl")
    with (sys_base_dir / "scores.txt").open("w") as f:
        f.write(f"""
INDIVIDUAL SCORES
-----------------

{all_scores.to_csv()}

{tabulate(all_scores, headers="keys")}

AGGREGATED SCORES
-----------------

{scores_aggregated.to_csv()}

{tabulate(scores_aggregated, headers="keys")}
""")

def evaluate(key_file, sys_file, metrics, NP_only, remove_nested,
        keep_singletons, min_span):
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only,
            remove_nested, keep_singletons, min_span)

    conll = 0
    conll_subparts_num = 0

    scores = {}
    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
                metric,
                beta=1)
        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1
        scores[name] = [recall, precision, f1]

    if conll_subparts_num == 3:
        conll = (conll / 3) * 100
        scores["conll"] = [None, None, conll]

    scores = pd.DataFrame(scores, index=["R", "P", "F1"])
    scores.index.name = "measure"
    return scores


if __name__ == '__main__':
    score_all()