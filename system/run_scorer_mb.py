#!/usr/bin/env python3

import sys
from coval.conll import util, reader
from coval.eval import evaluator
from pathlib import Path


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

    # -------------- score all sys files in a folder -----------------

    sys_file_dir = Path(sys.argv[2])
    assert sys_file_dir.exists()

    best_file = None
    best_score = -1
    for sys_file in sys_file_dir.iterdir():
        if sys_file.suffix == ".conll":
            scores = evaluate(key_file, sys_file.absolute(), metrics, NP_only, remove_nested, keep_singletons, min_span)
            lea_f1 = scores["lea"][-1]
            if lea_f1 > best_score:
                best_file = sys_file.absolute()
                best_score = lea_f1

    print(f"Best LEA F1 of {best_score} was achieved with {str(best_file)}.")


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

    return scores


if __name__ == '__main__':
    score_all()