from pathlib import Path
from scipy.stats import hmean

import pandas as pd
from tabulate import tabulate

# specify the root of a directory tree containing pickled coreference resolution here -> this script generates LaTeX
# tables for the main paper (and appendix)
scores_root = Path("...")
output_path = scores_root / "meta-aggregated-results"
output_path.mkdir(exist_ok=True, parents=True)

def walk(root: Path):
    for p in root.iterdir():
        yield p
        if p.is_dir():
            yield from walk(p)

PATH = "path"
SEED = "seed"
TOPICS = "topics"
TEST_CORPUS = "test-corpus"
DEV_CORPUS = "dev-corpus"
EXPERIMENT = "experiment"
SCENARIO = "scenario"
MEASURE = "measure"
METRIC = "metric"

all_scores_list = []
for p in walk(scores_root):
    if p.suffix == ".pkl":
        scores = pd.read_pickle(p)
        scores[PATH] = str(p.relative_to(scores_root))
        scores = scores.set_index(PATH, append=True).reorder_levels([PATH, SEED])
        all_scores_list.append(scores)

all_scores = pd.concat(all_scores_list)
all_scores.columns = all_scores.columns.to_flat_index()     # to allow merging non-multiindex columns without warnings

paths = all_scores.index.droplevel(SEED).drop_duplicates().to_series()
paths = paths.str.split("/", expand=True).rename(columns={0: SCENARIO, 1: EXPERIMENT, 4: TOPICS})
paths[TEST_CORPUS] = paths[3].map(lambda corpus: corpus.split("_")[-1])
paths[DEV_CORPUS] = paths[3].map(lambda corpus: corpus.split("_")[0])

# special handling for ss dev corpus
paths.loc[paths["scenario"] == "ss", DEV_CORPUS] = paths.loc[paths["scenario"] == "ss", EXPERIMENT].str.slice(9,12)
# lemma baseline does not have a dev corpus, and lemma-delta does not have a train corpus
paths.loc[paths["experiment"] == "lemma", DEV_CORPUS] = "n/a"

all_scores = all_scores.reset_index().merge(paths[[SCENARIO, EXPERIMENT, DEV_CORPUS, TEST_CORPUS, TOPICS]], left_on=PATH, right_on=PATH)
all_scores = all_scores.drop(columns=PATH).set_index([SCENARIO, EXPERIMENT, DEV_CORPUS, TEST_CORPUS, TOPICS, SEED])
all_scores.columns = pd.MultiIndex.from_tuples(all_scores.columns, names=[MEASURE, METRIC])

all_scores.to_csv(output_path / "all_scores_unaggregated.csv")
# ☝️ this previous dataframe is the most orderly dataframe with all results in it, pre-aggregation

# FCC-T closure topics results are identical with predicted topics, therefore duplicate predicted topics results and
# fit them in as closure results
fcct_predicted_topics_results = all_scores.loc[(all_scores.index.get_level_values(TEST_CORPUS) == "fcct") & (all_scores.index.get_level_values(TOPICS) == "predicted")]
fcct_closure_topics_results = fcct_predicted_topics_results.copy().rename(index={"predicted": "closure"})
all_scores = pd.concat([all_scores, fcct_closure_topics_results]).sort_index()

aggregated = all_scores.groupby([SCENARIO, EXPERIMENT, DEV_CORPUS, TEST_CORPUS, TOPICS]).describe()
aggregated_mean_stddev = aggregated.stack([MEASURE, METRIC])[["mean", "std"]]
aggregated_mean_stddev["std"].fillna(0.0, inplace=True)     # for baselines where we only have one seed

# move all metrics from [0-1] to [0-100], except for conll F1 which already is
aggregated_mean_stddev.loc[aggregated_mean_stddev.index.get_level_values(MEASURE) != "conll"] *= 100

# siunitx at the moment cannot auto-round uncertainties, so we export the table for a number of decimal places
for decimals in range(5):
    rounded = aggregated_mean_stddev.round(decimals)
    mean_stddev_in_one = rounded["mean"].astype(str).str.cat(rounded["std"].astype(str), sep="+-")

    for met in ["muc", "bcub", "ceafe", "conll", "lea"]:
        desired_column_order = [("ecbp", met, "P"),
                                ("ecbp", met, "R"),
                                ("ecbp", met, "F1"),
                                ("fcct", met, "P"),
                                ("fcct", met, "R"),
                                ("fcct", met, "F1"),
                                ("gvc", met, "P"),
                                ("gvc", met, "R"),
                                ("gvc", met, "F1")]
        if met == "conll":
            desired_column_order = [tup for tup in desired_column_order if tup[-1] == "F1"]

        unstacked_again_means_stddev = mean_stddev_in_one.unstack([TEST_CORPUS, MEASURE, METRIC])[desired_column_order]
        unstacked_again_means = rounded["mean"].unstack([TEST_CORPUS, MEASURE, METRIC])[desired_column_order]

        for topics in ["closure", "predicted"]:
            topics_means_stddev_xs = unstacked_again_means_stddev.xs(key=topics, level=TOPICS).dropna().copy()
            topics_means_xs = unstacked_again_means.xs(key=topics, level=TOPICS).dropna().copy()
            topics_means_stddev_xs[("hmean", met, "F1")] = topics_means_xs[(col for col in topics_means_xs.columns if col[-1] == "F1")].apply(lambda ser: hmean(ser), axis=1).astype(str)

            out_file = output_path / f"meta-scores_metric-{met}_topics-{topics}_decimals-{decimals}.txt"
            with out_file.open("w") as f:
                f.write(tabulate(topics_means_stddev_xs.reset_index(), tablefmt="latex", headers="keys", showindex=False))