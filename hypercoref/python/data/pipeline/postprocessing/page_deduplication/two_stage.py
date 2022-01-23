import tempfile
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from joblib import delayed, Parallel
from overrides import overrides
from tqdm import tqdm

from python.data.pipeline.postprocessing.page_deduplication.base import IdentifyPageDuplicates
from python.data.pipeline.postprocessing.page_deduplication.lsh import LshIdentifyPageDuplicates
from python.data.pipeline.postprocessing.page_deduplication.tfidf import TfidfIdentifyPageDuplicates
from python.util.progressbars import tqdm_joblib


class TwoStageIdentifyPageDuplicates(IdentifyPageDuplicates):
    """
    Performs LSH as a first rough pass, then runs TF-IDF on the candidates found by LSH (in parallel) as a second
    deduplication pass.
    TODO The two-stage approach gives great results, but is not particularly sound from an efficiency point of view.
    """
    def __init__(self,
                 lsh: Dict,
                 tfidf: Dict,
                 num_docs_to_fit_tfidf_on: int = 2500,
                 max_cores: int = None):
        self.lsh = LshIdentifyPageDuplicates(**lsh)
        self.tfidf = TfidfIdentifyPageDuplicates(**tfidf)
        self.num_docs_to_fit_tfidf_on = num_docs_to_fit_tfidf_on
        self.max_cores = max_cores

    @overrides
    def _find_duplicates(self, documents: pd.Series):
        clustering = self.lsh.find_duplicates(documents)

        # find non-singleton clusters and run the fine pass on those
        cluster_sizes = clustering.value_counts()
        non_singleton_clusters = cluster_sizes.loc[cluster_sizes > 1]

        with tempfile.TemporaryDirectory(prefix="twostage") as dir:
            # write fitted TFIDF deduplifier file for processes to read
            d = Path(dir)
            deduplifier_file = d / "fitted_tfidf_deduplifier.pkl"
            joblib.dump(self.tfidf, deduplifier_file)

            # generator which writes parts of the documents dataframe to disk, for processes to read
            def generate_clusters():
                for cluster_id in non_singleton_clusters.index.values:
                    rough_documents = documents.loc[clustering.loc[clustering == cluster_id].index]
                    destination = d / f"cluster_{cluster_id}.pkl"
                    joblib.dump(rough_documents, destination)
                    yield destination

            def fine_deduplicate(deduplifier_file: Path, data_file: Path):
                deduplifier = joblib.load(deduplifier_file)
                data = joblib.load(data_file)
                fine_cluster =  deduplifier.find_duplicates(data)
                # clean up right away so that we don't overburden the filesystem with a folder that has 200k files...
                data_file.unlink()
                return fine_cluster

            with tqdm_joblib(tqdm(desc="Deduplicate LSH clusters",
                                  unit="cluster",
                                  mininterval=10,
                                  total=len(non_singleton_clusters.index.values))) as bar:
                fine_clusters = Parallel(n_jobs=self.max_cores)(delayed(fine_deduplicate)(deduplifier_file, data_file) for data_file in generate_clusters())

            for fine_cluster in tqdm(fine_clusters,
                                     desc="Merging TF-IDF clusters with LSH clusters",
                                     unit="cluster",
                                     mininterval=10):
                # advance cluster IDs and update final clustering
                highest_unused_cluster_id = clustering.max() + 1
                fine_cluster += highest_unused_cluster_id
                clustering.update(fine_cluster)

        return clustering

    @overrides
    def fit(self, documents: pd.Series):
        # fit TFIDF deduplication strategy
        tfidf_docs = documents.sample(min(len(documents), self.num_docs_to_fit_tfidf_on), random_state=0)
        self.tfidf.fit(tfidf_docs)