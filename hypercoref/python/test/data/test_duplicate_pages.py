from unittest import TestCase

import pandas as pd

from python.data.pipeline.postprocessing.page_deduplication.lsh import LshIdentifyPageDuplicates
from python.data.pipeline.postprocessing.page_deduplication.tfidf import TfidfIdentifyPageDuplicates
from python.data.pipeline.postprocessing.page_deduplication.two_stage import TwoStageIdentifyPageDuplicates
from python.test.data import FIXTURES_ROOT


class TestDeduplicatePages(TestCase):
    # This file contains:
    # - two identical pages
    # - two pages which differ by one sentence
    # - four different pages which share one sentence
    # - eight further unique pages
    # Hence, we expect 14 clusters.
    CHALLENGE = pd.read_csv(FIXTURES_ROOT / "page_deduplication_challenge.csv", index_col=0, squeeze=True)
    SOLUTION = pd.read_csv(FIXTURES_ROOT / "page_deduplication_solution.csv", index_col=0, squeeze=True)

    def _compare(self, actual):
        # make series comparable (same order, re-create cluster IDs)
        solution = TestDeduplicatePages.SOLUTION.sort_index()
        solution = pd.Series(solution.factorize()[0], index=solution.index)

        assert actual.equals(solution)

    def test_tfidf(self):
        deduplifier = TfidfIdentifyPageDuplicates(max_chars=None, threshold=0.99, n_gram=11)
        deduplifier.fit(TestDeduplicatePages.CHALLENGE)
        clustering = deduplifier.find_duplicates(TestDeduplicatePages.CHALLENGE)
        self._compare(clustering)

    def test_lsh(self):
        deduplifier = LshIdentifyPageDuplicates(max_chars=None, seeds=100, bands=10, char_n_gram=5)
        deduplifier.fit(TestDeduplicatePages.CHALLENGE)  # nop
        clustering = deduplifier.find_duplicates(TestDeduplicatePages.CHALLENGE)
        self._compare(clustering)

    def test_two_stage(self):
        # this is mostly a test whether it runs at all, not for parameter tuning
        deduplifier = TwoStageIdentifyPageDuplicates(
            lsh={"max_chars": None, "seeds": 99, "bands": 33, "char_n_gram": 5},
            tfidf={"max_chars": None, "threshold": 0.99, "n_gram": 11}, num_docs_to_fit_tfidf_on=5)

        deduplifier.fit(TestDeduplicatePages.CHALLENGE)
        clustering = deduplifier.find_duplicates(TestDeduplicatePages.CHALLENGE)

        self._compare(clustering)