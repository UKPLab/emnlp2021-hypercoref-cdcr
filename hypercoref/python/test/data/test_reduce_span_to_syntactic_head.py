import os
import unittest
from unittest import TestCase

import pandas as pd

from python import CHARS_START, CHARS_END
from python.common_components.corenlp import CoreNlp
from python.data.pipeline.postprocessing.hyperlinks import HyperlinksPostProcessingStage
from python.test.data.using_corenlp import CoreNlpTest
from python.util.util import get_logger


class TestReduceSpanToSyntacticHead(CoreNlpTest):

    @unittest.skip
    def test_reduce_span_to_syntactic_head(self):
        func = HyperlinksPostProcessingStage.reduce_spans

        sentences = ["This is a test!",
                     "too aggressive",
                     "run fast",
                     "the people there",
                     "'there is a quote",
                     "  "]                  # if we test an empty span with nothing to parse, we expect to get the exact same span back
        expected_starts = [10, 4, 0, 4, 7, 0]
        expected_ends = [14, 14, 3, 10, 9, 2]
        expected_spans = ["test", "aggressive", "run", "people", "is", "  "]

        # this is just for us as a sanity check
        for expected_span, (sent, start, end) in zip(expected_spans, zip(sentences, expected_starts, expected_ends)):
            actual_span = sent[start:end]
            assert expected_span == actual_span

        result = func(pd.Series(sentences), self.corenlp)
        assert len(result) == len(sentences)
        for i, (expected_start, expected_end) in enumerate(zip(expected_starts, expected_ends)):
            actual_start = result.iloc[i][CHARS_START]
            actual_end = result.iloc[i][CHARS_END]
            assert expected_start == actual_start
            assert expected_end == actual_end