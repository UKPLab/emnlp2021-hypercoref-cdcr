import logging
from unittest import TestCase

import lxml.html
import numpy.testing as testing
import pandas as pd

from python import SENTENCE_IDX, SENTENCE, CHARS_START, CHARS_END, TO_URL
from python.common_components.corenlp import CoreNlp
from python.data.pipeline.warc_extraction.hyperlink_content_extractor import HyperlinkAndContentExtractor
from python.test.data import FIXTURES_ROOT
from python.test.data.using_corenlp import CoreNlpTest


class TestHyperlinkAndContentExtractor(CoreNlpTest):

    def test_extract(self):
        with open(FIXTURES_ROOT / "sentence_split_span_extract.html") as f:
            markup = f.read()
        root = lxml.html.document_fromstring(markup)

        extractor = HyperlinkAndContentExtractor()

        sentences, hyperlinks = extractor.extract(root, self.corenlp, self.logger, extract_full_content=True)

        sentences = pd.DataFrame(sentences)
        hyperlinks = pd.DataFrame(hyperlinks)

        sentences_expected = pd.DataFrame(
            [{SENTENCE_IDX: 0, SENTENCE: "This is a test with a hyperlink in the middle of a paragraph."},
             {SENTENCE_IDX: 1, SENTENCE: "Here is another paragraph."},
             {SENTENCE_IDX: 2, SENTENCE: "It has three sentences with the middle one starting with a hyperlink starting with a space."},
             {SENTENCE_IDX: 3, SENTENCE: "In the same paragraph, there is another link, wow right?"},
             {SENTENCE_IDX: 4, SENTENCE: "This paragraph is a special case, because the link has nested tags!"},
             {SENTENCE_IDX: 5, SENTENCE: "Spooky."}
             ])
        hyperlinks_expected = pd.DataFrame([{SENTENCE_IDX: 0, CHARS_START: 15, CHARS_END: 31, TO_URL: "http://example.com/"},
                                            {SENTENCE_IDX: 2, CHARS_START: 51, CHARS_END: 90, TO_URL: "http://example.com/example"},
                                            {SENTENCE_IDX: 3, CHARS_START: 32, CHARS_END: 44, TO_URL: "http://example.com/example2"},
                                            {SENTENCE_IDX: 4, CHARS_START: 42, CHARS_END: 66, TO_URL: "http://example.com/something"}])
        for col in [SENTENCE_IDX, SENTENCE]:
            testing.assert_array_equal(sentences[col], sentences_expected[col])
        for col in [SENTENCE_IDX, CHARS_START, CHARS_END, TO_URL]:
            testing.assert_array_equal(hyperlinks[col], hyperlinks_expected[col])

    def test_unicode_control_characters(self):
        # here, the challenge lies in removing the soft hyphen before "Barcelona" and still getting all the spans right

        with open(FIXTURES_ROOT / "unicode_control_characters.html") as f:
            markup = f.read()
        root = lxml.html.document_fromstring(markup)

        extractor = HyperlinkAndContentExtractor()

        sentences, hyperlinks = extractor.extract(root, self.corenlp, self.logger, extract_full_content=True)
        sentences = pd.DataFrame(sentences)
        hyperlinks = pd.DataFrame(hyperlinks)

        sentences_expected = pd.DataFrame(
            [{SENTENCE_IDX: 0, SENTENCE: "And former Ajax and Barcelona midfielder De Boer believes something."},
             {SENTENCE_IDX: 1, SENTENCE: "Here is another paragraph."}])
        hyperlinks_expected = pd.DataFrame(
            [{SENTENCE_IDX: 0, CHARS_START: 11, CHARS_END: 48, TO_URL: "http://destination.com/"}])

        for col in [SENTENCE_IDX, SENTENCE]:
            testing.assert_array_equal(sentences[col], sentences_expected[col])
        for col in [SENTENCE_IDX, CHARS_START, CHARS_END, TO_URL]:
            testing.assert_array_equal(hyperlinks[col], hyperlinks_expected[col])


    def test_repeated_snippets_inside_paragraph(self):
        # There were issues in the past with:
        #   1) pages having repeated text snippets (parts of a <p> or <a>). Those were being overwritten in the
        #      extractor implementation
        #   2) hyperlink spans which exceed detected sentence boundaries
        # This test tests both cases.

        with open(FIXTURES_ROOT / "repeated_snippets_inside_paragraph.html") as f:
            markup = f.read()
        root = lxml.html.document_fromstring(markup)

        extractor = HyperlinkAndContentExtractor()

        sentences, hyperlinks = extractor.extract(root, self.corenlp, self.logger, extract_full_content=True)
        sentences = pd.DataFrame(sentences)
        hyperlinks = pd.DataFrame(hyperlinks)

        sentences_expected = pd.DataFrame(
            [{SENTENCE_IDX: 0, SENTENCE: "Here is a link."},
             {SENTENCE_IDX: 1, SENTENCE: "Here is another link."}])
        hyperlinks_expected = pd.DataFrame(
            [{SENTENCE_IDX: 0, CHARS_START: 8, CHARS_END: 15, TO_URL: "http://destination.com/"},
             {SENTENCE_IDX: 1, CHARS_START: 8, CHARS_END: 21, TO_URL: "http://destination.com/bla"}])

        for col in [SENTENCE_IDX, SENTENCE]:
            testing.assert_array_equal(sentences[col], sentences_expected[col])
        for col in [SENTENCE_IDX, CHARS_START, CHARS_END, TO_URL]:
            testing.assert_array_equal(hyperlinks[col], hyperlinks_expected[col])

    # TODO test with extract_full_content=False
