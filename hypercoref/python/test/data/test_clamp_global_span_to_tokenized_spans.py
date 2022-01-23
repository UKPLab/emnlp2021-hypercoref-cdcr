from unittest import TestCase

from nltk.tokenize import PunktSentenceTokenizer

from python.util.spans import clamp_global_span_to_tokenized_spans


class TestClampGlobalSpanToTokenizedSpan(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.pst = PunktSentenceTokenizer()

        # function under test
        self.f = clamp_global_span_to_tokenized_spans

    def test_single_sentence(self) -> None:
        sent = "This is just a lonely sentence."
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        character_span = (0, 4)
        assert sent[character_span[0]:character_span[1]] == "This"

        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 0
        assert end == 4
        # -------------------------

        character_span = (2, 7)
        assert sent[character_span[0]:character_span[1]] == "is is"

        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 2
        assert end == 7
        # -------------------------

        character_span = (22, 31)
        assert sent[character_span[0]:character_span[1]] == "sentence."

        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 22
        assert end == 31

    def test_two_sentences_easy(self) -> None:
        sent = "This is just a lonely sentence. This is a second one."
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        character_span = (22, 31)
        assert sent[character_span[0]:character_span[1]] == "sentence."
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 22
        assert end == 31
        # -------------------------

        # check the first word from the second sentence
        character_span = (32, 36)
        assert sent[character_span[0]:character_span[1]] == "This"
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 1
        assert start == 0
        assert end == 4
        # -------------------------

    def test_two_sentences_no_whitespace(self) -> None:
        sent = "This is just a lonely sentence.Here is a second one, surprise!"
        tokenized_spans = [(0,31), (31,62)]
        assert sent[tokenized_spans[0][0]:tokenized_spans[0][1]] == "This is just a lonely sentence."
        assert sent[tokenized_spans[1][0]:tokenized_spans[1][1]] == "Here is a second one, surprise!"
        # -------------------------

        character_span = (22, 31)
        assert sent[character_span[0]:character_span[1]] == "sentence."
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 22
        assert end == 31
        # -------------------------

        character_span = (31, 35)
        assert sent[character_span[0]:character_span[1]] == "Here"
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 1
        assert start == 0
        assert end == 4


    def test_clamp_to_first_sentence(self) -> None:
        sent = "This is just a lonely sentence. This is a second one."
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        # check the last word from the first sentence, but extend the span by one character towards the second sentence -> the span should be truncated at the sentence boundary
        character_span = (22, 32)
        assert sent[character_span[0]:character_span[1]] == "sentence. "
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 22
        assert end == 31
        # -------------------------

        # take it a bit further and include more characters from the second sentence -> should be the same result as above
        character_span = (22, 35)
        assert sent[character_span[0]:character_span[1]] == "sentence. Thi"
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 22
        assert end == 31
        # -------------------------

        # this should also work if there is an additional space between the sentences
        sent = "This is just a lonely sentence.  This is a second one."
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        character_span = (22, 33)
        assert sent[character_span[0]:character_span[1]] == "sentence.  "
        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 22
        assert end == 31


    def test_clamp_whitespace_to_second_sentence(self) -> None:
        sent = "This is a normal sentence.   This one has many spaces before it."
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        # This is a challenging case where the character span should belong to the second sentence, but it includes whitespace from between the second and first sentence.
        character_span = (28, 37)
        assert sent[character_span[0]:character_span[1]] == " This one"

        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 1
        assert start == 0
        assert end == 8
        # -------------------------

        # this should also work with just one preceding space
        sent = "Another sentence. There is one space there, got it?"
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        character_span = (17, 30)
        assert sent[character_span[0]:character_span[1]] == " There is one"

        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 1
        assert start == 0
        assert end == 12

    def test_three_sentences(self) -> None:
        sent = "Sentence number one. And sentence number two. And three, mhh."
        tokenized_spans = list(self.pst.span_tokenize(sent))
        # -------------------------

        # this should be truncated after the first sentence
        character_span = (16, 49)
        assert sent[character_span[0]:character_span[1]] == "one. And sentence number two. And"

        i, (start, end) = self.f(character_span, tokenized_spans)
        assert i == 0
        assert start == 16
        assert end == 20