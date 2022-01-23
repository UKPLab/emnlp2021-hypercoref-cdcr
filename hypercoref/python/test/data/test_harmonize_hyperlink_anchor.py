from unittest import TestCase

from python.util.spans import harmonize_span_boundary


class TestHarmonizeSpanBoundary(TestCase):

    def test_harmonize_span_boundary(self):
        func = harmonize_span_boundary

        sent = "This is quite the test"
        start = 10  # i in "quite"
        end = 14  # "t" of "the" after "quite"
        start_corrected = func(sent, start, boundary="start")
        assert sent[start_corrected:end] == "quite "
        end_corrected = func(sent, end, boundary="end")
        assert sent[start:end_corrected] == "ite"

        start = 4  # space before "is"
        end = 6  # "s" in "is"
        start_corrected = func(sent, start, boundary="start")
        assert sent[start_corrected:end] == "i"
        end_corrected = func(sent, end, boundary="end")
        assert sent[start:end_corrected] == " is"

        # "This" at the start - nothing should change
        start = 0
        end = 4
        start_corrected = func(sent, start, boundary="start")
        assert start_corrected == start
        end_corrected = func(sent, end, boundary="end")
        assert end_corrected == end

        # "the" - nothing should change
        start = 14
        end = 17
        start_corrected = func(sent, start, boundary="start")
        assert start_corrected == start
        end_corrected = func(sent, end, boundary="end")
        assert end_corrected == end

        # "test" at the end - nothing should change
        start = 18
        end = 22
        start_corrected = func(sent, start, boundary="start")
        assert start_corrected == start
        end_corrected = func(sent, end, boundary="end")
        assert end_corrected == end