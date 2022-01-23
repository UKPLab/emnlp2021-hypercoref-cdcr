import unicodedata
from bisect import bisect_right
from logging import Logger
from string import digits, ascii_letters
from typing import Tuple, List, Dict, Callable, Optional, Any

import numpy as np


def clamp_global_span_to_tokenized_spans(global_span: Tuple[int, int], tokenized_spans: List[Tuple[int, int]]) -> Tuple[
    int, Tuple[int, int]]:
    """
    Assume the following scenario: You are given a paragraph of text and a character span (start, end) inside this
    paragraph. You sentence-tokenize the paragraph. Now, to which sentence does the character span belong and what
    are its character offsets inside the sentence? This method gives you the answer.
    In case the character span covers multiple sentence spans, this method will assign the character span to the
    first affected sentence, with the end of the span coinciding with the end of that sentence.
    Note: This method bears similarity to `allennlp.data.dataset_readers.reading_comprehension.util.char_span_to_token_span`.
    :param global_span: the start and end offsets of the span, with global offsets w.r.t. its surrounding paragraph (end
                        is exclusive!)
    :param tokenized_spans: the start and end offsets of the sentences (end offsets are exclusive!)
    :return: `(i, (start, end))` where `i` is the index of the sentence in which the start of the character span is
             located and `start`, `end` are the offsets of the span inside the sentence; clamped to the sentence
             boundary if necessary
    """

    start_global, end_global = global_span

    # i_sent is the sentence containing the start of the character span which we want to find
    i_sent = 1
    # Proceed as long as the currently examined sentence ends before the character span begins, or until there are
    # no more sentences to examine.
    while i_sent < len(tokenized_spans) and start_global >= tokenized_spans[i_sent - 1][1]:
        i_sent += 1

    # this happens if the span is somewhere inside the previous sentence
    if i_sent == len(tokenized_spans) or start_global < tokenized_spans[i_sent][0]:
        i_sent -= 1

    start_sentence, end_sentence = tokenized_spans[i_sent]

    # There is a special case where the start of global_span points to (typically whitespace) characters which lie
    # in between the end and start of the following tokenized span. In this case, we want to clamp the start of the
    # sentence to the beginning of the tokenized span which follows the in-between characters, i.e. 0.
    start_in_span = 0 if start_global < start_sentence else start_global - start_sentence

    # clamp the end of the span to the end of the sentence we assigned the global span to
    end_global_clamped = min(end_global, end_sentence)
    end_in_span = end_global_clamped - start_sentence

    return i_sent, (start_in_span, end_in_span)

# source: allennlp_models.rc.dataset_readers.utils.char_span_to_token_span
def char_span_to_token_span(token_offsets: List[Optional[Tuple[int, int]]],
                            character_span: Tuple[int, int],
                            logger: Logger) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).

    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.

    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.

    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether there was an error while matching the token spans exactly. If this is ``True``, it
        means there was an error in either the tokenization or the annotated character span. If this
        is ``False``, it means that we found tokens that match the character span exactly.
    """
    error = False
    start_index = 0
    while start_index < len(token_offsets) and (
        token_offsets[start_index] is None or token_offsets[start_index][0] < character_span[0]
    ):
        start_index += 1

    # If we overshot and the token prior to start_index ends after the first character, back up.
    if (
        start_index > 0
        and token_offsets[start_index - 1] is not None
        and token_offsets[start_index - 1][1] > character_span[0]
    ) or (
        start_index <= len(token_offsets)
        and token_offsets[start_index] is not None
        and token_offsets[start_index][0] > character_span[0]
    ):
        start_index -= 1

    if start_index >= len(token_offsets):
        raise ValueError("Could not find the start token given the offsets.")

    if token_offsets[start_index] is None or token_offsets[start_index][0] != character_span[0]:
        error = True

    end_index = start_index
    while end_index < len(token_offsets) and (
        token_offsets[end_index] is None or token_offsets[end_index][1] < character_span[1]
    ):
        end_index += 1
    if end_index == len(token_offsets):
        # We want a character span that goes beyond the last token. Let's see if this is salvageable.
        # We consider this salvageable if the span we're looking for starts before the last token ends.
        # In other words, we don't salvage if the whole span comes after the tokens end.
        if character_span[0] < token_offsets[-1][1]:
            # We also want to make sure we aren't way off. We need to be within 8 characters to salvage.
            if character_span[1] - 8 < token_offsets[-1][1]:
                end_index -= 1

    if end_index >= len(token_offsets):
        raise ValueError("Character span %r outside the range of the given tokens.")
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logger.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logger.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


def get_monotonous_character_alignment_func(orig: str, longer: str) -> Callable[[int], Optional[int]]:
    """
    Assume you detokenized a sequence of tokens and you applied span detection on the result. You now have character
    offsets into the detokenized sequence, but you want them for the original untokenized sequence. This method returns
    a function which, given a character offset into the detokenized version, returns the character offset in the
    original untokenized sequence. If the given offset does not have a corresponding character in the original sequence,
    `None` is returned. This offset does not necessarily need to conform to token boundaries, but there are other
    methods for fixing this.
    :param orig: `"".join(your_token_sequence)`
    :param longer: `detokenizer(your_token_sequence)` -> must contain the same characters as `orig` with additional
    ones in between! Otherwise, the result of this function is undefined.
    :return: function as described above
    """

    # TODO there might be computationally more efficient approaches, but this one works
    assert len(longer) > len(orig)

    checkpoints_orig = []
    checkpoints_longer = []

    idx_longer = 0
    for idx_orig in range(len(longer)):
        idx_orig_in_bounds = idx_orig < len(orig)

        does_char_match = idx_orig_in_bounds and longer[idx_longer] == orig[idx_orig]
        if does_char_match and len(checkpoints_longer) > 0:
            idx_longer += 1
        else:
            if not does_char_match:
                # create checkpoint for non-matching chars
                checkpoints_longer.append(idx_longer)
                checkpoints_orig.append(None)

                if not idx_orig_in_bounds:
                    # If we reach this point, we are in a situation where longer has superfluous characters at its end.
                    # We only need to create one last checkpoint for this case (which we have done above), therefore
                    # bail out.
                    break

                # advance index for longer until we find a matching char again
                while idx_longer < len(longer) and longer[idx_longer] != orig[idx_orig]:
                    idx_longer += 1

                # If we make it to this point, we have exhausted all characters in s_longer because we did not find a
                # match for the previous character in s. This means the strings are unalignable
                if idx_longer == len(longer):
                    raise ValueError

            # create checkpoint for matching pair
            checkpoints_longer.append(idx_longer)
            checkpoints_orig.append(idx_orig)
            idx_longer += 1

    assert checkpoints_longer[0] == 0   # the first char should always receive a checkpoint

    def func(i: int) -> Optional[int]:
        if i < 0 or i >= len(longer):
            raise IndexError

        # find the closest checkpoint preceding the given integer
        i_of_closest_preceding_checkpoint = bisect_right(checkpoints_longer, i) - 1

        aligned_value = checkpoints_orig[i_of_closest_preceding_checkpoint]
        if aligned_value is not None:
            # if there is a corresponding character in the shorter string, compute its exact index using the checkpoint
            distance_to_checkpoint = i - checkpoints_longer[i_of_closest_preceding_checkpoint]
            return aligned_value + distance_to_checkpoint
        else:
            # if there is no corresponding character, return None
            return None

    return func

# characters we want inside hyperlinks (at least those are the ones that we check at the boundaries)
WANTED_CHARS = digits + ascii_letters + '“‘\'"”’`´$€£#%*+@&'


def harmonize_span_boundary(sentence: str, idx: int, boundary: str, wanted_chars: str = WANTED_CHARS):
    # switch to inclusive indexing for the end span boundary
    if boundary == "end":
        idx -= 1

    if idx < 0 or idx >= len(sentence):
        print(f"Index out of range for sentence '{sentence}' at '{idx}', boundary was '{boundary}'.")
        return None

    if sentence[idx] in wanted_chars:
        # If correcting the start of the span: if there is a wanted character at the current index, go left as long as
        # there are more wanted chars. For the end of the span, move in the opposite direction.
        step = -1 if boundary == "start" else 1
        while idx + step > 0 and idx + step < len(sentence) and sentence[idx + step] in wanted_chars:
            idx += step
    else:
        # If correcting the start of the span: if there is an unwanted character at the current index, go right until
        # finding the first wanted char. For the end of the span, move in the opposite direction.
        step = 1 if boundary == "start" else -1
        while idx + step > 0 and idx + step < len(sentence) and sentence[idx] not in wanted_chars:
            idx += step

    # switch back to exclusive indexing for the end span boundary
    return idx + 1 if boundary == "end" else idx


# more utilities which require scipy
try:
    from scipy.optimize import linear_sum_assignment

    def span_matching(tagging_A: List[Tuple[int, int]],
                      tagging_B: List[Tuple[int, int]],
                      keep_A: bool = False) -> Dict[int, int]:
        """
        Assume we have a list of tokens which was tagged with spans by two different approaches A and B.
        This method tries to find the best 1:1 assignment of spans from B to spans from A. If there are more spans in A
        than in B, then spans from B will go unused and vice versa. The quality of an assignment between two spans
        depends on their overlap in tokens. This method removes entirely disjunct pairs of spans.
        Note: In case A contains two (or more) spans of the same length which are a single span in B (or vice versa),
        either of the spans from A may be mapped to the span in B. Which exact span from A is mapped is undefined.
        :param tagging_A: list of spans, defined by (start, end) token offsets (exclusive!), must be non-overlapping!
        :param tagging_B: a second list of spans over the same sequence in the same format as tagging_A
        :param keep_A: include unmatched spans from A as [idx_A, None] in the returned value
        :return: Dict[int,int] where keys are indices from A and values are indices from B
        """
        if not tagging_A:
            return {}
        elif not tagging_B:
            if keep_A:
                return {i:None for i in range(len(tagging_A))}
            else:
                return {}

        # Our cost function is span overlap:
        # (1) the basis: min(end indices) - max(start indices)
        # (2) If two spans are entirely disjunct, the result of (1) will be negative. Use max(0, ...) to set those
        #     cases to 0.
        # (3) High overlap should result in low costs, therefore multiply by -1
        overlap = lambda idx_a, idx_b: -1 * max(0,
                                                (min([tagging_A[idx_a][1],
                                                      tagging_B[idx_b][1]]) -
                                                 max([tagging_A[idx_a][0],
                                                      tagging_B[idx_b][0]])))
        cost_matrix = np.fromfunction(np.vectorize(overlap), (len(tagging_A), len(tagging_B)), dtype=np.int)    # type: np.ndarray
        a_indices, b_indices = linear_sum_assignment(cost_matrix)

        # throw away mappings which have no token overlap at all (i.e. costs == 0)
        assignment_costs = cost_matrix[a_indices, b_indices]
        valid_assignments = [i for i in range(len(a_indices)) if assignment_costs[i] < 0]

        # dropped_assignments = len(a_indices) - len(valid_assignments)
        # if dropped_assignments:
        #     self.logger.debug(f"Threw away {dropped_assignments} assignment without token overlap")

        # collect valid assignments
        assignments = {a_idx: b_idx for i, (a_idx, b_idx) in enumerate(zip(a_indices, b_indices)) if i in valid_assignments}

        if keep_A:
            a_to_none = {i: None for i in range(len(tagging_A))}
            a_to_none.update(assignments)
            assignments = a_to_none
        return assignments

except ImportError:
    pass