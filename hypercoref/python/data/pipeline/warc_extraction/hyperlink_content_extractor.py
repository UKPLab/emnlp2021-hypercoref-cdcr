from logging import Logger
from typing import List, Tuple, Dict

import numpy as np
from url_normalize import url_normalize

from python import SENTENCE_IDX, SENTENCE, CHARS_START, CHARS_END, TO_URL
from python.common_components.corenlp import CoreNlp
from python.util.ftfy import clean_string
from python.util.spans import clamp_global_span_to_tokenized_spans


class HyperlinkAndContentExtractor:

    def extract(self,
                top_node,
                corenlp: CoreNlp,
                logger: Logger,
                extract_full_content: bool = False) -> Tuple[List[Dict], List[Dict]]:
        sentences = []
        sentence_idx = 0
        hyperlinks = []

        for p in top_node.xpath("//p"):
            # find links in the paragraph
            a_in_p = []
            for a in p.xpath("a"):
                # The itertext() section below breaks if there are additional tags nested inside an <a> tag (cases like
                # <a>this is <i>the</i> link</a>). Replace each <a> tag's content with its pure text content to fix it.
                a_text = a.text_content()
                a_text = clean_string(a_text)

                # skip links without anchor text
                if not a_text.strip():
                    continue
                # remove all nested tags
                for elmt in a:
                    a.remove(elmt)
                a.text = a_text

                # without normalization, the pretty-printed URL and URL-encoded URL of the same page are considered to be different pages
                try:
                    a.attrib["href"] = url_normalize(a.get("href"))
                except (UnicodeError, ValueError, KeyError) as e:
                    # UnicodeError: sometimes URLs are malformed and do not comply with IDNA rules, so we skip it in those cases
                    # ValueError: some netlocs contain actual sentences (which triggers "netloc ... contains invalid characters under NFKC normalization")
                    # KeyError: "irc://..." URLs fail with KeyError because no default port is known
                    logger.warning(str(e))
                a_in_p.append(a)

            # obtain sentences
            if extract_full_content:
                # determine each tag's text content inside the paragraph, clean those, keep non-empty
                p_text_snippets = [c for c in map(lambda t: clean_string(t), p.itertext()) if c]

                # no text -> skip ahead as there will be no hyperlinks anyway
                if not p_text_snippets:
                    continue

                # determine the start offset of each text snippet - we need these to extract the character-level hyperlink spans
                p_text_snippets_lengths = [len(s) for s in p_text_snippets]
                p_text_snippets_start_offset = list(np.cumsum([0] + p_text_snippets_lengths[:-1]))

                # snippets by their start offset, in ascending order
                p_text_snippets_by_start_offset = list(zip(p_text_snippets_start_offset, p_text_snippets))

                # sentence tokenization - Punkt Sentence tokenizer is considerably faster than CoreNLP but produces
                # worse results, for example any occurrence of "St. Louis" is consistently split into two sentences
                p_text = "".join(p_text_snippets)
                annotation, exc = corenlp.parse_sentence(p_text, {"annotators": "ssplit",
                                                                  "tokenize.options": "ptb3Escaping=false"},
                                                         use_cache=False)
                if exc is not None:
                    logger.warning("Sentence tokenization failed.", exc)
                    continue
                sent_spans = [[anno.characterOffsetBegin, anno.characterOffsetEnd] for anno in annotation.sentence]
                for start, end in sent_spans:
                    sentences.append({SENTENCE_IDX: sentence_idx, SENTENCE: p_text[start:end]})
                    sentence_idx += 1
            else:
                sent_spans = None
                p_text_snippets_by_start_offset = None

            # find hyperlinks inside paragraph
            for a in a_in_p:
                to_url = a.get("href")
                if not to_url:  # skip empty hyperlinks (yes, these exist)
                    continue
                hyperlink = {TO_URL: to_url}

                if extract_full_content:
                    anchor_text = a.text_content()

                    # Look up the start char offset of the link's anchor text in the paragraph: Assuming the elements in
                    # a_in_p are ordered by the position of their appearance the document, we pop text snippets and
                    # their offset from the corresponding list. If a text snippets matches our link anchor_text, we
                    # have found its offset.
                    start_offset = None
                    while p_text_snippets_by_start_offset:
                        start_offset, text_snippet = p_text_snippets_by_start_offset.pop(0)
                        if text_snippet == anchor_text:
                            break
                    assert start_offset is not None, f"Failed to find start offset for '{anchor_text}' pointing to '{hyperlink[TO_URL]}'"
                    char_start_of_a_in_p = start_offset

                    try:
                        sent_idx_of_a_in_p, (char_start_of_a_in_sent, char_end_of_a_in_sent) = clamp_global_span_to_tokenized_spans((char_start_of_a_in_p, char_start_of_a_in_p + len(anchor_text)), sent_spans)
                    except IndexError as e:
                        # FIXME fix the actual problem once this shows up again
                        logger.warning(f"Global to tokenized span failed. sent_spans: {sent_spans}, start {char_start_of_a_in_p}, anchor len {len(anchor_text)}. Sentences: {' '.join(s[SENTENCE] for s in sentences)}", e)
                        continue

                    # we now know the sentence index inside the paragraph, compute the sent index on document-level
                    global_sent_idx_of_a = sentence_idx - len(sent_spans) + sent_idx_of_a_in_p

                    hyperlink.update({SENTENCE_IDX: global_sent_idx_of_a,
                                      CHARS_START: char_start_of_a_in_sent,
                                      CHARS_END: char_end_of_a_in_sent})    # remember that CHARS_END is exclusive!

                hyperlinks.append(hyperlink)
        return sentences, hyperlinks