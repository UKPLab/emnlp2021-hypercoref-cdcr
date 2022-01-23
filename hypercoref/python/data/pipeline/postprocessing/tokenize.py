from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from python import *
from python.common_components import CORENLP
from python.common_components.corenlp import CoreNlp
from python.pipeline.pipeline import PipelineStage
from python.util.spans import char_span_to_token_span
from python.util.util import write_dataframe, read_dataframe


class TokenizeStage(PipelineStage):
    """
    This stage tokenizes all input sentences and maps hyperlinks to token level.
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.page_infos_file = None
        self.hyperlinks_file = None
        self.sentences_file = None
        self.page_infos_tokenized_file = self.stage_disk_location / "page_infos"
        self.hyperlinks_tokenized_file = self.stage_disk_location / "hyperlinks"
        self.tokens_tokenized_file = self.stage_disk_location / "tokens"
        self.sentences_tokenized_file = self.stage_disk_location / "sentences"

        self.tail_pages_keep_first_n_sentences = config.get("tail_pages_keep_first_n_sentences", 5)

    def requires_files(self, provided: Dict[str, Path]):
        self.page_infos_file = provided[PAGE_INFOS]
        self.hyperlinks_file = provided[HYPERLINKS]
        self.sentences_file = provided[SENTENCES]

    def files_produced(self) -> Dict[str, Path]:
        # replace old index
        return {HYPERLINKS: self.hyperlinks_tokenized_file,
                TOKENS: self.tokens_tokenized_file,
                SENTENCES: self.sentences_tokenized_file,
                PAGE_INFOS: self.page_infos_tokenized_file}

    def run(self, live_objects: dict):
        corenlp = live_objects[CORENLP]  # type: CoreNlp

        page_infos = read_dataframe(self.page_infos_file)  # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)  # type: pd.DataFrame
        content_sentence_level = read_dataframe(self.sentences_file)    # type: pd.DataFrame

        # keeping the full text of tail pages (out-degree > 0, in-degree = 0) makes little sense - keep at most the
        # first n sentences of each
        if self.tail_pages_keep_first_n_sentences is not None and self.tail_pages_keep_first_n_sentences >= 0:
            pages_in = set(hyperlinks[TO_URL_NORMALIZED].unique())
            pages_out = set(hyperlinks[URL_NORMALIZED].unique())
            pages_scraped = set(page_infos.index.unique())
            scraped_tail_pages = (pages_in - pages_out) & pages_scraped

            sentences_to_drop = content_sentence_level.index.map(lambda tup: tup[0] in scraped_tail_pages and tup[1] >= self.tail_pages_keep_first_n_sentences)
            content_sentence_level.drop(content_sentence_level.index[sentences_to_drop], inplace=True)
            self.logger.info(f"{sentences_to_drop.values.sum()} sentences from tail pages were dropped and will not be tokenized. {len(content_sentence_level)} sentences remain.")

        hyperlinks = hyperlinks.sort_values(by=[URL_NORMALIZED, SENTENCE_IDX, CHARS_START])
        hyperlinks_token_level = []
        content_token_level = []

        def documents():
            for _, doc_sentences in content_sentence_level.groupby(URL_NORMALIZED):
                yield "\n".join(doc_sentences[SENTENCE].values)

        properties = {"annotators": "ssplit,tokenize",
                      "tokenize.options": "ptb3Escaping=false",
                      "ssplit.eolonly": True}
        doc_urls = (url for url, _ in content_sentence_level.groupby(URL_NORMALIZED))
        doc_annos = corenlp.parse_strings(documents(), properties=properties, use_cache=False)
        doc_urls_failed = []
        for url, annotations in tqdm(zip(doc_urls, doc_annos),
                                     desc="Tokenizing documents",
                                     unit="doc",
                                     mininterval=10,
                                     total=content_sentence_level.index.get_level_values(URL_NORMALIZED).unique().size):
            if annotations is None:
                self.logger.warning(f"Failed tokenizing {url}.")
                doc_urls_failed.append(url)
                continue

            tokens = []
            for sentence in annotations.sentence:
                # CoreNLP might tell us there is more than 1 sentence in the given string. We ignore that and trust our
                # own previously applied sentence tokenization. Therefore, tokens are counted across the corenlp
                # sentence boundaries here. TODO just use the setting which forbids CoreNLP from splitting sentences
                sent_tokens = []
                token_idx = 0
                for token in sentence.token:
                    start = token.beginChar - sentence.characterOffsetBegin
                    end = token.endChar - sentence.characterOffsetBegin
                    sent_tokens.append({TOKEN_IDX: token_idx,
                                        TOKEN: token.word,
                                        URL_NORMALIZED: url,
                                        SENTENCE_IDX: sentence.sentenceIndex,
                                        CHARS_START: start,
                                        CHARS_END: end})
                    token_idx += 1
                tokens += sent_tokens

            # combine tokens and mention info into new dataframes
            tokens = pd.DataFrame(tokens).set_index([URL_NORMALIZED, SENTENCE_IDX, TOKEN_IDX]).sort_index()

            num_sentences_before = content_sentence_level.loc[url].size
            num_sentences_after = tokens.index.unique(SENTENCE_IDX).size
            if num_sentences_before != num_sentences_after:
                self.logger.warning(f"Document {url}: number of sentences changed after tokenization ({num_sentences_before} before, {num_sentences_after} after). Document will be skipped.")
                doc_urls_failed.append(url)
                continue

            # convert hyperlink anchors from character spans to token spans
            hyperlinks_in_doc = hyperlinks.loc[hyperlinks[URL_NORMALIZED] == url]
            for index, row in hyperlinks_in_doc.iterrows():
                tokens_of_sentence = tokens.loc[(url, row[SENTENCE_IDX])]

                # Some strange sentences (for example some consisting entirely of URLs) can trip CoreNLP, leading to
                # token sequences which in sum are shorter than their single-string sentence counterparts. This breaks
                # the char to token span conversion, hence we check this here: tokenization is invalid and breaks
                # conversion if the end of the hyperlink span is out of bounds w.r.t. the token offsets.
                is_tokenization_invalid = row[CHARS_END] > tokens_of_sentence[CHARS_END].iloc[-1]
                if is_tokenization_invalid:
                    continue

                # set up token offsets with inclusive start and end
                token_offsets = pd.concat([tokens_of_sentence[CHARS_START], tokens_of_sentence[CHARS_END]], axis=1)
                (begin, end_inclusive), error = char_span_to_token_span(token_offsets.values,
                                                                        (row[CHARS_START], row[CHARS_END]),
                                                                        self.logger)
                if not error:
                    # go back to exclusive offsets
                    hyperlinks_token_level.append({"index": index, TOKEN_IDX_FROM: begin, TOKEN_IDX_TO: end_inclusive + 1})
            content_token_level.append(tokens[TOKEN])

        # assemble final dataframes
        self.logger.info(f"{len(hyperlinks) - len(hyperlinks_token_level)} hyperlinks had problematic spans after tokenization and were removed ({len(hyperlinks_token_level)} remain).")
        hyperlinks_token_level = pd.DataFrame(hyperlinks_token_level).set_index("index")
        hyperlinks_token_level.index.name = None
        # inner merge to drop hyperlinks for which tokenization caused mismatching spans
        hyperlinks = hyperlinks.merge(hyperlinks_token_level, left_index=True, right_index=True, how="inner")
        content_token_level = pd.concat(content_token_level)

        # drop documents where tokenization failed
        if doc_urls_failed:
            self.logger.warning(f"Tokenization failed for {len(doc_urls_failed)} documents (out of {len(page_infos)} total). Affected data will be removed.")
            page_infos.drop(index=doc_urls_failed, inplace=True)
            content_sentence_level.drop(index=doc_urls_failed, level=URL_NORMALIZED, inplace=True)
            content_token_level.drop(index=doc_urls_failed, level=URL_NORMALIZED, inplace=True, errors="ignore")
            hyperlinks = hyperlinks.loc[~hyperlinks[URL_NORMALIZED].isin(doc_urls_failed)]

        # write to disk
        write_dataframe(page_infos, self.page_infos_tokenized_file)
        write_dataframe(hyperlinks, self.hyperlinks_tokenized_file)
        write_dataframe(content_sentence_level, self.sentences_tokenized_file)
        write_dataframe(content_token_level.to_frame(TOKEN), self.tokens_tokenized_file)


component = TokenizeStage
