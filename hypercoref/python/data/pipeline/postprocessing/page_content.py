import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from python import *
from python.pipeline.pipeline import PipelineStage
from python.util.util import write_dataframe, read_dataframe


class PostProcessPageContentStage(PipelineStage):
    """
    This stage removes unwanted sentences from pages.
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.page_infos_file = None
        self.hyperlinks_file = None
        self.sentences_file = None
        self.page_infos_postprocessed_file = self.stage_disk_location / "page_infos_post_processed"
        self.hyperlinks_postprocessed_file = self.stage_disk_location / "hyperlinks_post_processed"
        self.sentences_postprocessed_file = self.stage_disk_location / "sentences_post_processed"

        self.strip_sentences = config.get("strip_sentences", True)

        if "sentence_blacklist_regexes" in config:
            # add all string values pandas might interpret as NaN later on (see below)
            all_regexes = config["sentence_blacklist_regexes"] + [r"^-NaN$", r"^#NA$", r"^<NA>$", r"^NA$", r"^NaN$",
                                                                  r"^NULL$", r"^-1\.#IND$", r"^-1\.#QNAN$",
                                                                  r"^#N\/A N\/A$", r"^#N\/A$", r"^1\.#IND$",
                                                                  r"^1\.#QNAN$", r"^N\/A$"]
            all_in_one = "|".join(f"({regex})" for regex in all_regexes)
            self.sentence_blacklist_regex = re.compile(all_in_one, flags=re.IGNORECASE)
        else:
            self.sentence_blacklist_regex = None

    def requires_files(self, provided: Dict[str, Path]):
        self.page_infos_file = provided[PAGE_INFOS]
        self.hyperlinks_file = provided[HYPERLINKS]
        self.sentences_file = provided[SENTENCES]

    def files_produced(self) -> Dict[str, Path]:
        # replace old index
        return {PAGE_INFOS: self.page_infos_postprocessed_file,
                HYPERLINKS: self.hyperlinks_postprocessed_file,
                SENTENCES: self.sentences_postprocessed_file}

    def _apply_blacklist(self, page_infos, hyperlinks, sentences) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert self.sentence_blacklist_regex is not None

        is_sentence_blacklisted = sentences[SENTENCE].map(lambda a: self.sentence_blacklist_regex.match(a) is not None)
        num_sentences_to_remove = is_sentence_blacklisted.sum()
        self.logger.info(
            f"{num_sentences_to_remove} sentences of {len(sentences)} in total are blacklisted and will be removed.")

        if num_sentences_to_remove:
            # remove any hyperlinks we had in those blacklisted sentences (reset_index is necessary to keep the index of
            # the hyperlinks dataframe despite the merge operation)
            blacklisted_sentences = is_sentence_blacklisted.loc[is_sentence_blacklisted]
            hyperlinks_in_blacklisted_sents = hyperlinks.reset_index().merge(blacklisted_sentences,
                                                                             on=[URL_NORMALIZED, SENTENCE_IDX])
            hyperlinks.drop(hyperlinks_in_blacklisted_sents["index"].values, inplace=True)

            # determine subset of pages with blacklisted sentences
            is_page_affected = is_sentence_blacklisted.groupby(URL_NORMALIZED).any()
            affected_pages = is_page_affected.loc[is_page_affected].index
            sentences_of_affected_pages = sentences.loc[affected_pages]
            hyperlinks_of_affected_pages = hyperlinks.loc[hyperlinks[URL_NORMALIZED].isin(affected_pages)]

            # within this subset of pages, filter sentences and update the sentence indices in the hyperlinks dataframe
            not_blacklisted_sentences = sentences_of_affected_pages.loc[~is_sentence_blacklisted].copy()

            # We now need to update the sentence indices so they're ascending without gaps again. This change needs to
            # be replicated in the hyperlinks dataframe.
            # Set up a dataframe mapping from old to new sentence IDs: use reset_index(drop=True) and groupby to obtain
            # a continuous ascending index for each sentence per document, and keep the old indices
            mapping_new_to_old = not_blacklisted_sentences.index.to_frame(index=False).groupby(URL_NORMALIZED).apply(
                lambda df: df.reset_index(drop=True))
            mapping_new_to_old.drop(columns=URL_NORMALIZED, inplace=True)
            mapping_new_to_old.index.names = [URL_NORMALIZED, "sentence-idx-new"]
            # this mapping is still the wrong way around (it maps new IDs to old IDs), so let's flip it
            mapping_old_to_new = mapping_new_to_old.reset_index().set_index([URL_NORMALIZED, SENTENCE_IDX])

            # map all sentence indices of affected pages in the hyperlinks dataframe to the new values
            hyperlinks_updated_sentence_idx = hyperlinks_of_affected_pages[[URL_NORMALIZED, SENTENCE_IDX]].apply(
                lambda row: mapping_old_to_new.at[tuple(row.values), "sentence-idx-new"], axis="columns")
            hyperlinks.loc[hyperlinks_updated_sentence_idx.index, SENTENCE_IDX] = hyperlinks_updated_sentence_idx

            # apply sentence index remapping to sentences dataframe
            not_blacklisted_sentences.reset_index(inplace=True)
            not_blacklisted_sentences[SENTENCE_IDX] = not_blacklisted_sentences[[URL_NORMALIZED, SENTENCE_IDX]].apply(
                lambda row: mapping_old_to_new.at[tuple(row.values), "sentence-idx-new"], axis="columns")
            not_blacklisted_sentences.set_index([URL_NORMALIZED, SENTENCE_IDX], inplace=True)
            sentences_of_unaffected_pages = sentences.loc[
                sentences.index.get_level_values(URL_NORMALIZED).difference(affected_pages)]
            sentences = pd.concat([sentences_of_unaffected_pages, not_blacklisted_sentences])

            # remove documents entirely which had all of their sentences removed
            page_infos = page_infos.loc[sentences.index.unique(URL_NORMALIZED)]
        return page_infos, hyperlinks, sentences

    def _strip_sentences(self, page_infos, hyperlinks, sentences) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # identify sentences needing lstrip
        does_sentence_need_lstrip = sentences[SENTENCE].str.match(r"^\s+")
        if does_sentence_need_lstrip.sum():
            # lstrip and count number of characters stripped for each sentence
            len_before_strip = sentences.loc[does_sentence_need_lstrip, SENTENCE].str.len()
            sentences.loc[does_sentence_need_lstrip, SENTENCE] = sentences.loc[does_sentence_need_lstrip, SENTENCE].str.lstrip()
            num_chars_stripped = len_before_strip - sentences.loc[does_sentence_need_lstrip, SENTENCE].str.len()

            # for all hyperlinks in affected sentences: update char offsets accordingly
            # (1) take hyperlinks which need to be adjusted now: use reset_index to retain each hyperlink's original
            #     index before merging
            affected_hyperlinks = hyperlinks.reset_index().merge(num_chars_stripped.to_frame("chars-stripped"), on=[URL_NORMALIZED, SENTENCE_IDX]).set_index("index")
            affected_hyperlinks[CHARS_START] = (affected_hyperlinks[CHARS_START] - affected_hyperlinks["chars-stripped"]).clip(lower=0)
            affected_hyperlinks[CHARS_END] = (affected_hyperlinks[CHARS_END] - affected_hyperlinks["chars-stripped"]).clip(lower=0)
            # (2) write back the changes
            hyperlinks.loc[affected_hyperlinks.index, [CHARS_START, CHARS_END]] = affected_hyperlinks[[CHARS_START, CHARS_END]]

            self.logger.info(f"{does_sentence_need_lstrip.sum()} sentences were lstripped, and {len(affected_hyperlinks)} hyperlinks were modified accordingly.")

        does_sentence_need_rstrip = sentences[SENTENCE].str.match(r".*\s+$")
        if does_sentence_need_rstrip.sum():
            # rstripping is easier, we only have to update the end indices of hyperlinks
            sentences.loc[does_sentence_need_rstrip, SENTENCE] = sentences.loc[does_sentence_need_rstrip, SENTENCE].str.rstrip()
            len_after_strip = sentences.loc[does_sentence_need_rstrip, SENTENCE].str.len()

            affected_hyperlinks = hyperlinks.reset_index().merge(len_after_strip.to_frame("len"), on=[URL_NORMALIZED, SENTENCE_IDX]).set_index("index")
            affected_hyperlinks[CHARS_START] = affected_hyperlinks[CHARS_START].clip(upper=affected_hyperlinks["len"])
            affected_hyperlinks[CHARS_END] = affected_hyperlinks[CHARS_END].clip(upper=affected_hyperlinks["len"])
            # write back the changes
            hyperlinks.loc[affected_hyperlinks.index, [CHARS_START, CHARS_END]] = affected_hyperlinks[[CHARS_START, CHARS_END]]

            self.logger.info(f"{does_sentence_need_rstrip.sum()} sentences were rstripped, and {len(affected_hyperlinks)} hyperlinks were modified accordingly.")

        # At this point, hyperlink spans which were located entirely in stripped whitespace would have identical start/
        # end due to the clipping (i.e. a span of zero). Remove those:
        if does_sentence_need_lstrip.sum() or does_sentence_need_rstrip.sum():
            hyperlinks = hyperlinks.loc[hyperlinks[CHARS_START] < hyperlinks[CHARS_END]]

        # assert that hyperlinks are healthy: all spans within bounds
        assert (hyperlinks[CHARS_START] >= 0).all()
        hyperlinks_ends = hyperlinks[[URL_NORMALIZED, SENTENCE_IDX, CHARS_END]].merge(sentences[SENTENCE].str.len().to_frame("sent-len"), on=[URL_NORMALIZED, SENTENCE_IDX])
        assert (hyperlinks_ends[CHARS_END] <= hyperlinks_ends["sent-len"]).all()
        # assert that all hyperlink spans are at least one character wide
        assert (hyperlinks[CHARS_END] > hyperlinks[CHARS_START]).all()

        return page_infos, hyperlinks, sentences

    def run(self, live_objects: dict):
        page_infos = read_dataframe(self.page_infos_file)  # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)  # type: pd.DataFrame
        sentences = read_dataframe(self.sentences_file)    # type: pd.DataFrame

        if self.sentence_blacklist_regex is not None:
            page_infos, hyperlinks, sentences = self._apply_blacklist(page_infos, hyperlinks, sentences)

        if self.strip_sentences:
            if self.sentence_blacklist_regex is None:
                self.logger.warning("Sentences are to be stripped, but no blacklist filtering was applied previously. It is recommended to apply blacklist filtering to remove sentences which consist entirely of whitespace, otherwise results are undefined for those sentences/documents after stripping.")
            page_infos, hyperlinks, sentences = self._strip_sentences(page_infos, hyperlinks, sentences)

        write_dataframe(page_infos, self.page_infos_postprocessed_file)
        write_dataframe(hyperlinks, self.hyperlinks_postprocessed_file)
        write_dataframe(sentences, self.sentences_postprocessed_file)

component = PostProcessPageContentStage
