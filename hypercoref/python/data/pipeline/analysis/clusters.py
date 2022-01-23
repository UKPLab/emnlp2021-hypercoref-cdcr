from pathlib import Path
from typing import Dict

import pandas as pd

from python import *
from python.pipeline.pipeline import PipelineStage
from python.util.util import read_dataframe


class ClusterAnalysisStage(PipelineStage):
    """
    Samples several clusters: pages with in-degree between 3 and 5 alongside the sentences and hyperlinks referencing
    them, and writes them into CSV files.
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.hyperlinks_file = self.sentences_file = self.page_infos_file = None

    def requires_files(self, provided: Dict[str, Path]):
        self.hyperlinks_file = provided[HYPERLINKS]
        self.sentences_file = provided[SENTENCES]
        self.page_infos_file = provided[PAGE_INFOS]

    def run(self, live_objects: dict):
        # load all the files
        page_infos = read_dataframe(self.page_infos_file)      # type: pd.DataFrame
        sentences = read_dataframe(self.sentences_file)        # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)      # type: pd.DataFrame

        # sampling settings
        num_subgraphs_to_sample = 25
        pages_indegree = hyperlinks[TO_URL].value_counts()
        pages_with_interesting_indegree = pages_indegree.loc[(pages_indegree > 2) & (pages_indegree < 6) & pages_indegree.index.isin(page_infos.index)]

        if pages_with_interesting_indegree.empty:
            self.logger.info("No interesting pages to export.")
            return

        subgraph_pages = pages_with_interesting_indegree.sample(num_subgraphs_to_sample, random_state=0)
        subgraphs = []
        for url in subgraph_pages.index.values:
            # look up text of that page
            target_page_pubdate = page_infos.at[url, PUBLISH_DATE]
            target_page_title = page_infos.at[url, TITLE]
            target_page_text = f"{target_page_pubdate}: {target_page_title} ------- " + " ".join(sentences.loc[url, SENTENCE].values.tolist())
            subgraphs.append((url, "target", target_page_text))

            hyperlinks_in_subgraph = hyperlinks.loc[hyperlinks[TO_URL] == url, [URL, SENTENCE_IDX, CHARS_START, CHARS_END]]

            # wrap link anchors in >>> <<<
            for _, hyperlink in hyperlinks_in_subgraph.iterrows():
                sent_idx = hyperlink[SENTENCE_IDX]
                context_sentences = sentences.loc[hyperlink[URL]].loc[(sent_idx-1):(sent_idx+1), SENTENCE]

                char_idx_from = hyperlink[CHARS_START]
                char_idx_to = hyperlink[CHARS_END]
                sentence_with_hyperlink = context_sentences.at[sent_idx]
                sentence_with_highlight = sentence_with_hyperlink[:char_idx_from] + ">>>" + sentence_with_hyperlink[char_idx_from:char_idx_to] + "<<<" + sentence_with_hyperlink[char_idx_to:]
                context_sentences.at[sent_idx] = sentence_with_highlight

                context_sentences = " ".join(context_sentences.values.tolist())
                subgraphs.append((url, "anchor-context", context_sentences))

        subgraphs = pd.DataFrame(subgraphs, columns=[TO_URL, "type", SENTENCE])
        subgraphs.to_csv(self.stage_disk_location / "example_subgraphs.csv", index=False)


component = ClusterAnalysisStage
