from pathlib import Path
from typing import Dict

import pandas as pd
from tabulate import tabulate

from python import *
from python.pipeline.pipeline import PipelineStage
from python.util.util import read_dataframe


class AnchorsIntoOneFileStage(PipelineStage):
    """
    Writes all hyperlink anchor texts into a single plaintext file (great for manual inspection).
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.page_infos_file = self.sentences_file = self.hyperlinks_file = None
        self.minimum_content_chars_for_article_status = config.get("minimum_content_chars_for_article_status", None)

    def requires_files(self, provided: Dict[str, Path]):
        self.page_infos_file = provided[PAGE_INFOS]
        self.sentences_file = provided[SENTENCES]
        self.hyperlinks_file = provided[HYPERLINKS]

    def run(self, live_objects: dict):
        # load all the files
        page_infos = read_dataframe(self.page_infos_file)      # type: pd.DataFrame
        sentences = read_dataframe(self.sentences_file)        # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)      # type: pd.DataFrame

        if self.minimum_content_chars_for_article_status is None:
            # if no threshold is given: plot distribution of text length (unit: number of characters) for all extracted pages
            ax = page_infos[CONTENT_CHARS].plot(kind="hist", bins=30, loglog=True)
            fig = ax.get_figure()
            fig.savefig(self.stage_disk_location / "characters_per_article.png")
            fig.clf()
        else:
            # identify non-article webpages (those with insufficient amount of text)
            # based on the above plot, we define that a webpage needs to have at least 1000 chars of content to count as an article
            too_short_articles = page_infos[CONTENT_CHARS] < self.minimum_content_chars_for_article_status
            non_article_webpages = page_infos.loc[~too_short_articles]   # TODO consider using this
            self.logger.info(f"{too_short_articles.value_counts().get(True, 0)} of {len(too_short_articles)} pages had less than {self.minimum_content_chars_for_article_status} characters of textual content and are therefore not considered as news articles.")

        # obtain link anchor text
        df = pd.merge(sentences, hyperlinks, on=[URL_NORMALIZED, SENTENCE_IDX])
        df[ANCHOR_TEXT] = df.apply(lambda v: v[SENTENCE][v[CHARS_START]:v[CHARS_END]], axis=1)

        # just extract the spans and sort them and print them
        self.logger.info("Writing hyperlink anchor texts to a file (for reference)...")
        with open(Path(self.stage_disk_location) / "link_anchors.txt", "w") as f:
            pretty_df = tabulate(df[[ANCHOR_TEXT, TO_URL_NORMALIZED]].sort_values(ANCHOR_TEXT), headers="keys", showindex=False)
            f.write(pretty_df)


component = AnchorsIntoOneFileStage
