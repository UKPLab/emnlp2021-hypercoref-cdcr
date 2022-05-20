from pathlib import Path
from typing import Dict, Set

import pandas as pd

from python import *
from python.pipeline.pipeline import PipelineStage


class SimplePageIndexPostProcessingStage(PipelineStage):
    """
    Makes several smaller modifications to a page index file:
    - Filters index based on URL blacklist/whitelist substrings.
    - Fixes the CommonCrawl URL prefix for old indexer API results, see [1].

    [1] https://commoncrawl.org/2022/03/introducing-cloudfront-access-to-common-crawl-data/
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.page_index_file = None
        self.page_index_filtered_simple_file = Path(self.stage_disk_location) / "page_index_filtered_simple.csv"

        self.url_substring_whitelist = set(config.get("url_substring_whitelist", []))
        self.url_substring_blacklist = set(config.get("url_substring_blacklist", []))

    def requires_files(self, provided: Dict[str, Path]):
        self.page_index_file = provided[PAGE_INDEX]

    def files_produced(self) -> Dict[str, Path]:
        # replace old index
        return {PAGE_INDEX: self.page_index_filtered_simple_file}

    def run(self, live_objects: dict):
        page_index = pd.read_csv(self.page_index_file, index_col=0)

        def contains(s: Set):
            return page_index.index.to_series().map(lambda url: any(term in url for term in s))
        blacklisted = pd.Series(True, index=page_index.index) if not self.url_substring_blacklist else contains(self.url_substring_blacklist)
        whitelisted = pd.Series(False, index=page_index.index) if not self.url_substring_whitelist else contains(self.url_substring_whitelist)
        to_drop = blacklisted & (~whitelisted)
        self.logger.info(f"Of {len(page_index)} pages, {(~to_drop).sum()} pages remain after filtering.")
        page_index.drop(to_drop.index[to_drop], inplace=True)

        # #1 fix the CommonCrawl URL prefix for old indexer API results
        old_prefix = "https://commoncrawl.s3.amazonaws.com/"
        current_prefix = "https://data.commoncrawl.org/"
        has_old_prefix = page_index[FILENAME].str.startswith(old_prefix)
        page_index.loc[has_old_prefix, FILENAME] = page_index.loc[has_old_prefix, FILENAME].map(
            lambda url: current_prefix + url[len(old_prefix):])
        self.logger.info(f"Fixed CommonCrawl URL prefix for {sum(has_old_prefix)} pages.")

        page_index.to_csv(self.page_index_filtered_simple_file)


component = SimplePageIndexPostProcessingStage
