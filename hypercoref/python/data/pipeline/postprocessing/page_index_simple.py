from pathlib import Path
from typing import Dict, Set

import pandas as pd

from python import *
from python.pipeline.pipeline import PipelineStage


class SimplePageIndexPostProcessingStage(PipelineStage):
    """
    This stage filters the page index based on URL blacklist/whitelist substrings.
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
        to_keep = ~blacklisted | whitelisted
        page_index_filtered = page_index.loc[to_keep]

        self.logger.info(f"Of {len(page_index)} pages, {len(page_index_filtered)} pages remain after filtering ({(~to_keep).sum()} were blacklisted).")

        page_index_filtered.to_csv(self.page_index_filtered_simple_file)


component = SimplePageIndexPostProcessingStage
