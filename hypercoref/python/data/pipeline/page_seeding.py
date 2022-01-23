from typing import Optional

import pandas as pd
from url_normalize import url_normalize

from python import *
from python.common_components import COMMON_CRAWL
from python.common_components.commoncrawl import CommonCrawl
from python.pipeline.pipeline import PipelineStage
from python.util.util import url_normalize_series


class PageSeedingStage(PipelineStage):
    """
    Queries the CommonCrawl index for all pages they have indexed for a given list of URL prefixes, deduplicates that
    list and stores result in a CSV file.
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

    def run(self, live_objects: dict):
        common_crawl = live_objects[COMMON_CRAWL]  # type: CommonCrawl

        news_outlets = self.config["news_outlets"]
        from_ = self.config["from"]
        to = self.config["to"]

        for name, params in news_outlets.items():
            self.logger.info(f"Applying for news outlet {name}...")

            news_outlet_location = self.stage_disk_location / name
            news_outlet_location.mkdir(exist_ok=True, parents=True)

            # query common crawl
            df_of_prefixes = []
            for prefix in params["promising_prefixes"]:
                df = common_crawl.cdx_query(url=prefix, wildcard_query=True, from_=from_, to=to)
                if df is not None:
                    df_of_prefixes.append(df)
            df = pd.concat(df_of_prefixes)
            if df is None or df.empty:
                self.logger.info("Skipping further steps...\n")
                continue
            else:
                df.to_csv(news_outlet_location / "commoncrawl_raw_cdx.csv")

            # post-process commoncrawl results: sort by url and timestamp (ascending), then drop all but the first entry
            # for each URL, then use URLs as index
            self.logger.info("Removing duplicates...")

            # TODO support HTTP 301 redirects
            # TODO support query redirection (cases like '?redirect_uri=https://...') if that's even possible

            # Normalize URLs. Long hostnames make python trip (see https://bugs.python.org/issue32958), this has
            # happened to us before here. We wrap normalization in try/except and drop any NaNs afterwards as a
            # workaround.
            def norm(u: str) -> Optional[str]:
                try:
                    return url_normalize(u)
                except:
                    return None
            df[URL] = df[URL].apply(norm)
            df.dropna(subset=[URL], inplace=True)

            # identify duplicate URLs (without taking the scheme into account) and keep oldest capture
            df["url-without-scheme"] = url_normalize_series(df[URL])[0]
            df.sort_values(by=["url-without-scheme", TIMESTAMP], inplace=True)
            df.drop_duplicates(subset="url-without-scheme", inplace=True)
            df.drop(columns="url-without-scheme", inplace=True)
            
            # apply URL blacklist, if present
            blacklist = set(params.get("blacklist", []))
            if blacklist:
                blacklisted = df[URL].map(lambda url: any(url.startswith(b) for b in blacklist))
                df = df.loc[~blacklisted]

            df.set_index(URL, inplace=True)

            df.to_csv(news_outlet_location / "commoncrawl_cdx_deduplicated.csv")
            self.logger.info(f"Done. {len(df)} documents found for {name}.\n")


component = PageSeedingStage
