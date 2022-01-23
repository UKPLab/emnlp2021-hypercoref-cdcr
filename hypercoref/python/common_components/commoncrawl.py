import pprint
from datetime import datetime

import cdx_toolkit
import pandas as pd
import tqdm
from typing import Optional

from python import *
from python.pipeline import GLOBAL, ComponentBase, DEVELOPMENT_MODE


class CommonCrawl(ComponentBase):
    TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"   # applies to wayback machine and common crawl
    DEFAULT_SIZE_ESTIMATE = 1500
    DEBUG_MAX_COMMONCRAWL_HITS = 200

    def __init__(self, config, config_global, logger):
        super(CommonCrawl, self).__init__(config, config_global, logger)

        self.cache = self._provide_cache("commoncrawl_cdx", scope=GLOBAL, size_limit=100*1024*1024*1024)
        self.cdx = cdx_toolkit.CDXFetcher(source="cc")

        self.debug = config_global.get(DEVELOPMENT_MODE, False)

    def cdx_query(self,
                  url: str,
                  wildcard_query: bool = False,
                  from_: datetime = None,
                  to: datetime = None,
                  limit: Optional[int] = None):
        """
        Query the Common Crawl CDX API for which pages were captured at which time.
        :param url: URL to query for
        :param wildcard_query: if True, this method will query for all pages which have the given url as their prefix
        :param from_: If set, only retrieves pages which were captured at least once after this datetime. If None,
                      retrieves only the past 12 months by default!
        :param to: if set, only retrieves pages which were captured at least once until this datetime
        :param limit: if set, return only n results
        :return: pandas DataFrame with columns url, timestamp, digest
        """
        if self.debug:
            limit = CommonCrawl.DEBUG_MAX_COMMONCRAWL_HITS
            self.logger.info(f"Running in debug mode, number of results is limited to {limit}")

        query = {"url": url, "wildcard_query": wildcard_query, "from": from_, "to": to, "limit": limit}
        query_serialized = pprint.pformat(query)

        # obtain CDX result
        if query_serialized in self.cache:
            df = self.cache[query_serialized]
        else:
            try:
                df = self._do_cdx_query(url, wildcard_query, from_, to, limit)
                self.cache[query_serialized] = df
            except ValueError as e:
                # do not cache in case of errors
                self.logger.error(e)
                df = None

        # post-process
        if df is not None and not df.empty:
            # convert all timestamps to datetimes
            df.loc[:, TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], format=self.TIMESTAMP_FORMAT, errors="coerce")
            # append warc prefix to obtain full URLs for WARC files
            df[FILENAME] = self.cdx.warc_url_prefix + "/" + df[FILENAME]
        return df

    def _do_cdx_query(self, url: str, wildcard_query: bool = False, from_: datetime = None, to: datetime = None, limit=None):
        self.logger.info(f"Querying Common Crawl CDX for {url}...")

        # set up the query URL
        query_parts = [url]
        if wildcard_query:
            if not url.endswith("/"):
                query_parts.append("/")
            query_parts.append("*")
        query = "".join(query_parts)

        kwargs = {
            "url": query,
            "filter": "status:200",
            "mime-detected": "text/html",
            "languages": "eng",
        }
        if from_:
            kwargs["from_ts"] = from_.strftime(self.TIMESTAMP_FORMAT)
        if to:
            kwargs["to"] = to.strftime(self.TIMESTAMP_FORMAT)

        if wildcard_query:
            size_estimate = self.cdx.get_size_estimate(**kwargs)
            self.logger.info(f"{size_estimate} estimated hits.")
        else:
            size_estimate = self.DEFAULT_SIZE_ESTIMATE
        if limit:
            kwargs["limit"] = size_estimate = limit

        captures = []
        with tqdm.tqdm(total=size_estimate, desc="CDX hits", mininterval=10) as p:
            it = self.cdx.iter(**kwargs)
            while True:
                try:
                    obj = next(it)
                except StopIteration:
                    break
                except Exception as e:
                    self.logger.warning(f"CDX iteration failed with '{str(e)}', continuing...")

                # sometimes there are crawled robots.txt files which we want to ignore
                if "robotstxt" in obj["filename"]:
                    continue

                # in theory, there is a parameter fl= with one can select the fields returned, but fl=url returns
                # nothing for an unknown reason, so we have to kludge it
                captures.append({URL: obj["url"],
                                 TIMESTAMP: obj["timestamp"],
                                 FILENAME: obj["filename"],
                                 OFFSET: obj["offset"],
                                 LENGTH: obj["length"]})
                if len(captures) % 1000 == 0:
                    p.update(len(captures))

        df = pd.DataFrame(captures)
        return df


component = CommonCrawl
