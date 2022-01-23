from pathlib import Path
from typing import Dict

import pandas as pd

from python import *
from python.data.pipeline.postprocessing.page_deduplication.lsh import LshIdentifyPageDuplicates
from python.data.pipeline.postprocessing.page_deduplication.tfidf import TfidfIdentifyPageDuplicates
from python.data.pipeline.postprocessing.page_deduplication.two_stage import TwoStageIdentifyPageDuplicates
from python.pipeline import MAX_CORES
from python.pipeline.pipeline import PipelineStage
from python.util.util import read_dataframe

IN_DEGREE = "in-degree"
OUT_DEGREE = "out-degree"


class PostProcessPageIndex(PipelineStage):
    """
    This stage:
    - Deduplicates the set of pages based on their textual content, using either/or/and locality sensitive hashing
      (LSH) on character shingles and TF-IDF+cosine similarity
    - Removes all pages with hyperlink in-degree == 0 and out-degree == 0.
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        # set up deduplifier
        deduplication_kwargs = config.get("deduplication_kwargs", {})
        deduplication_type = config["deduplication_type"]
        if deduplication_type == "tfidf":
            clazz = TfidfIdentifyPageDuplicates
            deduplication_kwargs[MAX_CORES] = config_global[MAX_CORES]
        elif deduplication_type == "lsh":
            clazz = LshIdentifyPageDuplicates
        elif deduplication_type == "two_stage":
            clazz = TwoStageIdentifyPageDuplicates
        else:
            raise ValueError(f"Unknown deduplication technique {deduplication_type}.")
        self.deduplifier = clazz(**deduplication_kwargs)
        self.page_index_file = None
        self.page_infos_file = None
        self.hyperlinks_file = None

        self.page_index_filtered_file = Path(self.stage_disk_location) / "page_index_filtered.csv"

    def requires_files(self, provided: Dict[str, Path]):
        self.page_index_file = provided[PAGE_INDEX]
        self.page_infos_file = provided[PAGE_INFOS]
        self.hyperlinks_file = provided[HYPERLINKS]

    def files_produced(self) -> Dict[str, Path]:
        # replace old index
        return {PAGE_INDEX: self.page_index_filtered_file}

    def run(self, live_objects: dict):
        page_index = pd.read_csv(self.page_index_file, index_col=0)
        page_infos = read_dataframe(self.page_infos_file)  # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)  # type: pd.DataFrame

        # Drop pages which have NaN text
        page_infos.dropna(subset=[TEXT], inplace=True)

        # Deduplicate pages based on short URLs: two pages may have been scraped which are identical, but may differ
        # because one was scraped via http and the other via https. Of all duplicates of a page, keep the longest one.
        page_infos.sort_values([URL_NORMALIZED, CONTENT_CHARS], ascending=False, inplace=True)
        page_infos.drop_duplicates(subset=URL_NORMALIZED, inplace=True)
        # at this point, the normalized URLs are unique, so they can be used as the index
        page_infos = page_infos.reset_index().set_index(URL_NORMALIZED)
        assert page_infos.index.is_unique
        # replicate dropping in hyperlinks dataframe
        hyperlinks = hyperlinks.loc[hyperlinks[URL].isin(page_infos[URL])]

        # remove those pages which have an in-degree and out-degree of 0
        out_degree = hyperlinks[URL_NORMALIZED].value_counts()
        in_degree = hyperlinks[TO_URL_NORMALIZED].value_counts()
        degree = pd.concat([out_degree, in_degree], axis=1).fillna(0).astype(int).rename(columns={URL_NORMALIZED: OUT_DEGREE, TO_URL_NORMALIZED: IN_DEGREE})
        page_infos_d = page_infos.loc[page_infos.index.intersection(degree.index)]
        page_infos_d.index.name = URL_NORMALIZED
        self.logger.info(f"Of {len(page_infos)} pages, {len(page_infos_d)} had an in- our out-degree greater zero ({len(page_infos) - len(page_infos_d)} pages were dropped).")

        self.logger.info("Deduplicating pages based on text...")
        self.deduplifier.fit(page_infos_d[TEXT])
        clustering = self.deduplifier.find_duplicates(page_infos_d[TEXT])

        # inside each cluster of duplicates, keep the page with the highest in-degree and out-degree
        self.logger.info("Deduplication complete, selecting the best page per cluster...")
        degree.sort_values(by=[IN_DEGREE, OUT_DEGREE], inplace=True)
        cluster_sizes = clustering.value_counts()
        non_singleton_clusters = cluster_sizes.loc[cluster_sizes > 1]
        pages_to_keep = clustering.loc[clustering.isin(cluster_sizes.loc[cluster_sizes == 1].index)].index

        pages_by_degree_and_length = degree.merge(page_infos_d[CONTENT_CHARS], how="left", left_index=True, right_index=True).sort_values(by=[IN_DEGREE, OUT_DEGREE, CONTENT_CHARS], ascending=False)
        for cluster_id in non_singleton_clusters.index.values:
            normalized_urls_in_cluster = clustering.loc[clustering == cluster_id]
            pages_in_cluster_by_degree_and_length = pages_by_degree_and_length.loc[normalized_urls_in_cluster.index]
            best_page = pages_in_cluster_by_degree_and_length.iloc[0:1]
            pages_to_keep = pages_to_keep.append(best_page.index)

        # page infos filtered by degree and deduplication
        page_infos_dd = page_infos_d.loc[pages_to_keep]

        # log and write debug information to disk
        self.logger.info(f"Of {len(page_infos_d)} pages, {len(page_infos_dd)} unique ones were found ({len(page_infos_d) - len(page_infos_dd)} duplicates over {len(non_singleton_clusters)} non-singleton clusters).")
        removed_pages = pd.merge(clustering.loc[clustering.isin(non_singleton_clusters)].to_frame("cluster-id"), page_infos_d, how="left", left_index=True, right_index=True)
        removed_pages.to_csv(self.stage_disk_location / "pages_removed_in_deduplication.csv")

        # swap index around to non-normalized URL again, then apply filtering to page index
        page_infos_dd = page_infos_dd.reset_index().set_index(URL)
        page_index_dd = page_index.loc[page_infos_dd.index]
        page_index_dd.to_csv(self.page_index_filtered_file)


component = PostProcessPageIndex
