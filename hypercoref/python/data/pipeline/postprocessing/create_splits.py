import itertools
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, Tuple

import kahypar
import pandas as pd

from python import *
from python.pipeline.pipeline import PipelineStage
from python.util.util import write_dataframe, read_dataframe, PROJECT_RESOURCES_PATH


class CreateSplitsStage(PipelineStage):
    """
    Creates random splits from a tokenized dataset.
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        # get split definitions, make sure to normalize
        splits_dict = config["splits"]   # type: Dict[str, float]
        splits = pd.DataFrame(splits_dict.items(), columns=["name", "percentage"]).set_index("name")["percentage"]
        splits /= splits.sum()
        self.splits = splits

        self.page_infos_file = None
        self.hyperlinks_file = None
        self.sentences_file = None
        self.tokens_file = None

        if config.get("use_kahypar", True):
            logger.info("Using kahypar hypergraph partitioning for creating splits.")
            self.splitter = self.iter_splits_kahypar
        else:
            logger.info("Using deprecated manual approach for creating splits. Should only be used for recreating the ABC/BBC splits used in the paper.")
            self.splitter = self.iter_splits_deprecated

    def requires_files(self, provided: Dict[str, Path]):
        self.page_infos_file = provided[PAGE_INFOS]
        self.hyperlinks_file = provided[HYPERLINKS]
        self.sentences_file = provided[SENTENCES]
        self.tokens_file = provided[TOKENS]

    def iter_splits_kahypar(self, page_infos, hyperlinks, sentences, tokens, epsilon: float = 0.03) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Uses hypergraph partitioning to create splits while losing as few links as possible. The size of a split is
        measured by its number of documents.

        Partially copied from `cdcr-beyond-corpus-tailored` project, and in there `python.handwritten_baseline.pipeline.
        data.processing.hyperlinks_hack.HyperlinksHackStage._hypergraph_fake_topic_partitioning`.
        """

        # kahypar does not support hyperedges where one vertex appears twice (== documents with two or more hyperlinks
        # to the same other document), see https://github.com/kahypar/kahypar/issues/78#issuecomment-768531716 . We
        # handle such cases by counting the number of edges between a source and target document and using that as edge
        # weights.
        edges_agg_by_doc = hyperlinks.groupby([URL_NORMALIZED, TO_URL_NORMALIZED]).size()
        edges_agg_by_doc.name = "weight"
        edges_agg_by_doc = edges_agg_by_doc.reset_index()

        num_nodes = edges_agg_by_doc[URL_NORMALIZED].nunique()
        num_nets = edges_agg_by_doc[TO_URL_NORMALIZED].nunique()

        # convert documents (==pins) into integers by factorizing
        URL_NORM_FACTOR = "url-normalized-factorized"
        edges_agg_by_doc[URL_NORM_FACTOR] = edges_agg_by_doc[URL_NORMALIZED].factorize()[0]

        # per event, collect integer IDs of documents which contain mentions of that event, i.e. the edge vector of
        # each hypergraph
        hyperedges_by_event = edges_agg_by_doc.groupby(TO_URL_NORMALIZED)[URL_NORM_FACTOR].apply(list)
        hyperedges = list(chain(*hyperedges_by_event.values.flat))
        hyperedge_indices = hyperedges_by_event.map(len).cumsum()
        hyperedge_indices = [0, *hyperedge_indices.values.tolist()]
        edge_weights = edges_agg_by_doc["weight"].values.tolist()
        node_weights = [1] * num_nodes

        # set up the target number of documents per partition: this needs to match the total amount exactly,
        # so we correct the first partition size to account for integer rounding errors from the `astype(int)` line
        k = len(self.splits)
        target_split_sizes = (self.splits * num_nodes).astype(int).tolist()
        target_split_sizes[0] = num_nodes - sum(target_split_sizes[1:])
        hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights,
                                        node_weights)

        context = kahypar.Context()
        context.loadINIconfiguration(str(PROJECT_RESOURCES_PATH / "kahypar" / "cut_kKaHyPar_sea20.ini"))
        context.setK(k)
        context.setEpsilon(epsilon)
        context.setCustomTargetBlockWeights(target_split_sizes)

        kahypar.partition(hypergraph, context)
        # retrieve target partition per document
        doc_ids_to_factorized_doc_ids = edges_agg_by_doc[
            [URL_NORMALIZED, URL_NORM_FACTOR]].drop_duplicates().set_index(URL_NORMALIZED)
        doc_id_to_partition = doc_ids_to_factorized_doc_ids[URL_NORM_FACTOR].map(hypergraph.blockID)

        # determine clusters and documents in each created partition
        clusters_by_partition = {}
        documents_by_partition = {}
        for i, name in enumerate(self.splits.index):
            docs_in_part = doc_id_to_partition.loc[doc_id_to_partition == i].index
            documents_by_partition[name] = docs_in_part
            clusters_in_part = hyperlinks.loc[hyperlinks[URL_NORMALIZED].isin(docs_in_part), TO_URL_NORMALIZED]
            clusters_by_partition[name] = set(clusters_in_part)

        # determine which clusters need to be dropped because they span multiple splits
        clusters_cut = set()
        for a, b in itertools.combinations(clusters_by_partition.values(), 2):
            clusters_cut |= a & b
        num_clusters_lost = len(clusters_cut)
        num_links_lost = hyperlinks[TO_URL_NORMALIZED].isin(clusters_cut).sum()
        if num_links_lost:
            self.logger.info(f"{num_links_lost} hyperlinks (from {num_clusters_lost} clusters) had to be dropped to ensure splits remain independent.")
        clusters_by_partition = {name: clusters - clusters_cut for name, clusters in clusters_by_partition.items()}

        # yield the resulting splits
        for name, documents in documents_by_partition.items():
            clusters = clusters_by_partition[name]

            # documents
            page_infos_in_split = page_infos.loc[documents]
            hyperlinks_in_split = hyperlinks.loc[hyperlinks[TO_URL_NORMALIZED].isin(clusters)]
            sentences_in_split = sentences.loc[documents]
            tokens_in_split = tokens.loc[documents]

            assert tokens_in_split.index.is_monotonic_increasing
            assert sentences_in_split.index.is_monotonic_increasing

            yield name, page_infos_in_split, hyperlinks_in_split, sentences_in_split, tokens_in_split

    def iter_splits_deprecated(self, page_infos, hyperlinks, sentences, tokens) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Naive, deprecated approach for creating splits which loses a lot of links in the process.

        Rough approach:
        The size of a split is measured by its number of hyperlinks. Determine events and their number of hyperlinks.
        Shuffle that. Then, determine at which indices one has to split this shuffled list of events to produce splits whose
        number of hyperlinks corresponds to each desired percentage.

        Splits are created from smallest to largest. When enforcing independence to previous splits, some links have to
        be dropped. Recommendation: Make test the smallest (created first - will be the cleanest without any removed
        links), dev a little larger to still have enough links, and train the rest.

        Note that the data splits created by this stage only consist of documents which contain at least one outgoing
        hyperlink. Documents which are only referenced by ingoing links but have no outgoing links will appear in neither
        of the splits created.
        """
        events = hyperlinks[TO_URL].value_counts()
        num_hyperlinks_total = events.sum()
        events_shuffled = events.sample(frac=1.0, random_state=0)
        # the cumulative number of hyperlinks when moving through our Series of shuffled events
        events_shuffled_num_hyperlinks = pd.Index(events_shuffled.cumsum().values)
        # determine start/end indices in `events_shuffled` for each split
        cutoff_points = self.splits.sort_values().cumsum().map(lambda pct: events_shuffled_num_hyperlinks.get_loc(pct * num_hyperlinks_total, method="pad"))
        splits_from_to = pd.DataFrame({"from": cutoff_points.shift(1, fill_value=0), "to": cutoff_points, })
        # ensure no hyperlink is wasted by extending the biggest split all the way to the end of the data (in case some rounding errors crept in)
        splits_from_to.iloc[-1]["to"] = len(events_shuffled)

        # We need to ensure each page is part only of one split. Track this with a set of all page identifiers.
        pages_available_for_splits = set(page_infos.index.unique())
        for name, span in splits_from_to.iterrows():
            # determine candidate events, hyperlinks and pages in split
            events_in_split = events_shuffled.iloc[span["from"]:span["to"]]
            hyperlinks_in_split = hyperlinks.loc[hyperlinks[TO_URL].isin(events_in_split.index)]
            page_infos_in_split = page_infos.loc[hyperlinks_in_split[URL_NORMALIZED].unique()]

            # now determine which pages (and hyperlinks) need to be dropped in accordance to other splits
            page_infos_in_split_actual = page_infos_in_split.loc[
                page_infos_in_split.index.isin(pages_available_for_splits)]
            hyperlinks_in_split_actual = hyperlinks_in_split.loc[
                hyperlinks_in_split[URL_NORMALIZED].isin(page_infos_in_split_actual.index)]
            sentences_in_split_actual = sentences.loc[page_infos_in_split_actual.index]
            tokens_in_split_actual = tokens.loc[page_infos_in_split_actual.index]

            assert tokens_in_split_actual.index.is_monotonic_increasing
            assert sentences_in_split_actual.index.is_monotonic_increasing

            # count how many hyperlinks were lost due to splits having to be entirely independent of each other
            num_links_lost = len(hyperlinks_in_split) - len(hyperlinks_in_split_actual)
            num_pages_lost = len(page_infos_in_split) - len(page_infos_in_split_actual)
            if num_links_lost:
                self.logger.info(
                    f"{num_links_lost} hyperlinks (from {num_pages_lost} pages) had to be dropped to ensure split '{name}' remains independent of other splits.")
            pages_available_for_splits -= set(page_infos_in_split_actual.index.unique())

            yield name, page_infos_in_split_actual, hyperlinks_in_split_actual, sentences_in_split_actual, tokens_in_split_actual

    def run(self, live_objects: dict):
        page_infos = read_dataframe(self.page_infos_file)  # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)  # type: pd.DataFrame
        sentences = read_dataframe(self.sentences_file)    # type: pd.DataFrame
        tokens = read_dataframe(self.tokens_file)[TOKEN]  # type: pd.Series

        for split_data in self.splitter(page_infos, hyperlinks, sentences, tokens):
            name, page_infos_split, hyperlinks_split, sentences_split, tokens_split = split_data

            # write to file
            path_of_split = self.stage_disk_location / name
            path_of_split.mkdir(parents=True, exist_ok=True)

            write_dataframe(page_infos_split, path_of_split / "page_infos")
            write_dataframe(sentences_split, path_of_split / "sentences")
            write_dataframe(tokens_split.to_frame(TOKEN), path_of_split / "tokens")
            write_dataframe(hyperlinks_split, path_of_split / "hyperlinks")

component = CreateSplitsStage
