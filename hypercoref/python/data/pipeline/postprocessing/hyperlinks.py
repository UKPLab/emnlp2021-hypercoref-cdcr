import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import pandas as pd
import pygtrie
import tldextract
from tabulate import tabulate

from python import *
from python.pipeline.pipeline import PipelineStage
from python.util.spans import harmonize_span_boundary
from python.util.util import url_normalize_series, write_dataframe, read_dataframe


class HyperlinksPostProcessingStage(PipelineStage):
    """
    This stage performs post-processing on the list of hyperlinks, in particular:
    - filtering hyperlinks by target URL or anchor text
    - removing hyperlinks to highly linked pages
    - fixing hyperlink span boundaries to match word boundaries
    - deduplicating hyperlinks by target URL and anchor text
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.page_infos_file = None
        self.sentences_file = None
        self.hyperlinks_file = None

        self.page_infos_postprocessed_file = self.stage_disk_location / "page_infos_postprocessed"
        self.sentences_postprocessed_file = self.stage_disk_location / "sentences_postprocessed"
        self.hyperlinks_postprocessed_file = self.stage_disk_location / "hyperlinks_postprocessed"

        self.analysis_output_location = self.stage_disk_location / "analysis"
        self.analysis_output_location.mkdir(parents=True, exist_ok=True)

        self.netloc_blacklist = set(config.get("netloc_blacklist", []))
        self.netloc_whitelist = set(config.get("netloc_whitelist", []))
        self.normalized_url_prefix_blacklist = set(config.get("normalized_url_prefix_blacklist", []))
        self.normalized_url_suffix_blacklist = set(config.get("normalized_url_suffix_blacklist", []))
        self.normalized_url_substring_blacklist = set(config.get("normalized_url_substring_blacklist", []))

        self.in_degree_long_tail_cutoff = config.get("in_degree_long_tail_cutoff", 1.0)
        self.out_degree_long_tail_cutoff = config.get("out_degree_long_tail_cutoff", 1.0)

        # with to_url_pattern_target_mass < 0.5, there could be multiple valid solutions
        self.to_url_pattern_target_mass = config.get("to_url_pattern_target_mass", None)
        if self.to_url_pattern_target_mass < 0.5 or self.to_url_pattern_target_mass > 1.0:
            raise ValueError("`to_url_pattern_target_mass` needs to be from the interval [0.5, 1.0], otherwise results are undefined!")

        self.harmonize_span_boundaries = config.get("harmonize_span_boundaries", True)

        self.max_anchor_target_duplicate_occurrences = config.get("max_anchor_target_duplicate_occurrences", 7)

        if "anchor_blacklist_regexes" in config:
            all_in_one = "|".join(f"({regex})" for regex in config["anchor_blacklist_regexes"])
            self.anchor_blacklist_regex = re.compile(all_in_one, flags=re.IGNORECASE)
        else:
            self.anchor_blacklist_regex = None

        self.reduce_spans_to_syntactic_head = config.get("reduce_spans_to_syntactic_head", False)
        self.extend_spans_to_full_sentence = config.get("extend_spans_to_full_sentence", False)
        if self.reduce_spans_to_syntactic_head and self.extend_spans_to_full_sentence:
            raise ValueError("Cannot reduce spans to syntactic head and extend to full sentence at the same time!")

    def requires_files(self, provided: Dict[str, Path]):
        self.page_infos_file = provided[PAGE_INFOS]
        self.sentences_file = provided[SENTENCES]
        self.hyperlinks_file = provided[HYPERLINKS]

    def files_produced(self) -> Dict[str, Path]:
        # when dropping hyperlinks, we drop affected pages / sentences too
        return {PAGE_INFOS: self.page_infos_postprocessed_file,
                SENTENCES: self.sentences_postprocessed_file,
                HYPERLINKS: self.hyperlinks_postprocessed_file}

    def _heuristic_headline_detection(self,
                                      hyperlinks: pd.DataFrame,
                                      max_title_case_words: int = 3,
                                      max_title_case_words_ratio: float = 0.6,
                                      min_words: int = 7):
        """
        Heuristic detection for headlines based on the ratio of words starting with a capital letter. This catches
        Title Case Headlines and also ALL CAPS HEADLINES.
        """
        # Note: the ratio isn't perfect, words like "N'Gambe" count as two title case words but only one word,
        # resulting in a ratio of 2. These cases don't matter so much in practice though.
        title_case_pattern = re.compile(r"[A-Z]\w+")
        num_title_case_words = hyperlinks[ANCHOR_TEXT].map(lambda s: len(title_case_pattern.findall(s)))
        num_words = hyperlinks[ANCHOR_TEXT].map(lambda s: len(s.split()))
        title_by_total = num_title_case_words / num_words
        is_first_char_upper = hyperlinks[ANCHOR_TEXT].str.slice(0,1).str.isupper()
        is_headline = is_first_char_upper & \
                      (num_words >= min_words) & \
                      (num_title_case_words > max_title_case_words) & \
                      (title_by_total > max_title_case_words_ratio)
        return is_headline

    def _page_degree_analysis(self,
                              hyperlinks: pd.DataFrame,
                              in_degree_long_tail_cutoff: float,
                              out_degree_long_tail_cutoff: float) -> Tuple[pd.Series, pd.Series]:
        # create graph from hyperlink information
        g = nx.DiGraph()
        for tup in hyperlinks[[URL_NORMALIZED, TO_URL_NORMALIZED]].itertuples():
            _from, _to = tup[1:3]
            g.add_edge(_from, _to)

        self.logger.info("Writing hyperlink graph to disk (for reference)...")
        nx.write_gpickle(g, os.path.join(self.analysis_output_location, "hyperlink_graph.pkl"))

        # The number of in- and out-degrees per page seems to follow a Zipfian distribution. We want to cut off the
        # long tail at a certain percentage of mass (say 95%).
        in_out_degree_by_page = []
        in_out_degree_frequency = []
        in_out_degree_filter = []
        for obj, cutoff, what_degree in zip([g.in_degree, g.out_degree], [in_degree_long_tail_cutoff, out_degree_long_tail_cutoff], ["In", "Out"]):
            # page -> (in/out)-degree
            degree_by_page = pd.Series(dict(obj))
            # (in/out)-degree -> number of pages with this (in/out)-degree
            degree_frequency = degree_by_page.value_counts().sort_index(ascending=True)
            # the above, but relative and cumulative
            degree_frequency_cumu = degree_frequency.cumsum() / degree_frequency.sum()
            # find infimum to long_tail_cutoff and obtain it's index -> this is the maximum (in/out)-degree we still permit
            max_permitted_degree = degree_frequency_cumu.loc[degree_frequency_cumu < cutoff].last_valid_index()

            page_has_high_degree = degree_by_page > max_permitted_degree # type: pd.Series

            self.logger.info(f"{what_degree}-degree cutoff is {cutoff} ({max_permitted_degree} absolute), therefore {page_has_high_degree.value_counts().get(True, 0)} pages with degree > {max_permitted_degree} will be removed.")

            in_out_degree_by_page.append(degree_by_page)
            in_out_degree_filter.append(page_has_high_degree)
            in_out_degree_frequency.append(degree_frequency)

        in_degree_by_page, out_degree_by_page = in_out_degree_by_page
        in_degree_frequency, out_degree_frequency = in_out_degree_frequency
        page_has_high_in_degree, page_has_high_out_degree = in_out_degree_filter

        # print/plot indegree and outdegree stats
        IN_DEGREE = "in_degree"
        OUT_DEGREE = "out_degree"
        for name, ser in [(IN_DEGREE, in_degree_frequency), (OUT_DEGREE, out_degree_frequency)]:
            path = os.path.join(self.analysis_output_location, f"{name}.txt")
            with open(path, "w") as f:
                table = tabulate(ser.to_frame("nb. occurrences"), headers="keys")
                f.write(table)

            # # TODO plotting didn't look good
            # ax = ser.plot(kind="hist", bins=30, loglog=True)
            # fig = ax.get_figure()
            # fig.savefig(os.path.join(self.analysis_output_location, f"{col}.png"))
            # fig.clf()

        # log some more statistics:
        # nodes with out degree 0 and in-degree 1 are on the fringe: keep them for now, one might want to crawl those
        # at a later point in time
        degrees = pd.DataFrame({IN_DEGREE: in_degree_by_page, OUT_DEGREE: out_degree_by_page})
        num_tail_pages = len(degrees.loc[(degrees[IN_DEGREE] > 0) & (degrees[OUT_DEGREE] == 0)])
        self.logger.info(f"{num_tail_pages} pages are tails (in-degree > 0, out-degree = 0).")
        num_head_pages = len(degrees.loc[(degrees[IN_DEGREE] == 0) & (degrees[OUT_DEGREE] > 0)])
        self.logger.info(f"{num_head_pages} pages are heads (in-degree = 0, out-degree > 0).")
        self.logger.info(f"{len(degrees) - num_head_pages - num_tail_pages} pages are in between (in-degree > 0, out-degree > 0).")

        return page_has_high_in_degree, page_has_high_out_degree

    def _url_prefix_analysis(self,
                             hyperlinks_index: pd.Index,
                             to_url_split: pd.DataFrame,
                             target_url_mass: float) -> pd.Series:
        """
        We want to find the pattern which "good", article-y URLs have to weed out any other links to hub pages, one-off promotion webpages, etc.
        The rough approach is:
          - to generalize URLs, split them at "/", then replace digit sequences and long alphanumeric sequences with constant literals
          - put these generalized URLs in a trie data structure
          - traverse the trie to find the generalized URL prefixes which together contain just over X% of all links
          - drop any hyperlinks which do not match those prefixes
        """
        def generalize_url(_netloc: str, _path: str) -> str:
            # subdomains pose a problem: they specialize URLs to the left of the domain (`very-specific-subdomain.specific-subdomain.domain.tld/specific/very-specific`)
            # which conflicts with our trie idea, so we fit them in after the domain (`domain.tld/specific-subdomain/very-specific-subdomain/specific/very-specific`),
            # similar to the Java package naming scheme
            domain_parts = tldextract.extract(_netloc)

            generalized = [domain_parts.suffix, domain_parts.domain]
            if domain_parts.subdomain:
                generalized += domain_parts.subdomain.split(".")[::-1]      # subdomain in reverse order
            else:
                generalized.append("S")     # no subdomain? create a dummy one called "S"

            # generalize parts of the path: decimal sequences become "#", short alphanumeric sequences ("news", "en",
            # "video", ...) are kept as literals, longer sequences become "A"
            for part in _path.split("/"):
                if not part:
                    continue
                elif part.isdecimal():
                    generalized.append("#")
                elif len(part) <= 10:
                    generalized.append(part.lower().strip())
                else:
                    generalized.append("A")

            return "/".join(generalized)

        generalized_urls = to_url_split[["netloc", "path"]].apply(lambda ser: generalize_url(*ser.values), axis=1)

        # make a trie which counts how often each generalized URL occurs
        trie = pygtrie.StringTrie(separator="/")
        for url in generalized_urls:
            if url in trie:
                trie[url] += 1
            else:
                trie[url] = 1
        # normalize counts
        for key in trie.keys():
            trie[key] /= len(generalized_urls)

        def traverse_callback(_, path, children, share=None) -> Tuple[pd.Series, bool]:
            """
            Traverses the trie of URLs. At each node in the trie, determines the percentage of all URLs for which the current
            node is a prefix for. Knowing the percentage of each child, it tries to find the set of prefixes (from large to
            small) which, when combined, come closest to the given `target_url_mass`.
            :returns: tuple of prefixes and their share, and a boolean denoting whether a subset of prefixes was chosen (this
                      is important for internal use for recursion)
            """
            p = path if path else ""
            self_share = pd.Series({p: share}) if share is not None else None

            # if there are no children, pass the share of this leaf URL pattern upwards (recursion anchor #1)
            if not bool(children):
                return self_share, False

            # process child prefixes
            _children = list(children)

            # if _children already contains an accepted solution, pass it upwards (recursion anchor #2)
            for share_by_path, is_done in _children:
                if is_done:
                    return share_by_path, True

            # collect shares of children plus percentage of URLs at this inner node
            shares_of_self_and_children = [share_of_child for share_of_child, _ in _children]
            if share is not None:
                shares_of_self_and_children.append(self_share)
            shares_of_self_and_children = pd.concat(shares_of_self_and_children).sort_values(ascending=False)

            # if there is no reason to make a choice yet (total share of prefix currently being investigated is less than
            # target), pass it upwards (recursion anchor #3)
            if shares_of_self_and_children.sum() < target_url_mass:
                return pd.Series({p: shares_of_self_and_children.sum()}), False
            else:
                # if the total share of URLs at this node is greater than the target, make the choice which prefixes to keep
                nearest_index_to_target = pd.Index(shares_of_self_and_children.cumsum().values).get_loc(target_url_mass,
                                                                                                        method="nearest")
                selection = shares_of_self_and_children.iloc[:nearest_index_to_target + 1]
                return selection, True
        # determine good prefixes
        prefixes_selected, _ = trie.traverse(traverse_callback)
        prefixes_selected_as_string = prefixes_selected.index.map(lambda tups: "/".join(tups)).values

        # write details to disk for reference
        trie_as_dict = {k:v for k, v in trie.items()}
        with (self.analysis_output_location / "trie_of_generalized_hyperlink_to_urls.json").open("w") as f:
            json.dump(trie_as_dict, f)
        prefixes_selected.to_csv(self.analysis_output_location / "trie_selected_prefixes.csv")

        # make the selection
        hyperlinks_with_generalized_urls = pd.Series(generalized_urls.values, index=hyperlinks_index)
        hyperlinks_to_keep = hyperlinks_with_generalized_urls.map(lambda gen_url: any(gen_url.startswith(prefix) for prefix in prefixes_selected_as_string))
        return hyperlinks_to_keep

    def run(self, live_objects: dict):
        page_infos = read_dataframe(self.page_infos_file)  # type: pd.DataFrame
        sentences = read_dataframe(self.sentences_file)  # type: pd.DataFrame
        hyperlinks = read_dataframe(self.hyperlinks_file)  # type: pd.DataFrame

        self.logger.info(f"{len(hyperlinks)} hyperlinks exist before postprocessing.")

        # for filtering and deduplication, urlsplit all hyperlink target URLs
        to_url_split = url_normalize_series(hyperlinks[TO_URL])[1]

        # for reference, write the target netloc's and their frequency to disk
        with open(os.path.join(self.analysis_output_location, "to_url_netlocs__before_postprocessing.txt"), "w") as f:
            df = to_url_split["netloc"].value_counts().to_frame("count")
            f.write(tabulate(df, headers="keys"))

        # keep only http or https schemes (get rid of mailto:, tel: and friends)
        blacklisted_scheme = to_url_split["scheme"].map(lambda url: url not in ["http", "https"])
        self.logger.info(f"{blacklisted_scheme.value_counts().get(True, 0)} links will be removed based on their URL scheme.")

        # remove "listeditor@domain.com" etc.
        blacklisted_netloc_characters = to_url_split["netloc"].str.contains("@")
        self.logger.info(f"{blacklisted_netloc_characters.value_counts().get(True, 0)} links will be removed because their netloc contains unwanted characters.")

        # check netloc blacklist and url prefix blacklist
        if self.netloc_blacklist:
            blacklisted_netloc = to_url_split["netloc"].map(lambda url: any(url.endswith(b) for b in self.netloc_blacklist))
            self.logger.info(f"{blacklisted_netloc.value_counts().get(True, 0)} links will be removed based on {len(self.netloc_blacklist)} entries in the netloc blacklist.")
        else:
            blacklisted_netloc = pd.Series(False, index=hyperlinks.index)
        if self.normalized_url_prefix_blacklist:
            blacklisted_prefix = hyperlinks[TO_URL_NORMALIZED].map(lambda url: any(url.startswith(b) for b in self.normalized_url_prefix_blacklist))
            self.logger.info(f"{blacklisted_prefix.value_counts().get(True, 0)} links will be removed based on {len(self.normalized_url_prefix_blacklist)} entries in the URL prefix blacklist.")
        else:
            blacklisted_prefix = pd.Series(False, index=hyperlinks.index)
        # check url whitelist (mutually exclusive to netloc blacklist and url prefix blacklist)
        if self.netloc_whitelist:
            whitelisted_netloc = to_url_split["netloc"].map(lambda url: any(url.endswith(w) for w in self.netloc_whitelist))
            self.logger.info(f"{whitelisted_netloc.value_counts().get(True, 0)} links will be kept based on {len(self.netloc_whitelist)} entries in the netloc whitelist.")
        else:
            whitelisted_netloc = None

        # use netloc whitelist if present, otherwise apply netloc and url prefix blacklist
        to_drop_based_on_scheme_netloc = (blacklisted_netloc | blacklisted_prefix) if whitelisted_netloc is None else ~whitelisted_netloc
        to_drop_based_on_scheme_netloc |= blacklisted_scheme | blacklisted_netloc_characters
        hyperlinks.drop(hyperlinks.index[to_drop_based_on_scheme_netloc], inplace=True)
        self.logger.info(f"Overall, {to_drop_based_on_scheme_netloc.value_counts().get(True, 0)} links were removed based on scheme or netloc white- and blacklists.")

        # further suffix and substring blacklists
        if self.normalized_url_suffix_blacklist:
            blacklisted_suffix = hyperlinks[TO_URL_NORMALIZED].map(lambda url: any(url.endswith(b) for b in self.normalized_url_suffix_blacklist))
            self.logger.info(f"{blacklisted_suffix.value_counts().get(True, 0)} links will be removed based on {len(self.normalized_url_suffix_blacklist)} entries in the URL suffix blacklist.")
        else:
            blacklisted_suffix = pd.Series(False, index=hyperlinks.index)
        if self.normalized_url_substring_blacklist:
            blacklisted_substring = hyperlinks[TO_URL_NORMALIZED].map(lambda url: any(sub in url for sub in self.normalized_url_substring_blacklist))
            self.logger.info(f"{blacklisted_substring.value_counts().get(True, 0)} links will be removed based on {len(self.normalized_url_substring_blacklist)} entries in the URL substring blacklist.")
        else:
            blacklisted_substring = pd.Series(False, index=hyperlinks.index)
        # apply further blacklist filters
        to_drop_based_on_suffix_substring = blacklisted_suffix | blacklisted_substring
        hyperlinks.drop(hyperlinks.index[to_drop_based_on_suffix_substring], inplace=True)
        self.logger.info(f"Overall, {to_drop_based_on_suffix_substring.value_counts().get(True, 0)} links were removed based on URL substring and suffix blacklists.")

        # commence anchor text related postprocessing
        hyperlinks_with_surrounding_sent = hyperlinks.join(sentences, on=[URL_NORMALIZED, SENTENCE_IDX])

        # harmonize hyperlink anchor text spans: remove parens, dashes, whitespace included at the span boundaries, but
        # include any alphanum, digit or selected symbol characters that precede/follow the anchor text
        if self.harmonize_span_boundaries:
            hyperlinks_with_surrounding_sent[CHARS_START] = hyperlinks_with_surrounding_sent.apply(
                lambda row: harmonize_span_boundary(row[SENTENCE], row[CHARS_START],
                                                                             boundary="start"), axis=1, result_type="reduce")
            hyperlinks_with_surrounding_sent[CHARS_END] = hyperlinks_with_surrounding_sent.apply(
                lambda row: harmonize_span_boundary(row[SENTENCE], row[CHARS_END], boundary="end"),
                axis=1, result_type="reduce")

        # add the resulting anchor text
        hyperlinks_with_surrounding_sent[ANCHOR_TEXT] = hyperlinks_with_surrounding_sent.apply(lambda v: v[SENTENCE][v[CHARS_START]:v[CHARS_END]], axis=1)

        # remove:
        # - all groups of hyperlinks with the same anchor text on the same page (happens rarely, but high likelihood of garbage)
        # - links with whitespace-only anchors
        has_same_anchor_on_same_page = hyperlinks_with_surrounding_sent.duplicated(subset=[URL_NORMALIZED, ANCHOR_TEXT], keep=False)
        has_empty_anchor = hyperlinks_with_surrounding_sent[ANCHOR_TEXT].str.strip().map(len) == 0
        to_drop_based_on_anchor = has_same_anchor_on_same_page | has_empty_anchor
        self.logger.info(f"Of {len(to_drop_based_on_anchor)} links, {to_drop_based_on_anchor.value_counts().get(True, 0)} were removed due to duplicate anchor on the same page or empty anchor text.")
        hyperlinks_with_surrounding_sent.drop(hyperlinks_with_surrounding_sent.index[to_drop_based_on_anchor], inplace=True)

        # heuristically detect hyperlinks which are headlines and remove them
        is_headline = self._heuristic_headline_detection(hyperlinks_with_surrounding_sent)
        hyperlinks_with_surrounding_sent.drop(hyperlinks_with_surrounding_sent.index[is_headline], inplace=True)
        self.logger.info(f"Of {len(is_headline)} links, {is_headline.value_counts().get(True, 0)} were suspected to be headlines and were removed.")

        # Groups of hyperlinks which point to the same target page using the same anchor text may be noise (read more-type
        # links), but may also be valid mentions of events ("Watergate", "coronavirus outbreak", ...) so we offer the
        # possibility to not remove them entirely. In initial experiments, removing these groups entirely worked better
        # though, so doing that is advised.
        is_anchor_target_duplicate = hyperlinks_with_surrounding_sent[[TO_URL_NORMALIZED, ANCHOR_TEXT]].duplicated(keep=False)
        anchor_target_duplicates = hyperlinks_with_surrounding_sent.loc[is_anchor_target_duplicate]
        # We count how many duplicates exist, and keep only those which are relatively rare duplicates according to a
        # threshold
        anchor_target_occurrences = anchor_target_duplicates.groupby([ANCHOR_TEXT, TO_URL_NORMALIZED]).size()
        is_frequent_duplicate = anchor_target_duplicates.apply(lambda row: anchor_target_occurrences.loc[(row[ANCHOR_TEXT], row[TO_URL_NORMALIZED])] >= self.max_anchor_target_duplicate_occurrences, axis=1)
        self.logger.info(f"Of {len(is_frequent_duplicate)} links, {is_frequent_duplicate.value_counts().get(True, 0)} were removed because their anchor and target appears too frequently.")
        to_drop_for_frequent_anchor_target_duplicates = pd.Series(False, index=hyperlinks_with_surrounding_sent.index) | is_frequent_duplicate
        hyperlinks_with_surrounding_sent.drop(hyperlinks_with_surrounding_sent.index[to_drop_for_frequent_anchor_target_duplicates], inplace=True)

        # apply hyperlinks anchor blacklists
        if self.anchor_blacklist_regex is not None:
            blacklisted_by_anchor_text = hyperlinks_with_surrounding_sent[ANCHOR_TEXT].map(lambda a: self.anchor_blacklist_regex.match(a) is not None)
            hyperlinks_with_surrounding_sent.drop(hyperlinks_with_surrounding_sent.index[blacklisted_by_anchor_text], inplace=True)
            self.logger.info(f"Of {len(blacklisted_by_anchor_text)} links, {blacklisted_by_anchor_text.value_counts().get(True, 0)} were removed based on anchor blacklist rules.")

        # drop extra columns again which were added for anchor text processing
        hyperlinks = hyperlinks_with_surrounding_sent.drop(columns=[ANCHOR_TEXT, SENTENCE])

        # commence distributional analysis / filtering:
        # (1) retain only those links whose href URLs have a frequently occurring pattern
        if self.to_url_pattern_target_mass is not None:
            # it's best to re-normalize URLs here, as a lot of hyperlinks have been dropped midway
            # TODO this df could be kept/updated for improved efficiency
            to_url_split = url_normalize_series(hyperlinks[TO_URL])[1]
            to_keep_based_url_pattern = self._url_prefix_analysis(hyperlinks.index, to_url_split, self.to_url_pattern_target_mass)
            hyperlinks.drop(hyperlinks.index[~to_keep_based_url_pattern], inplace=True)
            self.logger.info(f"{to_keep_based_url_pattern.value_counts().get(False, 0)} hyperlinks ({100 * (1 - (to_keep_based_url_pattern.sum() / len(to_keep_based_url_pattern))):.2f}%) were removed based on their infrequent URL pattern.")

        # (2) A small set of pages tend to be linked to very frequently (either because they truly describe a highly
        # significant events, or because they're hub pages, or because our choice of trimming query arguments from href
        # URLs incorrectly produces large clusters). With `in_degree_long_tail_cutoff`, links to pages with
        # disproportionately high in-degree can be removed, keeping only links to rarer pages. Similarly, pages with
        # high out-degree are suspicious and will be removed.
        if self.in_degree_long_tail_cutoff < 1.0 or self.out_degree_long_tail_cutoff < 1.0:
            page_has_high_in_degree, page_has_high_out_degree = self._page_degree_analysis(hyperlinks,
                                                                                           self.in_degree_long_tail_cutoff,
                                                                                           self.out_degree_long_tail_cutoff)
            high_in_degree_pages = page_has_high_in_degree.loc[page_has_high_in_degree]
            high_out_degree_pages = page_has_high_out_degree.loc[page_has_high_out_degree]

            # save the hyperlinks which were removed here for analysis
            removed_for_high_in_degree = hyperlinks.loc[hyperlinks[TO_URL_NORMALIZED].isin(high_in_degree_pages.index)]
            removed_for_high_in_degree.to_csv(self.analysis_output_location / "removed_for_high_in_degree.csv")
            hyperlinks.drop(removed_for_high_in_degree.index, inplace=True)

            removed_for_high_out_degree = hyperlinks.loc[hyperlinks[URL_NORMALIZED].isin(high_out_degree_pages.index)]
            removed_for_high_out_degree.to_csv(self.analysis_output_location / "removed_for_high_out_degree.csv")
            hyperlinks.drop(removed_for_high_out_degree.index, inplace=True, errors="ignore")   # could be that some pages were already removed for high in-degree

            self.logger.info(f"{len(removed_for_high_in_degree) + len(removed_for_high_out_degree)} links were removed for in-degree or out-degree.")

        # write 'em to disk
        pages_involved_in_hyperlinks = pd.Index(set(hyperlinks[URL_NORMALIZED]) | set(hyperlinks[TO_URL_NORMALIZED]), name=URL_NORMALIZED)
        scraped_pages_involved = page_infos.index.intersection(pages_involved_in_hyperlinks)
        self.logger.info(f"{len(hyperlinks)} hyperlinks remain after postprocessing. {len(page_infos) - len(scraped_pages_involved)} now irrelevant pages were removed ({len(scraped_pages_involved)} remain).")
        page_infos = page_infos.loc[scraped_pages_involved]
        sentences = sentences.loc[scraped_pages_involved]

        write_dataframe(page_infos, self.page_infos_postprocessed_file)
        write_dataframe(sentences, self.sentences_postprocessed_file)
        write_dataframe(hyperlinks, self.hyperlinks_postprocessed_file)


component = HyperlinksPostProcessingStage
