import pathlib
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Callable

import pandas as pd
import requests
import urllib3
from joblib import delayed, Parallel
from lxml import etree
from tqdm import tqdm
from warcio import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from python import *
from python.common_components import CORENLP
from python.common_components.corenlp import CoreNlp
from python.common_components.document_cleaner import DocumentCleaner
from python.common_components.document_cleaner.goose import GooseDocumentCleaner
from python.common_components.document_cleaner.newspaper import NewspaperDocumentCleaner
from python.data.pipeline.warc_extraction.hyperlink_content_extractor import HyperlinkAndContentExtractor
from python.pipeline import MAX_CORES
from python.pipeline.pipeline import PipelineStage
from python.util.util import get_logger, url_normalize_series, write_dataframe


def _download_clean_extract(batch: int,
                            df: pd.DataFrame,
                            doc_cleaner: DocumentCleaner,
                            content_extractor: HyperlinkAndContentExtractor,
                            corenlp: CoreNlp,
                            full_content_extraction: bool,
                            wait_between_requests_seconds: float,
                            logging_config: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Performs the following steps for each given page (with metadata):
    - Obtains a crawled version of the page by downloading the specific warc.gz snippet from a webserver (most likely Common Crawl)
    - Cleans the page markup (remove toolbars, etc.)
    - Retrieves basic webpage information
    - Retrieves and resolves hyperlinks inside the page text
    - If desired, additionally:
        - Cleans and retrieves page sentences
        - Cleans start- and end offsets of, and retrieves hyperlinks
    :param batch: batch identifier
    :param df: a section of a dataframe with URLs to web pages, the corresponding .warc.gz URL for each URL, start offsets and lengths of the pages
    :param doc_cleaner: HTML markup cleaner
    :param content_extractor: hyperlink and page sentence extractor
    :param full_content_extraction: if `True`, extracts page pages
    :param wait_between_requests_seconds: seconds to wait in between requests
    :param logging_config: config for the logger object - the actual object currently cannot be passed here with the
                           loky backend, see https://github.com/joblib/joblib/issues/1017 (passing the logger name to
                           retrieve the object here also does not work, I tried)
    :return: information about webpages, sentences and hyperlinks
    """
    logger = get_logger(logging_config)
    TIMEOUT = 15    # seconds

    batch_page_infos = []
    batch_sentences = []
    batch_hyperlinks = []

    # use the batch identifier to let processes start requesting in staggered fashion, instead of all at once at the start
    time_of_last_query = time.time() + batch

    # re-establishing HTTP/S connections quickly becomes the bottleneck, therefore use a session object (which uses keep-alive)
    session = requests.Session()

    for tup in tqdm(df.itertuples(), total=len(df), mininterval=10, position=batch, unit="page", desc=f"Batch {batch:02}"):
        d = tup._asdict()
        page_url = d["Index"]
        warc_url = d[FILENAME]
        offset, length = d[OFFSET], d[LENGTH]

        # download only the bytes we need
        offset_end = offset + length - 1

        # apply rate limiting: make sure at least wait_between_requests_seconds seconds are between each request
        now = time.time()
        time_to_sleep = max(0, wait_between_requests_seconds - (now - time_of_last_query))
        time.sleep(time_to_sleep)

        # obtain markup, clean document
        try:
            # TODO it were cool to groupby filename and submit only one request per file using multiple ranges in the
            #   HTTP header (see [1]). However, Amazon S3 does not support multiple ranges in a request (see [2]), it
            #   just returns the whole file instead. So we cannot do that right now.
            #   [1] https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Range
            #   [2] https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html#API_GetObject_RequestSyntax
            with session.get(warc_url, headers={"Range": f"bytes={offset}-{offset_end}"}, stream=True, timeout=TIMEOUT) as r:
                # obtain HTML markup (ArchiveIterator detects WARC/ARC format, and content_stream will pick the right encoding)
                it = iter(ArchiveIterator(r.raw))
                record = next(it)
                raw_markup = record.content_stream().read()

                # close for safety (we use stream=True and have a pool of connections)
                r.close()
            # run document cleaner on it
            doc_cleaner.load_markup(page_url, raw_markup)
        except (requests.exceptions.RequestException, ArchiveLoadFailed, urllib3.exceptions.ReadTimeoutError, OSError, StopIteration) as e:
            logger.warning("Could not obtain " + page_url + " from " + warc_url + ".", e)
            continue
        except ValueError as e:
            logger.warning("Could not load markup for " + page_url + ".", e)
            continue
        finally:
            time_of_last_query = now

        # gather document-level information
        cleaned_text = doc_cleaner.cleaned_text
        page_info = {URL: page_url, TEXT: cleaned_text, CONTENT_CHARS: len(cleaned_text)}
        if full_content_extraction:
            page_info.update({TITLE: doc_cleaner.title,
                             PUBLISH_DATE: doc_cleaner.publish_date,
                             AUTHORS: doc_cleaner.authors,
                             TOP_IMAGE_URL: doc_cleaner.top_image_url})

        # sometimes, goose doesn't find a top node but several separate ones, account for that: make up an lxml node on the spot
        top_node = doc_cleaner.top_node
        if type(doc_cleaner.top_node) is list:
            dummy_top_node = etree.Element("div")
            for elmt in top_node.get_children():
                dummy_top_node.append(elmt)
            top_node = dummy_top_node

        # this happens sometimes
        if top_node is None:
            continue

        top_node.make_links_absolute(base_url=page_url, handle_failures="discard")

        # run the content extractor
        sentences, hyperlinks = content_extractor.extract(top_node, corenlp, logger, full_content_extraction)

        # add document identifier
        for s in sentences:
            s[URL] = page_url
        for h in hyperlinks:
            h[URL] = page_url

        batch_page_infos.append(page_info)
        batch_sentences += sentences
        batch_hyperlinks += hyperlinks

    session.close()
    corenlp.clean_up()

    return batch_page_infos, batch_sentences, batch_hyperlinks


class WarcExtractionStage(PipelineStage):
    """
    Given a list of URLs to retrieve from CommonCrawl, this stage downloads the pages, strips toolbars/menus, and runs
    sentence tokenization.
    TODO Sentence tokenization being applied already here  is a legacy implementation decision. Other than making
         scraped documents a bit easier to work with when debugging/developing the pipeline, it doesn't have any benefit
         here. It should be possible to move it into a later stage (TokenizeStage?) and refactor all affected stages on
         the way with "relative ease".
    """

    def __init__(self, pos, config, config_global, logger):
        super().__init__(pos, config, config_global, logger)

        self.full_content_extraction = config.get("full_content_extraction", False)

        markup_cleaners = {"python-goose": GooseDocumentCleaner, "newspaper3k": NewspaperDocumentCleaner}
        markup_cleaner = config.get("markup_cleaner", "newspaper3k")
        self.markup_cleaner = markup_cleaners[markup_cleaner]()  # type: DocumentCleaner
        self.logger.info(f"Using {markup_cleaner} for cleaning markup.")

        self.content_extractor = HyperlinkAndContentExtractor()

        self.wait_between_requests_seconds = config.get("wait_between_requests_seconds", 0.1)

        # output files
        self.page_index_file = None
        self.page_infos_file, self.sentences_file, self.hyperlinks_file = [pathlib.Path(self.stage_disk_location) / name for name in [PAGE_INFOS, SENTENCES, HYPERLINKS]]

    def files_produced(self) -> Dict[str, Path]:
        return {PAGE_INFOS: self.page_infos_file,
                SENTENCES: self.sentences_file,
                HYPERLINKS: self.hyperlinks_file}

    def requires_files(self, provided: Dict[str, Path]):
        self.page_index_file = provided[PAGE_INDEX]

    def run(self, live_objects: dict):
        page_index = pd.read_csv(self.page_index_file, index_col=0)

        # Create version of page_index where pages coming from the same WARC archive are grouped together, but the
        # groups are shuffled. The grouping might result in fewer cache misses on the S3 server we send requests to, or
        # so. Since we parallelize by splitting this DF in batches, the shuffling is important to ensure that each
        # batch roughly takes the same time to complete (we had previously sorted by filename and offset, and this led
        # to 2 or 3 slow batches which took 3-4 times as long as the rest, probably because older crawls are stored on
        # slower storage servers or so).
        # See https://stackoverflow.com/a/63542879 for the approach.
        page_index.sort_values(OFFSET, inplace=True)
        filenames = page_index[FILENAME].unique()
        rnd = random.Random(x=0)
        rnd.shuffle(filenames)
        page_index = page_index.reset_index().set_index(FILENAME).loc[filenames].reset_index().set_index(URL)

        # TODO try to use extra CoreNLP client here with only tokenize,ssplit and maybe keep tokens right away to save effort
        corenlp = live_objects[CORENLP]
        self.logger.info(f"Downloading and cleaning approximately {len(page_index)} pages...")

        max_cores = self.config_global[MAX_CORES]
        if max_cores > 1:
            # divide the given data evenly into batches, see https://stackoverflow.com/a/40428356
            n = page_index.shape[0]
            batch_size = max(max_cores, int(n / max_cores))
            batches = [page_index.iloc[i:i + batch_size, :] for i in range(0, n, batch_size)]

            jobs = [delayed(_download_clean_extract)(i,
                                                     b,
                                                     self.markup_cleaner,
                                                     self.content_extractor,
                                                     corenlp,
                                                     self.full_content_extraction,
                                                     self.wait_between_requests_seconds,
                                                     self.config_global["logging"]) for i, b in enumerate(batches)]
            list_of_tuples = Parallel(n_jobs=max_cores)(jobs)
        else:
            list_of_tuples = [_download_clean_extract(0,
                                                      page_index,
                                                      self.markup_cleaner,
                                                      self.content_extractor,
                                                      corenlp,
                                                      self.full_content_extraction,
                                                      self.wait_between_requests_seconds,
                                                      self.config_global["logging"])]
        self.logger.info("Done.")

        # unwrap list of lists from each document and create dataframes, and add normalized versions of all URLs
        page_infos_gen = (page_info for batch_page_infos, _, _ in list_of_tuples for page_info in batch_page_infos)
        page_infos = pd.DataFrame(page_infos_gen)
        if not page_infos.empty:
            page_infos[URL_NORMALIZED] = url_normalize_series(page_infos[URL])[0]
            page_infos.set_index(URL, inplace=True)

            # normalize publish dates if present - convert any timezone-aware date into UTC
            if PUBLISH_DATE in page_infos.columns:
                page_infos[PUBLISH_DATE] = pd.to_datetime(page_infos[PUBLISH_DATE], utc=True, errors="coerce")

        sentences_gen = (sentence for _, batch_sentences, _ in list_of_tuples for sentence in batch_sentences)
        sentences = pd.DataFrame(sentences_gen)
        if not sentences.empty:
            assert not page_infos.empty
            sentences[URL_NORMALIZED] = sentences[URL].map(page_infos[URL_NORMALIZED])
            sentences.set_index([URL, SENTENCE_IDX], inplace=True)

        hyperlinks_gen = (hyperlink for _, _, batch_hyperlinks in list_of_tuples for hyperlink in batch_hyperlinks)
        hyperlinks = pd.DataFrame(hyperlinks_gen)
        if not hyperlinks.empty:
            assert not page_infos.empty
            hyperlinks[URL_NORMALIZED] = hyperlinks[URL].map(page_infos[URL_NORMALIZED])
            hyperlinks[TO_URL_NORMALIZED], to_url_split = url_normalize_series(hyperlinks[TO_URL])

            # very simple filtering steps:
            # (1) remove links with broken href attribute (which will have empty scheme or netloc), for example https://www.bbc.com/sport/football/36535204 which contains `<a href="said:"` -> drop these
            # (2) remove page-internal links (anchor links, etc.)
            # (3) keep only links with valid span
            hyperlinks = hyperlinks.loc[(to_url_split["scheme"] != "") & \
                                        (to_url_split["netloc"] != "") & \
                                        (hyperlinks[URL_NORMALIZED] != hyperlinks[TO_URL_NORMALIZED])]
            if CHARS_START in hyperlinks.columns:
                hyperlinks = hyperlinks.loc[hyperlinks[CHARS_START] < hyperlinks[CHARS_END]]

        # having extracted the full content, we switch indexing to normalized URLs
        if self.full_content_extraction:
            self.logger.debug("Pages will now be indexed by their normalized URL.")
            page_infos = page_infos.reset_index().set_index(URL_NORMALIZED)
            assert page_infos.index.is_unique

            sentences = sentences.reset_index().set_index([URL_NORMALIZED, SENTENCE_IDX])
            sentences.drop(columns=URL, inplace=True)
            assert sentences.index.is_unique

        write_dataframe(page_infos, self.page_infos_file)
        write_dataframe(sentences, self.sentences_file)
        write_dataframe(hyperlinks, self.hyperlinks_file)

        self.logger.info(f"Wrote infos on {len(page_infos)} pages with in total {len(sentences)} sentences and {len(hyperlinks)} valid cross-page hyperlinks.")


component = WarcExtractionStage
