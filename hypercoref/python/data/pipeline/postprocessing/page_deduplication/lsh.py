from typing import Optional

import pandas as pd
from lsh.minhash import MinHasher
from overrides import overrides
from tqdm import tqdm

from python import URL_NORMALIZED
from python.data.pipeline.postprocessing.page_deduplication.base import IdentifyPageDuplicates, CLUSTER_ID
from python.data.pipeline.postprocessing.page_deduplication.improved_lsh_cache import Cache
from python.util.scipy import sparse_closure


class LshIdentifyPageDuplicates(IdentifyPageDuplicates):
    """
    Applies locality sensitive hashing for approximate and scalable deduplication of pages.
    See https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html
    """
    def __init__(self,
                 max_chars: Optional[int] = None,
                 seeds: int = 256,
                 bands: int = 16,
                 char_n_gram: int = 5):
        if seeds % bands != 0:
            raise ValueError(f"bands ({bands}) must be a multiple of seeds ({seeds}).")

        hasher = MinHasher(seeds=seeds, char_ngram=char_n_gram, hashbytes=8, random_state=0)
        self.cache = Cache(hasher, bands)

        self.max_chars = max_chars

    @overrides
    def _find_duplicates(self, documents: pd.Series):
        for doc_id, doc in tqdm(documents.iteritems(),
                                desc="Hashing documents",
                                mininterval=10,
                                unit="doc",
                                total=len(documents)):
            if self.max_chars is not None:
                doc = doc[:self.max_chars]
            self.cache.add_doc(doc, doc_id)

        # what we can get from the bins: for each band, a set of documents which have the same value in that band
        doc_url_to_doc_idx = {}
        buckets_by_document = []
        documents_by_bucket = []
        bucket_id = 0
        for bin in tqdm(self.cache.bins,
                        desc="Findings documents in bins",
                        mininterval=10,
                        unit="bin"):
            gen = (bucket for bucket in bin.values() if len(bucket) > 1)
            for bucket in gen:
                doc_idxs_in_bucket = []
                for url in bucket:
                    if not url in doc_url_to_doc_idx:
                        doc_url_to_doc_idx[url] = len(doc_url_to_doc_idx)
                        buckets_by_document.append([])
                    doc_idx = doc_url_to_doc_idx[url]
                    doc_idxs_in_bucket.append(doc_idx)
                    buckets_by_document[doc_idx].append(bucket_id)
                documents_by_bucket.append(doc_idxs_in_bucket)
                bucket_id += 1
        clusters_of_potential_duplicates = sparse_closure(buckets_by_document, documents_by_bucket)
        # each set in `clusters_of_potential_duplicates` now contains documents which were identified as potential
        # duplicates via LSH
        doc_idx_to_doc_url = {idx: url for url, idx in doc_url_to_doc_idx.items()}

        # create a Series mapping document URLs to their candidate cluster ID: first, assign each document to a
        # singleton cluster...
        clustering = documents.reset_index()[URL_NORMALIZED].reset_index().rename(columns={"index": CLUSTER_ID}).set_index(URL_NORMALIZED)[CLUSTER_ID]
        clustering += len(clusters_of_potential_duplicates)
        # ...then overwrite the cluster ID's of non-singleton documents
        for cluster_id, doc_idxs in enumerate(clusters_of_potential_duplicates):
            clustering.loc[[doc_idx_to_doc_url[idx] for idx in doc_idxs]] = cluster_id

        return clustering