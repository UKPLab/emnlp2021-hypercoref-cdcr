from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse
from numpy.testing import assert_almost_equal
from overrides import overrides
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from python import URL
from python.data.pipeline.postprocessing.page_deduplication.base import IdentifyPageDuplicates
from python.util.scipy import batch_pairwise_dot


class TfidfIdentifyPageDuplicates(IdentifyPageDuplicates):
    """
    Exhaustive pairwise comparison to find clusters of duplicate pages.
    """
    def __init__(self,
                 max_chars: Optional[int] = None,
                 threshold: float = 0.4,
                 n_gram: int = 13):
        self.clustering = DocumentClustering(threshold)

        # we use HashingVectorizer instead of CountVectorizer to circumvent scalability issues with large datasets
        p = Pipeline([("feature_extractor", FunctionTransformer(get_documents_from_X)),
                      ("identifier_and_cluster_id", FeatureUnion([
                          ("get_doc_id", FunctionTransformer(get_url_from_X)),
                          ("tfidf_clustering", Pipeline([
                              ("get_text", DocumentTextExtraction(max_chars)),
                              ("vectorizer", HashingVectorizer(lowercase=True,
                                                               stop_words="english",
                                                               ngram_range=(n_gram, n_gram))),
                              ("tfidf", TfidfTransformer()),
                              ("clustering", self.clustering)
                          ]))
                      ])),
                      ("prettify", DocumentClusteringPrettifier())])
        self.p = p

    @overrides
    def _find_duplicates(self, documents: pd.Series):
        self.clustering.skip_clustering = False
        return self.p.transform(documents)

    @overrides
    def fit(self, documents: pd.Series):
        self.clustering.skip_clustering = True
        self.p.fit(documents)


class DocumentClustering(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.skip_clustering = False

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X_csr: csr_matrix, y=None):
        # optionally skip this step so that we can fit the TF-IDF part of the pipeline without having to run time-consuming
        # clustering here afterwards
        if self.skip_clustering:
            return np.arange(X_csr.shape[0]).reshape((-1, 1))

        assert scipy.sparse.issparse(X_csr)

        # if there are no features in the matrix (all terms too rare?), assume the documents are unique (== singletons)
        if X_csr.getnnz() == 0:
            return np.arange(X_csr.shape[0]).reshape((-1, 1))

        # identify documents which have a TFIDF vector of zero: these have highly unique ngrams, i.e. no duplicates to
        # be expected -> make them singletons in the clustering
        row_agg = abs(X_csr) * np.ones((X_csr.shape[1], 1))
        idx_of_non_zero_docs, _ = row_agg.nonzero()
        X_nonzero_rows_csr = X_csr[idx_of_non_zero_docs, :]

        # check one document vector to make sure vectors are normalized
        assert_almost_equal(norm(X_nonzero_rows_csr[0, :], axis=1).item(0), 1.0,
                            err_msg="Expected TF-IDF vectors coming from sklearn to be L2-normalized!")

        Y_similarity = batch_pairwise_dot(X_nonzero_rows_csr)
        Y_distance = np.maximum(0, 1 - Y_similarity)  # we need the maximum for numerical stability when Y_inverted has elements == 1.0
        Z = linkage(Y_distance, method="average")
        nonzero_clusters = fcluster(Z, t=self.threshold, criterion="distance")

        # create list of clusters: start with list of unique singleton cluster IDs into which the IDs of the duplicate
        # documents are inserted
        first_singleton_cluster_id = np.max(nonzero_clusters) + 1
        clusters = np.arange(first_singleton_cluster_id, first_singleton_cluster_id + X_csr.shape[0])
        clusters[idx_of_non_zero_docs] = nonzero_clusters

        return clusters.reshape((-1, 1))


class DocumentTextExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, max_chars: Optional[int] = None):
        self.max_chars = max_chars

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        if self.max_chars is not None:
            return [text[:self.max_chars] for _, text in X]
        else:
            return [text for _, text in X]


def get_url_from_X(X):
    return np.array([url for url, _ in X]).reshape((-1, 1))


def get_documents_from_X(X: pd.Series):
    return list(X.iteritems())


class DocumentClusteringPrettifier(BaseEstimator, TransformerMixin):

    def _convert(self, X):
        # X is a np.array of shape (num_docs, 2) with document identifier and cluster id
        clustering = pd.Series(X[:, 1].astype(int), index=pd.Index(X[:, 0], name=URL), name="cluster")
        return clustering

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return self._convert(X)

    def predict(self, X):
        return self._convert(X)
