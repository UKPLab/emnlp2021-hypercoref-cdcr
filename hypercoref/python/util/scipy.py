import itertools
import tempfile
from pathlib import Path
from typing import Set, List

import joblib
import numpy as np
from joblib import delayed, Parallel
from more_itertools import chunked


def batch_pairwise_dot(arr, batch_size=1024):
    """
    Computes dot product between all pairs in arr, returned as a vector-form distance matrix as returned by scipy's
    squareform (i.e. overall, very similar to pdist). Computation is performed in batches to avoid "RuntimeError: nnz
    of the result is too large" when computing pairwise distances by multiplying the TF-IDF matrix with its transpose.
    """
    n, e = arr.shape

    # crucial here: the order of pairs generated by itertools.combinations corresponds to the order of pairs used in
    # scipy vector-form distance matrices, meaning the computed vector distances between each pair can be appended
    # without further modification to produce the desired end result
    pairs = itertools.combinations(range(n), 2)
    num_pairs = (n * (n - 1)) // 2

    progress_iterator = chunked(pairs, batch_size)
    distances = np.empty((num_pairs), dtype=np.float32)
    for i, list_of_pairs in enumerate(progress_iterator):
        a_indices, b_indices = zip(*list_of_pairs)
        mat_a = arr[list(a_indices)]
        mat_b = arr[list(b_indices)]
        dot = mat_a.multiply(mat_b).sum(axis=1).reshape((1, -1))
        distances[i * batch_size:i * batch_size + mat_a.shape[0]] = dot
    return distances


def parallel_batch_pairwise_dot(arr, batch_size=1024 * 1024, n_jobs: int = -1):
    """
    Multiprocessversion of `batch_pairwise_dot`. Not as fast as the single thread version, kept for reference.
    """
    n, e = arr.shape

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csr_destination = tmp_path / "csr.pkl"
        joblib.dump(arr, csr_destination)

        pairs = itertools.combinations(range(n), 2)
        batches = chunked(pairs, batch_size)

        batch_files = []
        for i, batch in enumerate(batches):
            destination = tmp_path / f"batch_{i}.pkl"
            joblib.dump(batch, destination)
            batch_files.append(destination)

        def _compute(batch_filename, arr_filename):
            batch = joblib.load(batch_filename)
            arr = joblib.load(arr_filename)  # memmap is not supported for CSR matrices :(
            a_indices, b_indices = zip(*batch)
            mat_a = arr[list(a_indices)]
            mat_b = arr[list(b_indices)]
            dot = mat_a.multiply(mat_b).sum(axis=1)
            return dot

        jobs = [delayed(_compute)(batch_file, csr_destination) for batch_file in batch_files]
        list_of_arrs = Parallel(n_jobs=n_jobs)(jobs)

        distances = np.concatenate(list_of_arrs)
        return np.asarray(distances).ravel()


def sparse_closure(buckets_by_document: List[List[int]], documents_by_bucket: List[List[int]]):
    """
    Computes transitive closure via some BFS/DFS mixture on a sparse binary matrix of minhash buckets. Imagine a
    (sparse) matrix where rows represent documents, columns represent buckets and binary values indicate whether a
    document is part of a bucket. This method accepts such a matrix in a very condensed form, namely a list of list of
    bucket indices (document -> buckets mapping) and a list of list of document indices (bucket -> documents mapping).
    Returns the closure / clustering as a list of list of document indices.
    """
    docs_not_visited = set(range(len(buckets_by_document)))
    clustering = []

    def get_buckets(rows: Set) -> Set:
        return set(bucket for row in rows for bucket in buckets_by_document[row])
    def get_documents(col) -> Set:
        return set(documents_by_bucket[col])

    while docs_not_visited:
        seed_doc = docs_not_visited.pop()
        docs_visited = {seed_doc}                   # this is the current cluster in the making
        buckets_visited = set()
        bucket_queue = get_buckets({seed_doc})      # find buckets this document is part of (horizontal matrix lookup)
        while bucket_queue:
            bucket = bucket_queue.pop()
            buckets_visited.add(bucket)
            docs = get_documents(bucket)            # find documents in this bucket (vertical matrix lookup)
            connected_docs_not_yet_visited = docs - docs_visited
            if connected_docs_not_yet_visited:
                bucket_queue |= get_buckets(connected_docs_not_yet_visited) - buckets_visited
                docs_visited |= docs
        docs_not_visited -= docs_visited
        clustering.append(list(docs_visited))

    return clustering