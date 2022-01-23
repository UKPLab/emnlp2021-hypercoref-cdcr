from unittest import TestCase

from numpy.linalg import norm
from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

from python.util.scipy import batch_pairwise_dot, parallel_batch_pairwise_dot


class TestScipy(TestCase):

    def test_batch_pairwise_dot(self):
        rs = RandomState(0)
        a = rs.rand(1000, 5)
        a = a / norm(a, axis=1).reshape((-1, 1))
        a = csr_matrix(a)

        cosine_sim = a * a.transpose()
        cosine_sim.setdiag(0)
        expected = squareform(cosine_sim.todense())
        actual = batch_pairwise_dot(a, batch_size=83)
        assert_array_almost_equal(expected, actual)

        actual_parallel = parallel_batch_pairwise_dot(a, batch_size=83, n_jobs=2)
        assert_array_almost_equal(expected, actual_parallel)