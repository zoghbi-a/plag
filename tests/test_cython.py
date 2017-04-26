import numpy as np
import unittest

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import plag


class plagCythonTest(unittest.TestCase):
    """Test the cython code"""


    def test_PLagBase_init(self):
        """Test the initialization of clagBase
        """
        t = np.arange(4, dtype=np.double)
        m = plag._plag.PLagBase(t, t, t, 1.0, 1)
        assert(m.n == 4)
        assert(m.mu == np.sum(t)/4.)
        assert(m.npar == 1)
    

    def test_PLagBase_covariance(self):
        """Test covariance_kernel and Covairance"""
        t = np.arange(4, dtype=np.double)
        m = plag._plag.PLagBase(t, t, t, 1.0, 1)
        tu, iu = np.unique(t[:,None]-t[None,:], return_inverse=1)
        assert(m.nU == len(tu))

        cov = m._get_cov_matrix(np.array([1.]))

        # cov-related arrays #
        uarr = m._get_cov_arrays()
        np.testing.assert_array_almost_equal(tu, uarr[0])
        np.testing.assert_array_almost_equal(np.zeros_like(tu), uarr[1])
        np.testing.assert_array_almost_equal(iu, uarr[2])

        # cov-matrix #
        np.testing.assert_array_almost_equal(np.zeros((4, 4)) * 1.0, cov)

    


