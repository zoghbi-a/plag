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
        raise ValueError
        assert(m.n == 4)
        assert(m.mu == np.sum(t)/4.)
        assert(m.npar == 1)
        assert(m.nU == len(np.unique(t[:,None]-t[None,:])))