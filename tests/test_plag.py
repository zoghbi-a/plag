import numpy as np
import unittest

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import plag


class plagPythonTest(unittest.TestCase):
    """Test the python code"""


    def test_plag__init(self):
        
        n = 8
        t = np.arange(n, dtype=np.double)
        fqL = np.array([1./8,0.5])
        x = np.random.randn(n) + 4
        inp = np.array([1.])
        p = plag.psd(t, x, x*0+0.1, 1.0, fqL)
        P = plag.PLag('psd', [t]*2, [x]*2, [x*0+0.1]*2, 1.0, fqL)
        assert(P.nmod == 2)
        np.testing.assert_allclose(
                P.logLikelihood(inp),2*p.logLikelihood(inp))


    def test_plag_psd_logLikelihood(self):
        """logLikelihood in PLagCython
        """
        n = 12
        t = np.arange(n, dtype=np.double)
        fqL = np.array([1./12,0.5])
        x   = np.random.randn(n) + 4
        p1  = plag._plag.psd(t, x, x*0+0.1, 1.0, fqL, 2, 0)
        p2  = plag.psd(t, x, x*0+0.1, 1.0, fqL)
        inp = np.array([1.])
        np.testing.assert_almost_equal(
            p1.logLikelihood(inp, 1, 0), p2.logLikelihood(inp))


    def test_plag_psd_dLikelihood(self):
        """Test the gradient from dLogLikelihood
        of plag.psd vs _plag.psd
        """
        np.random.seed(3094)
        n, dt = 4, 1.0
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n)*2 + 4
        xe = x*0+0.01
        fqL = np.array([0.25,0.5])
        inpars = np.array([1.])
        p1 = plag._plag.psd(t, x, xe, dt, fqL, 2, 0)
        p2 = plag.psd(t, x, xe, dt, fqL)
        l1, g1, h1 = p1.dLogLikelihood(np.array([1.]))
        l2, g2, h2 = p2.dLogLikelihood(np.array([1.]))
        np.testing.assert_almost_equal(l1, l2)
        np.testing.assert_array_almost_equal(g1, g2)
        np.testing.assert_array_almost_equal(h1, h2)


    def test_plag_dLogLikelihood(self):
        """Test the dLogLikelihood of the plag container
        """
        np.random.seed(395)
        n, dt = 12, 1.0
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n)*2 + 4
        xe = x*0+0.01
        fqL = np.array([0.25,0.5])
        inpars = np.array([1.])
        ind = [range(i,i+4) for i in range(0,n,4)]
        T = [t[i] for i in ind]
        X = [x[i] for i in ind]
        Xe = [xe[i] for i in ind]
        p = [plag.psd(t, x, xe, dt, fqL) for t,x,xe in zip(T,X,Xe)]
        P = plag.PLag('psd', T, X, Xe, dt, fqL)

        res1 = [ p[i].dLogLikelihood(inpars) for i in range(3)]
        l1   = sum([r[0] for r in res1])
        g1   = np.sum([r[1] for r in res1],0)
        h1   = np.sum([r[2] for r in res1],0)
        l2, g2, h2 = P.dLogLikelihood(inpars)
        np.testing.assert_almost_equal(l1, l2)
        np.testing.assert_array_almost_equal(g1, g2)
        np.testing.assert_array_almost_equal(h1, h2)

