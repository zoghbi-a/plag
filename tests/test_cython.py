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


    def test_PLagBase_logLikelihood(self):
        """Test the loglikelihood of PLagBase with direct Cov
        """
        import scipy.linalg as alg
        np.random.seed(394)
        n, dt = 4, 1.0
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n)*2 + 4
        C = np.random.randn(n**2).reshape((n,n))
        C += C.T + np.identity(4)*3
        C = C.T # make it fortran contiguous
        p = plag._plag.PLagBase(t, x, x*0, dt, 1)
        
        x = x-x.mean()
        Ci = alg.inv(C)
        chi2 = np.dot(x, np.dot(Ci, x))
        logLike_2 = -0.5 * ( chi2 + np.log(alg.det(C)) + n*np.log(2*np.pi) )

        p._set_cov_matrix(C)
        logLike_1 = p.logLikelihood(np.array([1.]), 0, 1)
        C = p._get_cov_matrix()
        np.testing.assert_almost_equal(logLike_1, logLike_2)
        np.testing.assert_array_almost_equal(np.tril(Ci), np.tril(C) )

    
    def test_PLagBin_init(self):
        """Test the initialization of PLagBin
        """
        n, mu = 4, 20.
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n) + mu
        m = plag._plag.PLagBin(t, x, x*0+0.1, 1.0, 1, np.array([0.25, 0.5]))
        assert(m.nfq==1)
        assert(m.n==n)


    def test_clagfqL_Cfq_integrals(self):
        """Test the values of the pre-calculated integrals cfq in modfqL
        """
        t = np.arange(4, dtype=np.double)
        t = np.concatenate((t,t))
        fqL = np.array([0.25, 0.3,0.5])
        dt = 1.0
        p = plag._plag.PLagBin(t, t, t, dt, 1, fqL)

        import scipy.integrate as INT
        def Cfun(f, tt):
            return (np.sin(np.pi*f*dt)/(np.pi*f*dt))**2
        
        def Cint_1(tt): 
            return INT.quad(Cfun, fqL[0], fqL[1], (tt,), 
                    weight='cos', wvar=2*np.pi*tt)[0]
        def Cint_2(tt): 
            return INT.quad(Cfun, fqL[1], fqL[2], (tt,), 
                    weight='cos', wvar=2*np.pi*tt)[0]

        T = (t - t[:,None])
        tu = np.unique(T)
        cu1 = [Cint_1(tt) for tt in tu]
        cu2 = [Cint_2(tt) for tt in tu]
        np.testing.assert_array_almost_equal(cu1, p._get_integrals('c', 0))
        np.testing.assert_array_almost_equal(cu2, p._get_integrals('c', 1))


    def test_clagfqL_Sfq_integrals(self):
        """Test the values of the pre-calculated integrals sfq in modfqL
        """
        t = np.arange(4, dtype=np.double)
        t = np.concatenate((t,t))
        fqL = np.array([0.25, 0.3,0.5])
        dt = 1.0
        p = plag._plag.PLagBin(t, t, t, dt, 1, fqL)

        import scipy.integrate as INT
        def Sfun(f, tt):
            return (np.sin(np.pi*f*dt)/(np.pi*f*dt))**2
        
        def Sint_1(tt): 
            return INT.quad(Sfun, fqL[0], fqL[1], (tt,), 
                    weight='sin', wvar=2*np.pi*tt)[0]
        def Sint_2(tt): 
            return INT.quad(Sfun, fqL[1], fqL[2], (tt,), 
                    weight='sin', wvar=2*np.pi*tt)[0]

        T = (t - t[:,None])
        tu = np.unique(T)
        su1 = [Sint_1(tt) for tt in tu]
        su2 = [Sint_2(tt) for tt in tu]
        np.testing.assert_array_almost_equal(su1, p._get_integrals('s', 0))
        np.testing.assert_array_almost_equal(su2, p._get_integrals('s', 1))


    def test_psd_init(self):
        """Test the initialization of psd
        """
        t = np.arange(4, dtype=np.double)
        fqL = np.array([0.25,0.5])
        p = plag._plag.psd(t, t, t, 1.0, fqL, 1, 0)
        assert(p.n == 4)
        assert(p.mu == np.sum(t)/4)
        assert(p.nfq == 1)
        assert(p.npar == 1)

        # do_sig=1 #
        p = plag._plag.psd(t, t, t, 1.0, fqL, 1, 1)
        assert(p.npar == 2)


    def test_psd_logLikelihood(self):
        """Test that logLikelihood of psd runs
        """
        n = 12
        t = np.arange(n, dtype=np.double)
        fqL = np.array([1./12,0.5])
        x = np.random.randn(n) + 4
        p = plag._plag.psd(t, x, x*0+0.1, 1.0, fqL, 1, 0)
        inp = np.array([1.])
        l1 = p.logLikelihood(inp, 1, 0)
        assert(np.isfinite(l1))

        # do_sig=1, and log_factor = 0.0
        p = plag._plag.psd(t, x, x*0+0.1, 1.0, fqL, 1, 1)
        inp = np.array([0., 1.])
        assert(l1 == p.logLikelihood(inp, 1, 0))


    def test_psd_norm(self):
        """Test that logLikelihood of psd runs
        """
        n = 12
        t = np.arange(n, dtype=np.double)
        fqL = np.array([1./12,0.5])
        x = np.random.randn(n) + 4

        p0 = plag._plag.psd(t, x, x*0+0.1, 1.0, fqL, 0, 0)
        p1 = plag._plag.psd(t, x, x*0+0.1, 1.0, fqL, 1, 0)
        p2 = plag._plag.psd(t, x, x*0+0.1, 1.0, fqL, 2, 0)

        inp = np.array([1.])
        mu = x.mean()
        l0 = p0.logLikelihood(inp, 1, 0)
        l1 = p1.logLikelihood(np.log(np.exp(inp)/mu), 1, 0)
        l2 = p2.logLikelihood(np.log(np.exp(inp)/mu**2), 1, 0)
        np.testing.assert_almost_equal(l0, l1, l2)


    def test_psd_gradient_1(self):
        """Test the gradient from dLogLikelihood
        vs scipy.misc.derivative for psd
        """
        np.random.seed(3094)
        n, dt = 4, 1.0
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n)*2 + 4
        xe = x*0+0.01
        fqL = np.array([0.25,0.5])
        inpars = np.array([1.])
        p = plag._plag.psd(t, x, xe, dt, fqL, 0, 0)
        logLike1, g1, h = p.dLogLikelihood(inpars)
        
        from scipy.misc import derivative
        def fun(x, i):
            pp = np.array(inpars)
            pp[i] = x
            return p.logLikelihood(pp, 1, 0)
        g2 = [derivative(fun, inpars[i], 1e-5, 1, (i,)) 
                    for i in range(1)]
        np.testing.assert_almost_equal(g1,g2)


    def test_psd_gradient_2(self):
        """Test the gradient from dLogLikelihood
        vs scipy.misc.derivative for psd with do_sig=1
        """
        np.random.seed(3094)
        n, dt = 4, 1.0
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n)*2 + 4
        xe = x*0+0.01
        fqL = np.array([0.25,0.5])
        inpars = np.array([0., 1.])
        p = plag._plag.psd(t, x, xe, dt, fqL, 0, 1)
        logLike1, g1, h = p.dLogLikelihood(inpars)

        from scipy.misc import derivative
        def fun(x, i):
            pp = np.array(inpars)
            pp[i] = x
            return p.logLikelihood(pp, 1, 0)
        g2 = [derivative(fun, inpars[i], 1e-5, 1, (i,)) 
                    for i in range(2)]
        np.testing.assert_almost_equal(g1,g2)


    def test_psd_gradient_3(self):
        """Test the gradient from dLogLikelihood
        vs scipy.misc.derivative for psd with do_sig=1, and inorm=2
        """
        np.random.seed(3094)
        n, dt = 4, 1.0
        t = np.arange(n, dtype=np.double)
        x = np.random.randn(n)*2 + 4
        xe = x*0+0.01
        fqL = np.array([0.25,0.5])
        inpars = np.array([0., 1.])
        p = plag._plag.psd(t, x, xe, dt, fqL, 2, 1)
        logLike1, g1, h = p.dLogLikelihood(inpars)

        from scipy.misc import derivative
        def fun(x, i):
            pp = np.array(inpars)
            pp[i] = x
            return p.logLikelihood(pp, 1, 0)
        g2 = [derivative(fun, inpars[i], 1e-5, 1, (i,)) 
                    for i in range(2)]
        np.testing.assert_almost_equal(g1,g2)


