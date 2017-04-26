
import numpy as np
cimport numpy as np


cdef class PLagBase:
    """Base Class for calculations. It does the
    likelihood calculations and derivatives"""
    

    # global parameters #
    cdef:
        public int n, npar, nU
        public double dt, mu
        double[::1] yarr, sig2
        double[::1,:] Cov
        double[::1] tU, cU
        long[::1] iU



    def __init__(self,
            np.ndarray[dtype=np.double_t, ndim=1] t,
            np.ndarray[dtype=np.double_t, ndim=1] y, 
            np.ndarray[dtype=np.double_t, ndim=1] ye, 
            double dt, int npar):
        """Initialize the base class. It takes the data arrays,
            time sampling dt and the number of parameters in the
            model.

        Args:
            t: np.ndarray containing the time axis
            y: np.ndarray containing the rate axis
            ye: np.ndarray containing the measurement uncertainties.
            dt: time sampling
            npar: number of the parameters that the model takes

        """

        # initialize global variables #
        self.n = t.shape[0]
        self.dt = dt
        self.npar = npar
        self.mu = y.mean()
        self.yarr = y - self.mu
        self.sig2 = ye * ye


        # unique time lags for covariance  #
        # so calculations are not repeated #
        # tU: unique values of covariance lag #
        # iU: index matrix to reconstruct cov #
        cdef:
            int n = self.n, i, j
            double[::1, :] tt = (t[None, :] - t[:, None]).T

        self.tU, self.iU = np.unique(tt, return_inverse=True)
        self.cU = np.array(self.tU)
        self.nU = self.tU.shape[0]
        self.Cov = np.empty((n, n), np.double, 'F')



    cdef covariance_kernel(self, double[:] params):
        """Calculate the covariance kernel at covariance lags tU.
            This function is meant to be inherited by
            child classes. It takes in the model parameters
            as input, and calculates the covariance at lags
            defined self.tU.


        Args:
            params: parameters of the model

        The results are written to self.cU
        
        """
        self.cU[:] = 0



    cdef Covariance(self, double[:] params):
        """Calculate the covariance matrix by calling
            covariance_kernel and constructing the matrix

        Args:
            params: parameters of the model

        The result is written to self.Cov

        """
        cdef:
            int i, n=self.n, nU = self.nU
            double* Cov = &self.Cov[0,0]

        self.covariance_kernel(params)  
        for i in range(n*n):
            Cov[i] = self.cU[self.iU[i]]



    def _get_cov_arrays(self):
        """Returns the covariance-related arrays.
            To be called by python for testing

        Returns:
            tU, cU, iU
        """
        return (np.asarray(self.tU), np.asarray(self.cU),
                    np.asarray(self.iU))



    def _get_cov_matrix(self, pars):
        """Calculate and return the covariance matrix
            for the parameter array pars

        Args:
            pars: array of parameters

        Returns:
            the Covariance Matrix (without the diagonal sigma)
        """

        self.Covariance(np.array(pars, np.double))
        return np.asarray(self.Cov)

