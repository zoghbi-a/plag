
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc.math cimport log, sin, cos, fabs, sqrt, pow, M_PI, exp


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



    cdef add_measurement_noise(self, double[:] params):
        """Add measurement noise to the covariance matrix.
            self.Cov will be modified (usually just the diagonal)

        Args:
            params: array in model parameters in case it is 
                needed


        """
        cdef int i, n = self.n
        cdef double* Cov = &self.Cov[0,0]

        for i in range(n): Cov[i*n+i] += self.ye2[i]



    cpdef double logLikelihood(self, double[:] params, int icov, int inv):
        """Calculate the log Likelihood for given model params
        
        Args:
            params: parameters of the model
            icov: if not 0, calculate the covariance matrix.
                Otherwise, assume it has already been calculated
                and stored in self.Cov. It is calculated only
                if called from DlnLikelihood
            inv: if > 0, also calculate the inverse of the
                covariance matrix. It is not needed for the
                logLikelihood, but it is needed when doing the
                derivative. If requested, it will be stored in 
                self.Cov.
                If == 0, just do the likelihood calculation.
                If < 0, store the residuals in the diagonal elements
                    of self.Cov

        Returns:
            log likelihood value

        """

        ## covariance matrix ##
        cdef double* Cov = &self.Cov[0,0]
        if icov != 0:
            self.Covariance(params)


        ## define some variables ##
        cdef:
            int i, n = self.n, info, nrhs = 1
            double[::1] tmp_x, tmp_y = np.array(self.yarr)
            double chi2, logDet = 0
            double* yarr = &self.yarr[0]

        # observation error #
        self.add_measurement_noise(params)



        ## Cholesky Factor ##
        lapack.dpotrf('L', &n, Cov, &n, &info)
        if info: return -np.inf

        #-- Determinant --#
        for i in range(n): logDet += log(Cov[i+n*i])
        logDet *= 2
        if inv < 0:
            tmp_x = np.zeros(n, np.double)
            for i in range(n): tmp_x[i] = Cov[i+n*i]**2

        # --chi2 = x^T C x --#
        lapack.dpotrs('L', &n, &nrhs, Cov, &n, &tmp_y[0], &n, &info)
        chi2 = blas.ddot(&n, yarr, &nrhs, &tmp_y[0], &nrhs)
        if inv < 0:
            for i in range(n):
                Cov[i+n*i] = tmp_y[i] * yarr[i] * tmp_x[i]

        #-- invert C? --#
        if inv > 0:
            lapack.dpotri('L', &n, Cov, &n, &info)

        #-- loglikelihood --#
        logLike = -0.5 * ( chi2 + logDet + n*log(2*M_PI) )
        return logLike



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




