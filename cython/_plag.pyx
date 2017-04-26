
import numpy as np
cimport numpy as np
np.import_array()
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc.math cimport log, sin, cos, fabs, sqrt, pow, M_PI, exp
cimport dlfcn

## -- sine/cosine integrals   -- ##
## -- sici from scipy.special -- ##
import scipy.special as sp
ctypedef  int(*sici_t)(double, double*, double*)
cdef sici_t sici = <sici_t>dlfcn.load_func(
    sp.__path__[0]+'/_ufuncs.so', 'cephes_sici')
## ----------------------------- ##


cdef class PLagBase:
    """Base Class for calculations. It does the
    likelihood calculations and derivatives"""
    

    # global parameters #
    cdef:
        public int n, npar, nU
        public double dt, mu
        double[::1] yarr, sig2
        double[::1] tU, cU
        long[::1] iU
        double[::1,:] Cov, dCov
        double[::1,:] yyT, yyTmC, CidCCi
        double[::1,:,:] CidC



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

        # for derivatives #
        self.dCov = np.empty((n, n), np.double, 'F')
        self.yyT = np.empty((n, n), np.double, 'F')
        self.yyTmC = np.empty((n, n), np.double, 'F')
        self.CidCCi = np.empty((n, n), np.double, 'F')
        self.CidC = np.empty((n, n, npar), np.double, 'F')

        for i in range(n):
            for j in range(n): self.yyT[i, j] = self.yarr[i]*self.yarr[j]



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


    
    cdef covariance_kernel_deriv(self, double[:] params, int ik):
        """Calculate the covariance kernel derivative with 
            respect to parameter num ik at covariance lags. 
            This function is meant to be inherited by child 
            classes. It takes in the model parameters as input, 
            and calculate the covariance derivative with respect
            to parameter number ik at lags defined self.tU.


        Args:
            params: parameters of the model
            ik: parameter number with respect to which we
                calculate the derivative (0-based) 

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



    cdef dCovariance(self, double[:] params, int ik):
        """Calculate the derivative of the covariance matrix 
            by calling covariance_kernel_deriv and constructing 
            the matrix

        Args:
            params: parameters of the model
            ik: parameter number with respect to which
                we take the derivative

        The result is written to self.dCov

        """
        cdef:
            int i, n=self.n, nU = self.nU
            double* dCov = &self.dCov[0,0]

        self.covariance_kernel_deriv(params, ik)    
        for i in range(n*n):
            dCov[i] = self.cU[self.iU[i]]



    cdef add_measurement_noise(self, double[:] params):
        """Add measurement noise to the covariance matrix.
            self.Cov will be modified (usually just the diagonal)

        Args:
            params: array in model parameters in case it is 
                needed


        """
        cdef:
            int i, n = self.n
            double* Cov = &self.Cov[0,0]
            double* sig2 = &self.sig2[0]


        for i in range(n): Cov[i*n+i] += sig2[i]



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



    def dLogLikelihood(self, double[:] params):
        """Calculate the logLikelihood gradient and Hessian
            for input parameters params
        
        Args:
            params: parameters of the model
            

        Returns:
            logLikelihood, gradient_arrat, hessian_matrix


        """

        cdef:
            int i, j, k, l, n = self.n, npar = self.npar
            double logLike
            double alpha = 1, beta = 0, g, h
            double[:]   grad = np.empty(npar, np.double)
            double[:,:] hess = np.empty((npar,npar), np.double)

        # get covariance matrix and yyTmC before #
        # Cov is messed up in the factorization  #
        self.Covariance(params)
        for i in range(n):
            for j in range(n): 
                self.yyTmC[i, j] = self.yyT[i, j] - self.Cov[i, j]
            self.yyTmC[i, i] -= self.sig2[i]

        # logLikelihood #
        logLike = self.logLikelihood(params, 0, 1)

        ## -- calculate grad and hess following Bond+1998 -- ##
        for k in range(npar):
            self.dCovariance(params, k)
            blas.dsymm('L', 'L', &n, &n, &alpha, &self.Cov[0,0],
                &n, &self.dCov[0,0], &n, &beta, &self.CidC[0,0,k], &n)
            blas.dsymm('R', 'L', &n, &n, &alpha, &self.Cov[0,0] ,
                &n, &self.CidC[0,0,k], &n, &beta, &self.CidCCi[0,0], &n)
            g = 0
            for i in range(n):
                g += self.yyTmC[i,i]*self.CidCCi[i,i]
                for j in range(i):
                    g += 2*self.yyTmC[i,j]*self.CidCCi[j,i]
            grad[k] = 0.5*g

        for k in range(npar):
            for l in range(k+1):
                h = 0
                for i in range(n):
                    for j in range(n):
                        h += self.CidC[i,j,k] * self.CidC[j,i,l]
                hess[k,l] = 0.5*h
                hess[l,k] = 0.5*h
        return logLike, np.asarray(grad), np.asarray(hess)



    def _get_cov_arrays(self):
        """Returns the covariance-related arrays.
            To be called by python for testing

        Returns:
            tU, cU, iU
        """
        return (np.asarray(self.tU), np.asarray(self.cU),
                    np.asarray(self.iU))



    def _get_cov_matrix(self, pars=None):
        """Calculate and return the covariance matrix
            for the parameter array pars

        Args:
            pars: array of parameters. If None, just return
                self.Cov

        Returns:
            the Covariance Matrix (without the diagonal sigma)
        """

        if pars is not None:
            self.Covariance(np.array(pars, np.double))
        return np.asarray(self.Cov)



    def _set_cov_matrix(self, C):
        """Set the covariance matrix by hand.
            Used for testing only.

        Args:
            C: Matrix of size (n, n)
        """
        self.Cov = np.array(C)




cdef class PLagBin(PLagBase):
    """psd/lag base class for models using predefined
        frequency bins
    """

    # global parameters #
    cdef:
        public int nfq
        double[:] fqL
        double[:,:] cfq, sfq
    

    def __init__( self, 
            np.ndarray[dtype=np.double_t, ndim=1] t,
            np.ndarray[dtype=np.double_t, ndim=1] y, 
            np.ndarray[dtype=np.double_t, ndim=1] ye, 
            double dt, int npar, double[:] fqL):
        """Base class for models using predefined frequency bins
            It is not meant to be initialized directly.
            It initializes PLagBase and pre-calculate the integrals.

        Args:
            ...: Similar to PLagBase
            fqL: a list or array of the frequency bin boundaries"""
        
        self.nfq = fqL.shape[0] - 1
        self.fqL = np.array(fqL)
        PLagBase.__init__(self, t, y, ye, dt, npar)
        self.calculate_integrals()



    cdef calculate_integrals(self):
        """Precalculate model integrals resulting from
            the Fourier transform. For predefined frequency
            bins, these integrals do not depend on the parameters
            so we calculate them only once.
            The results are written to two arrays: self.cfq and 
            self.sfq (of dims (nfq, nU) )

            cfq:
                Integrate[Cos[2 pi f  t]*Sinc[pi f dt]^2, f]
            sfq:
                Integrate[Sin[2 pi f  t]*Sinc[pi f dt]^2, f]

            The cases of t=0 and t=-+dt are special cases and they
            are handled separately

            t in in this case is the covariance time lags in self.tU

        """
        
        cdef:
            int i, k, nU = self.nU, nfq = self.nfq, sgn
            double tt, dt=self.dt, norm, pi=M_PI
            double[:] cf, sf, fqL = np.array(self.fqL)
            double s1, c1, s2, c2, s3, c3, dum1, dum2, dum3


        self.cfq  = np.empty((nfq, nU), np.double)
        self.sfq  = np.empty((nfq, nU), np.double)


        cf   = np.empty(nfq+1, np.double)
        sf   = np.empty(nfq+1, np.double)
        norm = 1./(pi*dt)**2

        for i in range(self.nU):
            tt  = self.tU[i]
            sgn = 1 if tt>0 else -1 
            if np.isclose(tt, 0):
                # -- tt=0 -- #
                for k in range(nfq+1):
                    dum1   = 2*pi*fqL[k]*dt
                    sici(dum1, &s1, &c1)
                    cf[k] = pi*dt*s1 + (cos(dum1)-1)/(2*fqL[k])
                    if k>0:
                        self.cfq[k-1, i] = norm * (cf[k] - cf[k-1])
                        self.sfq[k-1, i] = 0
            elif np.isclose(fabs(tt), dt):
                # -- abs(tt) = dt -- #
                for k in range(nfq+1):
                    dum1 = 2*pi*fqL[k]*dt
                    dum2 = 2*dum1
                    sici(dum1, &s1, &c1)
                    sici(dum2, &s2, &c2)
                    cf[k] = pi*dt*(-s1+s2) + (
                                1-2*cos(dum1)+cos(dum2))/(4*fqL[k])
                    sf[k] = pi*dt*( c1-c2) + (
                                 -2*sin(dum1)+sin(dum2))/(4*fqL[k])
                    if k>0:
                        self.cfq[k-1, i] = norm * (cf[k] - cf[k-1])
                        self.sfq[k-1, i] = norm * (sf[k] - sf[k-1]) * sgn
            else:
                # -- general -- #
                for k in range(nfq+1):
                    dum1 = 2*pi*fqL[k]*(dt-tt)
                    dum2 = 2*pi*fqL[k]*(tt)
                    dum3 = 2*pi*fqL[k]*(dt+tt)
                    sici(dum1, &s1, &c1)
                    sici(dum2, &s2, &c2)
                    sici(dum3, &s3, &c3)
                    cf[k] = (dum1*s1-2*dum2*s2+dum3*s3 + 
                                cos(dum1)-2*cos(dum2)+cos(dum3))/(4*fqL[k])
                    sf[k] = (dum1*c1+2*dum2*c2-dum3*c3 - 
                                sin(dum1)-2*sin(dum2)+sin(dum3))/(4*fqL[k])
                    if k>0:
                        self.cfq[k-1, i] = norm * (cf[k] - cf[k-1])
                        self.sfq[k-1, i] = norm * (sf[k] - sf[k-1])


    def _get_integrals(self, str stype, int idx):
        """Make the integral arrays accessible from
            python, so we can test they are correct.
            This is used for testing only

        """
        return np.asarray(self.sfq[idx, :] 
                if stype=='s' else self.cfq[idx, :])



cdef class psd(PLagBin):
    """Class for calculating PSD at pre-defined frequency bins"""
    
    # global parameters #
    cdef:
        double norm
        int do_sig


    def __init__(self, 
            np.ndarray[dtype=np.double_t, ndim=1] t,
            np.ndarray[dtype=np.double_t, ndim=1] y, 
            np.ndarray[dtype=np.double_t, ndim=1] ye, 
            double dt, double[:] fqL, int inorm, int do_sig):
        """Calculate psd at predefined frequency bins.
            if do_sig==1, then the model parameters are the 
            psd values (in log units) in the frequency bins 
            (i.e. len(fqL)-1 of them). If do_sig == 0,
            then the first parameter is the sig2_factor
            and the rest are the psd values.

        Args:
            t: array of time axis
            y: array of rate values corresponding to t
            ye: 1-sigma measurement error in r
            dt: time sampling of the data
            fqL: frequency bin boundaries
            inorm: normalization type. 0:var, 1:leahy, 2:rms
            do_sig: if == 1, include a parameter that multiplies
                the measurement errors

        """
        cdef int npar = fqL.shape[0] - 1
        if do_sig == 1: npar += 1
        PLagBin.__init__(self, t, y, ye, dt, npar, fqL)
        self.norm = self.mu**inorm
        self.do_sig = do_sig

    
    cdef covariance_kernel(self, double[:] params):
        """Calculate the covariance kernel at covariance lags.
            It takes in the model parameters
            as input, and calculates the covariance at lags
            defined in self.tU.


        Args:
            params: parameters of the model

        The results are written to self.cU

        """
        cdef:
            int iu, k, k0, nfq = self.nfq
            double res, norm = self.norm
        k0 = 1 if self.do_sig else 0

        for iu in range(self.nU):
            res = 0
            for k in range(nfq):
                res += exp(params[k+k0]) * self.cfq[k, iu] * norm
            self.cU[iu] = res


    cdef add_measurement_noise(self, double[:] params):
        """see @PLagBase.add_measurement_noise
        """
        cdef:
            int i, n = self.n
            double* Cov = &self.Cov[0,0]
            double* sig2 = &self.sig2[0]
            double fac = 1.0
        if self.do_sig: fac = exp(params[0])
        for i in range(n): Cov[i*n+i] += fac*sig2[i]


    cdef covariance_kernel_deriv(self, double[:] params, int ik):
        """Calculate the covariance kernel derivative with 
            respect to parameter num ik at covariance lags. 
            It takes in the model parameters as input, 
            and calculate the covariance derivative with respect
            to parameter number ik at lags defined self.tU.


        Args:
            params: parameters of the model
            ik: parameter number with respect to which we
                calculate the derivative (0-based) 

        """
        cdef:
            int iu, k, k0, nfq = self.nfq
            double res, norm = self.norm
        k0 = 1 if self.do_sig else 0

        if self.do_sig and ik==0:
            self.cU[:] = 0.
        else:
            for iu in range(self.nU):
                self.cU[iu] = exp(params[ik]) * self.cfq[ik-k0, iu] * norm


    cdef dCovariance(self, double[:] params, int ik):
        """see @PLagBase.dCovariance"""
        cdef:
            int i, n = self.n
            double* dCov = &self.dCov[0,0]
            double* sig2 = &self.sig2[0]

        if self.do_sig and ik==0:
            self.dCov[:, :] = 0
            for i in range(n): dCov[i*n+i] = exp(params[0])*sig2[i]
        else:
            PLagBin.dCovariance(self, params, ik)

