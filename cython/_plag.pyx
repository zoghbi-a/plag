
import numpy as np
cimport numpy as np
#np.import_array()
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc.math cimport log, sin, cos, fabs, sqrt, pow, M_PI, exp
from scipy.special.cython_special cimport sici


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



    cdef add_measurement_noise(self, double[:] params, double* arr, int sign):
        """Add measurement noise to the covariance matrix.
            self.Cov will be modified (usually just the diagonal)

        Args:
            params: array in model parameters in case it is 
                needed
            arr: a pointer to a (n,n) whose diagonals are to be modified
            sign: +1,-1 for add or subtract from the diagonal


        """
        cdef:
            int i, n = self.n
            double* sig2 = &self.sig2[0]
        for i in range(n): arr[i*n+i] += sign*sig2[i]



    cpdef double logLikelihood_(self, double[:] params, int icov, int inv):
        """Calculate the log Likelihood for given model params
        
        Args:
            params: parameters of the model
            icov: if not 0, calculate the covariance matrix.
                Otherwise, assume it has already been calculated
                and stored in self.Cov. It is not calculated only
                if called from dLogLikelihood
            inv: if > 0, also calculate the inverse of the
                covariance matrix. It is not needed for the
                logLikelihood, but it is needed when doing the
                derivative. If requested, it will be stored in 
                self.Cov.
                If == 0, just do the likelihood calculation.

        Returns:
            log likelihood value

        """

        ## covariance matrix ##
        cdef double* Cov = &self.Cov[0,0]
        if icov != 0:
            self.Covariance(params)


        ## define some variables ##
        cdef:
            int i, j, n = self.n, info, nrhs = 1
            double[::1] tmp_y = np.array(self.yarr)
            double chi2, logDet = 0
            double* yarr = &self.yarr[0]

        # observation error #
        self.add_measurement_noise(params, Cov, 1)



        ## Cholesky Factor ##
        lapack.dpotrf('L', &n, Cov, &n, &info)
        if info: return -np.inf

        #-- Determinant --#
        for i in range(n): logDet += log(Cov[i+n*i])
        logDet *= 2

        # --chi2 = x^T C x --#
        lapack.dpotrs('L', &n, &nrhs, Cov, &n, &tmp_y[0], &n, &info)
        chi2 = blas.ddot(&n, yarr, &nrhs, &tmp_y[0], &nrhs)


        #-- invert C? --#
        if inv > 0:
            lapack.dpotri('L', &n, Cov, &n, &info)
            for i in range(n):
                for j in range(i):
                    Cov[j+n*i] = Cov[i+n*j]

        #-- loglikelihood --#
        logLike = -0.5 * ( chi2 + logDet + n*log(2*M_PI) )
        return logLike


    cpdef double logLikelihood(self, double[:] params, int icov, int inv):
        """Calculate the log Likelihood for given model params
        
        Args:
            params: parameters of the model
            icov: if not 0, calculate the covariance matrix.
                Otherwise, assume it has already been calculated
                and stored in self.Cov. It is not calculated only
                if called from dLogLikelihood
            inv: if > 0, also calculate the inverse of the
                covariance matrix. It is not needed for the
                logLikelihood, but it is needed when doing the
                derivative. If requested, it will be stored in 
                self.Cov.
                If == 0, just do the likelihood calculation.

        Returns:
            log likelihood value

        """

        ## covariance matrix ##
        cdef double* Cov = &self.Cov[0,0]
        if icov != 0:
            self.Covariance(params)


        ## define some variables ##
        cdef:
            int i, j, n = self.n, info, nrhs = 1, lwork
            double[::1] tmp_y = np.array(self.yarr)
            double chi2, logDet, alpha=1.0
            double* yarr = &self.yarr[0]
            int[:] ipiv = np.empty(n, np.int32)
            double[::1] W

        # observation error #
        self.add_measurement_noise(params, Cov, 1)



        # LDLT decomposition #
        lwork = -1
        lapack.dsytrf('L', &n, Cov, &n, &ipiv[0], &alpha, &lwork, &info)
        lwork = np.int(alpha)
        W = np.empty(lwork, np.double)
        lapack.dsytrf('L', &n, Cov, &n, &ipiv[0], &W[0], &lwork, &info)
        if info: return -np.inf


        #-- Determinant --#
        logDet = 0
        for i in range(n): logDet += log(np.abs(Cov[i+n*i]))
        

        lapack.dsytrs('L', &n, &nrhs, Cov, &n, &ipiv[0], &tmp_y[0], &n, &info)
        if info: return -np.inf
        chi2 = blas.ddot(&n, yarr, &nrhs, &tmp_y[0], &nrhs)


        #-- invert C? --#
        if inv > 0:
            lapack.dsytri('L', &n, Cov, &n, &ipiv[0], &W[0], &info)
            for i in range(n):
                for j in range(i):
                    Cov[j+n*i] = Cov[i+n*j]

        #-- loglikelihood --#
        logLike = -0.5 * ( chi2 + logDet + n*log(2*M_PI) )
        return logLike



    def dLogLikelihood(self, double[:] params):
        """Calculate the logLikelihood gradient and Hessian
            for input parameters params
        
        Args:
            params: parameters of the model
            

        Returns:
            logLikelihood, gradient_array, hessian_matrix


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
        self.add_measurement_noise(params, &self.yyTmC[0,0], -1)
            

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



    def _get_cov_matrix(self, pars=None, add_noise=False):
        """Calculate and return the covariance matrix
            for the parameter array pars

        Args:
            pars: array of parameters. If None, just return
                self.Cov
            add_noise: add measurement noise to the diagonal?

        Returns:
            the Covariance Matrix (without the diagonal sigma)
        """

        if pars is not None:
            self.Covariance(np.array(pars, np.double))
            if add_noise:
                self.add_measurement_noise(np.array(pars, np.double), 
                    &self.Cov[0,0], 1)
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
        int do_sig
    

    def __init__( self, 
            np.ndarray[dtype=np.double_t, ndim=1] t,
            np.ndarray[dtype=np.double_t, ndim=1] y, 
            np.ndarray[dtype=np.double_t, ndim=1] ye, 
            double dt, int npar, double[:] fqL, int do_sig):
        """Base class for models using predefined frequency bins
            It is not meant to be initialized directly.
            It initializes PLagBase and pre-calculate the integrals.

        Args:
            ...: Similar to PLagBase
            fqL: a list or array of the frequency bin boundaries
            do_sig: if == 1, include a parameter that multiplies
                the measurement errors

        """
        
        self.nfq = fqL.shape[0] - 1
        self.fqL = np.array(fqL)
        PLagBase.__init__(self, t, y, ye, dt, npar)
        self.calculate_integrals()
        self.do_sig = do_sig



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

    
    cdef add_measurement_noise(self, double[:] params, double* arr, int sign):
        """see @PLagBase.add_measurement_noise
        """
        cdef:
            int i, n = self.n
            double* sig2 = &self.sig2[0]
            double fac = 1.0
        if self.do_sig: fac = exp(params[0])
        for i in range(n): arr[i*n+i] += sign*fac*sig2[i]


cdef class psd(PLagBin):
    """Class for calculating PSD at pre-defined frequency bins"""
    
    # global parameters #
    cdef:
        double norm


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
        PLagBin.__init__(self, t, y, ye, dt, npar, fqL, do_sig)
        self.norm = self.mu**inorm

    
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
            int iu, k0, nfq = self.nfq
            double res, norm = self.norm, p
        k0 = 1 if self.do_sig else 0

        if self.do_sig and ik==0:
            self.cU[:] = 0.
        else:
            p = exp(params[ik]) * norm
            for iu in range(self.nU):
                self.cU[iu] = p * self.cfq[ik-k0, iu]


    cdef dCovariance(self, double[:] params, int ik):
        """see @PLagBase.dCovariance"""
        cdef:
            int i, n = self.n

        if self.do_sig and ik==0:
            self.dCov[:, :] = 0
            self.add_measurement_noise(params, &self.dCov[0,0], 1)
        else:
            PLagBin.dCovariance(self, params, ik)



cdef class lag(PLagBin):
    """Class for calculating CSD/LAG at pre-defined frequency bins"""
    
    # global parameters #
    cdef:
        double norm
        int n1
        psd pm1, pm2
        double[2] sig_fac
        double[::1,:] C1, C2


    def __init__(self, 
            list T, list Y, list Ye, 
            double dt, double[:] fqL, int inorm, int do_sig,
            double[:] p1, double[:] p2):
        """Calculate cross spectrum and phase lag at predefined 
            frequency bins. The model parameters are the cross 
            spectrum and phase values in the frequency bins 
            (i.e. 2*(len(fqL)-1) of them)

        Args:
            T: a list of two arrays of time axes for the two 
                light curves
            Y: a list of two arrays of rate values corresponding to T
            Ye: 1-sigma measurement error corresponding to Y
            dt: time sampling of the data
            fqL: frequency bin boundaries
            inorm: normalization type. 0:var, 1:leahy, 2:rms
            do_sig: if == 1, include a parameter that multiplies
                the measurement errors
            p1: best fit psd parameters for light curve 1
            p2: best fit psd parameters for light curve 2

        """
        cdef:
            int npar = 2*(fqL.shape[0] - 1)
            np.ndarray t, y, ye
        #if do_sig == 1: npar += 1

        self.pm1 = psd(T[0], Y[0], Ye[0], dt, fqL, inorm, do_sig)
        self.pm2 = psd(T[1], Y[1], Ye[1], dt, fqL, inorm, do_sig)
        self.n1 = self.pm1.n
        self.norm = (self.pm1.mu * self.pm2.mu)**(inorm*0.5)
        self.sig_fac[0] = exp(p1[0]) if do_sig else 1.0
        self.sig_fac[1] = exp(p2[0]) if do_sig else 1.0


        t  = np.concatenate((T[0], T[1]))
        y  = np.concatenate((Y[0]-self.pm1.mu, Y[1]-self.pm2.mu))
        ye = np.concatenate((Ye[0], Ye[1]))
        do_sig = 0
        PLagBin.__init__(self, t, y, ye, dt, npar, fqL, do_sig)



        # the fixed part of the covariance matrix #
        self.pm1.Covariance(p1)
        self.pm2.Covariance(p2)
        self.C1 = np.array(self.pm1.Cov)
        self.C2 = np.array(self.pm2.Cov)


    
    cdef covariance_kernel(self, double[:] params):
        """see @PLagBase.covariance_kernel"""
        cdef:
            int iu, k, nfq = self.nfq
            double res, norm = self.norm

        for iu in range(self.nU):
            res = 0
            for k in range(nfq):
                res += exp(params[k]) * (self.cfq[k, iu]*cos(params[k+nfq]) - 
                                         self.sfq[k, iu]*sin(params[k+nfq]))
            self.cU[iu] = res * norm


    cdef covariance_kernel_deriv(self, double[:] params, int ik):
        """see @PLagBase.covariance_kernel_deriv"""
        cdef:
            int iu, k, nfq = self.nfq
            double res, norm = self.norm, cx, phi


        if (ik)<nfq:
            cx  = exp(params[ik]) * norm
            phi = params[ik+nfq]
            for iu in range(self.nU):
                self.cU[iu] = cx * (
                    self.cfq[ik, iu]*cos(phi) - self.sfq[ik, iu]*sin(phi))
        else:
            k   = ik - nfq
            cx  = exp(params[k]) * norm
            phi = params[ik]
            for iu in range(self.nU):
                self.cU[iu] = cx * (
                    -self.cfq[k, iu]*sin(phi) - self.sfq[k, iu]*cos(phi))


    cdef Covariance(self, double[:] params):
        """see @PLagBase.Covariance"""
        cdef int n1 = self.n1
        PLagBin.Covariance(self, params)
        self.Cov[:n1, :n1] = self.C1[:,:]
        self.Cov[n1:, n1:] = self.C2[:,:]


    cdef dCovariance(self, double[:] params, int ik):
        """see @PLagBase.dCovariance"""
        cdef:
            int n1 = self.n1, n = self.n, i, j

        PLagBin.dCovariance(self, params, ik)
        for i in range(n1):
            for j in range(n1): self.dCov[i, j] = 0
            for j in range(n1, n): self.dCov[i, j] = self.dCov[j, i]
        for i in range(n1,n):
            for j in range(n1,n): self.dCov[i, j] = 0


    cdef add_measurement_noise(self, double[:] params, double* arr, int sign):
        """see @PLagBase.add_measurement_noise
        """
        cdef:
            int i, n = self.n, n1 = self.n1
            double* sig2 = &self.sig2[0]
            double fac1 = self.sig_fac[0], fac2 = self.sig_fac[1]
        for i in range(n1): 
            arr[i*n+i] += sign*fac1*sig2[i]
        for i in range(n1, n): 
            arr[i*n+i] += sign*fac2*sig2[i]


cdef class psdlag(PLagBin):
    """Class for calculating PSD/CSD/LAG at pre-defined frequency bins"""
    
    # global parameters #
    cdef:
        double norm
        int n1
        psd pm1, pm2


    def __init__(self, 
            list T, list Y, list Ye, 
            double dt, double[:] fqL, int inorm, int do_sig):
        """Calculate psd, cross spectrum and phase lag at predefined 
            frequency bins. The model parameters are the psd's for the
            two light curves, cross spectrum and phase values in the 
            frequency bins  (i.e. 4*(len(fqL)-1) of them) plus 2 sigma
            values if do_sig=1

        Args:
            T: a list of two arrays of time axes for the two 
                light curves
            Y: a list of two arrays of rate values corresponding to T
            Ye: 1-sigma measurement error corresponding to Y
            dt: time sampling of the data
            fqL: frequency bin boundaries
            inorm: normalization type. 0:var, 1:leahy, 2:rms
            do_sig: if == 1, include a parameter that multiplies
                the measurement errors, here we include 2 parameters
                one for each light curve

        """
        cdef:
            int npar = 4*(fqL.shape[0] - 1)
            np.ndarray t, y, ye
        if do_sig == 1: npar += 2

        self.pm1 = psd(T[0], Y[0], Ye[0], dt, fqL, inorm, do_sig)
        self.pm2 = psd(T[1], Y[1], Ye[1], dt, fqL, inorm, do_sig)
        self.n1  = self.pm1.n
        self.norm = (self.pm1.mu * self.pm2.mu)**(inorm*0.5)


        t  = np.concatenate((T[0], T[1]))
        y  = np.concatenate((Y[0]-self.pm1.mu, Y[1]-self.pm2.mu))
        ye = np.concatenate((Ye[0], Ye[1]))
        PLagBin.__init__(self, t, y, ye, dt, npar, fqL, do_sig)


    
    cdef covariance_kernel(self, double[:] params):
        """see @PLagBase.covariance_kernel
        The psd part is calculated in Covariance
        """
        cdef:
            int iu, k, k0, nfq = self.nfq
            double res, norm = self.norm
        k0  = 2*nfq + (2 if self.do_sig else 0)

        for iu in range(self.nU):
            res = 0
            for k in range(nfq):
                res += exp(params[k+k0]) * (self.cfq[k, iu]*cos(params[k+k0+nfq]) - 
                                            self.sfq[k, iu]*sin(params[k+k0+nfq]))
            self.cU[iu] = res * norm


    cdef covariance_kernel_deriv(self, double[:] params, int ik):
        """see @PLagBase.covariance_kernel_deriv"""
        cdef:
            int iu, k, k0, nfq = self.nfq
            double res, norm = self.norm, cx, phi

        k0  = 2*nfq + (2 if self.do_sig else 0)

        if ik<k0:
            # we do it in dCovariance
            pass
        else:
            if (ik-k0)<nfq:
                cx  = exp(params[ik]) * norm
                phi = params[ik+nfq]
                for iu in range(self.nU):
                    self.cU[iu] = cx * (
                        self.cfq[ik-k0, iu]*cos(phi) - self.sfq[ik-k0, iu]*sin(phi))
            else:
                k   = ik - nfq
                cx  = exp(params[k]) * norm
                phi = params[ik]
                for iu in range(self.nU):
                    self.cU[iu] = cx * (
                        -self.cfq[k-k0, iu]*sin(phi) - self.sfq[k-k0, iu]*cos(phi))



    cdef Covariance(self, double[:] params):
        """see @PLagBase.Covariance
        params has:
            [p1_f1, p1_f2 ..., p2_f1, p2_f2 ..., cx_f1, cx_f2 ..., l_f1, l_f2 ..]
                if do_lag == 0
            else:
            [p1_s, p1_f1, p1_f2 ..., p2_s, p2_f1, p2_f2 ..., cx_f1, cx_f2 ..., l_f1, l_f2 ..]
        """
        cdef:
            int n1 = self.n1, nfq = self.nfq
            int np1 = nfq + self.do_sig
            double[:] p1 = params[:np1], p2 = params[np1:(2*np1)]
        self.pm1.Covariance(p1)
        self.pm2.Covariance(p2)
        #self.pm1.add_measurement_noise(p1, &self.pm1.Cov[0,0], 1)
        #self.pm2.add_measurement_noise(p2, &self.pm2.Cov[0,0], 1)

        PLagBin.Covariance(self, params)
        self.Cov[:n1, :n1] = self.pm1.Cov[:,:]
        self.Cov[n1:, n1:] = self.pm2.Cov[:,:]


    cdef dCovariance(self, double[:] params, int ik):
        """see @PLagBase.dCovariance"""
        cdef:
            int n1 = self.n1, n = self.n, i, j, nfq = self.nfq
            int np1 = nfq + self.do_sig
            double[:] p1 = params[:np1], p2 = params[np1:(2*np1)]

        if ik < np1:
            self.dCov[:, :] = 0
            self.pm1.dCovariance(p1, ik)
            self.dCov[:n1, :n1] = self.pm1.dCov[:,:]
        elif (ik >= np1) and (ik < (2*np1)):
            self.dCov[:, :] = 0
            self.pm2.dCovariance(p2, ik-np1)
            self.dCov[n1:, n1:] = self.pm2.dCov[:,:]
        else:
            PLagBin.dCovariance(self, params, ik)
            for i in range(n1):
                for j in range(n1): self.dCov[i, j] = 0
                for j in range(n1, n): self.dCov[i, j] = self.dCov[j, i]
            for i in range(n1,n):
                for j in range(n1,n): self.dCov[i, j] = 0


    cdef add_measurement_noise(self, double[:] params, double* arr, int sign):
        """see @PLagBase.add_measurement_noise
        """

        cdef:
            int n1 = self.n1, nfq = self.nfq, i, n = self.n
            int np1 = nfq + self.do_sig
            double* sig2 = &self.sig2[0]
            double fac1 = 1.0, fac2 = 1.0, fac
        if self.do_sig: 
            fac1 = exp(params[0])
            fac2 = exp(params[np1])
        for i in range(n1):
            arr[i*n+i] += sign*fac1*sig2[i]
        for i in range(n1, n):
            arr[i*n+i] += sign*fac2*sig2[i]


cdef double _psdf__pl(double fq, double[:] pars):
    """Power law model with pars = [norm, index]"""
    return exp(pars[0]) * pow(fq,pars[1])

cdef double _psdf__bpl(double fq, double[:] pars):
    """Bending Power law model with pars = [norm, index, bend]
    """
    cdef double a, b, c
    a = exp(pars[0]) # norm
    b = pars[1] # index
    c = exp(pars[2]) #  break
    return (a/fq) * 1./(1 + pow(fq/c,-b-1))

cdef double _psdf__lor(double fq, double[:] pars):
    """Lorentzian model with pars = [norm, fq_center, fq_sigma]
    """
    cdef double a, b, c
    a = exp(pars[0]) # norm
    b = exp(pars[1]) # fq_center
    c = exp(pars[2]) #  fq_sigma
    return a * (c/(2*M_PI)) / ( (fq-b)*(fq-b) + (c/2)*(c/2) )

cdef double _psdf__lor0(double fq, double[:] pars):
    """0-centered Lorentzian model with pars = [norm, fq_sigma]
    """
    cdef double a, c
    a = exp(pars[0]) # norm
    c = exp(pars[1]) #  fq_sigma
    return a * (c/(2*M_PI)) / ( fq*fq + (c/2)*(c/2) )

cdef double _psdf__plor(double fq, double[:] pars):
    """PL + lore: pnorm, pindex, l_norm, l_cent, l_sigm"""
    return _psdf__pl(fq, pars[:2]) + _psdf__lor(fq, pars[2:])

cdef double _psdf__2bpl(double fq, double[:] pars):
    """BPL + BPL: [norm, index, bend]_1 [norm, index, bend]_2"""
    return _psdf__bpl(fq, pars[:3]) + _psdf__bpl(fq, pars[3:])

cdef double _psdf__2lor(double fq, double[:] pars):
    """Lor+Lor: [norm, fq_c, fq_w]_1 [norm, fq_c, fq_w]_2"""
    return _psdf__lor(fq, pars[:3]) + _psdf__lor(fq, pars[3:])


cdef double _Dpsdf__pl(double fq, double[:] pars, int ik):
    """Derivative of _psdf__pl"""
    if ik == 0:
        return _psdf__pl(fq, pars)
    else:
        return log(fq) * _psdf__pl(fq, pars)

cdef double _Dpsdf__bpl(double fq, double[:] pars, int ik):
    """Derivative of _psdf__bpl"""
    cdef double a, b, c, r
    a = exp(pars[0]) # norm
    b = pars[1] # index
    c = exp(pars[2]) #  break
    if ik == 0:
        r = _psdf__bpl(fq, pars)
    elif ik == 1:
        r = a*pow(fq/c, -1-b) * log(fq/c) / (fq*pow(1+pow(fq/c,-1-b), 2))
    else:
        r = ((-1-b)*a/c * pow(fq/c,-2-b)) / pow(1+pow(fq/c,-1-b),2)
    return r

cdef double _Dpsdf__lor(double fq, double[:] pars, int ik):
    """Derivative of _psdf__lor"""
    cdef double a, b, c, r
    a = exp(pars[0]) # norm
    b = exp(pars[1]) # fq_center
    c = exp(pars[2]) #  fq_sigma
    if ik == 0:
        r = _psdf__lor(fq, pars)
    elif ik == 1:
        r = a*b*c*(fq-b)/(M_PI*pow(pow(-b+fq,2)+(c*c/4), 2))
    else:
        r = (-a*c*c*c/(4*M_PI*pow(pow(-b+fq,2)+(c*c/4), 2)) +
            a*c/(2*M_PI*(pow(-b+fq,2)+(c*c/4))) )
    return r

cdef double _Dpsdf__lor0(double fq, double[:] pars, int ik):
    """Derivative of _psdf__lor0"""
    cdef double a, b, c, r=0
    a = exp(pars[0]) # norm
    c = exp(pars[1]) #  fq_sigma
    if ik == 0:
        r = _psdf__lor0(fq, pars)
    elif ik == 1:
        r = (-a*c*c*c/(4*M_PI*pow(pow(fq,2)+(c*c/4), 2)) +
            a*c/(2*M_PI*(pow(fq,2)+(c*c/4))) )
    return r

cdef double _Dpsdf__plor(double fq, double[:] pars, int ik):
    """Derivative of _psdf__plor"""
    if ik < 2:
        return _Dpsdf__pl(fq, pars[:2], ik)
    else:
        return _Dpsdf__lor(fq, pars[2:], ik-2)

cdef double _Dpsdf__2bpl(double fq, double[:] pars, int ik):
    """Derivative of _psdf__2bpl"""
    if ik < 3:
        return _Dpsdf__bpl(fq, pars[:3], ik)
    else:
        return _Dpsdf__bpl(fq, pars[3:], ik-3)

cdef double _Dpsdf__2lor(double fq, double[:] pars, int ik):
    """Derivative of _psdf__2lor"""
    if ik < 3:
        return _Dpsdf__lor(fq, pars[:3], ik)
    else:
        return _Dpsdf__lor(fq, pars[3:], ik-3)

cdef int _identify_function(int ifunc, double (**fmodel) (double, double[:]), 
        double (**Dfmodel) (double, double[:], int)):
    """Given the choice of ifunc, choose the psd or cxd/lag
        function

    Args:
        ifunc: int. identifying the function
        fmodel: pointer to the main function
        Dfmodel: pointer to the derivative function
    Returns:
        number of parameters

    """

    if ifunc == 1:
        fmodel[0]  = &_psdf__pl
        Dfmodel[0] = &_Dpsdf__pl
        return 2
    elif ifunc == 2:
        fmodel[0]  = &_psdf__bpl
        Dfmodel[0] = &_Dpsdf__bpl
        return 3
    elif ifunc == 3:
        fmodel[0]  = &_psdf__lor
        Dfmodel[0] = &_Dpsdf__lor
        return 3
    elif ifunc == 4:
        fmodel[0]  = &_psdf__lor0
        Dfmodel[0] = &_Dpsdf__lor0
        return 2
    elif ifunc == 13:
        fmodel[0]  = &_psdf__plor
        Dfmodel[0] = &_Dpsdf__plor
        return 5
    elif ifunc == 22:
        fmodel[0]  = &_psdf__2bpl
        Dfmodel[0] = &_Dpsdf__2bpl
        return 6
    elif ifunc == 33:
        fmodel[0]  = &_psdf__2lor
        Dfmodel[0] = &_Dpsdf__2lor
        return 6


cdef class psdf(PLagBin):
    """Class for fitting functions directly to the psd"""

    # global parameters #
    cdef:
        double norm
        double (*fmodel) (double, double[:])
        double (*Dfmodel) (double, double[:], int)
        double[:] fq
        int NFQ


    def __init__(self, 
            np.ndarray[dtype=np.double_t, ndim=1] t,
            np.ndarray[dtype=np.double_t, ndim=1] y, 
            np.ndarray[dtype=np.double_t, ndim=1] ye, 
            double dt, double[:] fqL, int inorm, int do_sig, 
            int ifunc, int NFQ):
        """Model psd using pre-defined functions.
            The integration is approximated by NFQ bins.
            fqL here has two elements, taken as the limit
            of the integration

        Args:
            t: array of time axis
            y: array of rate values corresponding to t
            ye: 1-sigma measurement error in r
            dt: time sampling of the data
            fqL: frequency bin boundaries of length 2
            inorm: normalization type. 0:var, 1:leahy, 2:rms
            do_sig: if == 1, include a parameter that multiplies
                the measurement errors
            ifunc: int indicating what functional form to use
            NFQ: how many points to use to approximate the integration

        """
        cdef:
            int npar, i
            double[:] FQL
        FQL = np.logspace(np.log10(fqL[0]), np.log10(fqL[1]), NFQ)

        self.fq = 10**((np.log10(FQL[1:]) + np.log10(FQL[:-1]))/2.)

        npar = _identify_function(ifunc, &self.fmodel, &self.Dfmodel)
        if do_sig == 1: npar += 1

        PLagBin.__init__(self, t, y, ye, dt, npar, FQL, do_sig)
        self.norm = self.mu**inorm
        self.NFQ = NFQ


    cdef covariance_kernel(self, double[:] params):
        """
        params are: fmodel_params
        """
        cdef:
            int iu, k, k0, nfq = self.nfq
            double res, dum, norm = self.norm
            double[:] f
            double* fq = &self.fq[0]
            double (*fmod) (double, double[:])
        fmod = self.fmodel
        k0 = 1 if self.do_sig else 0
        f = np.zeros(nfq, np.double)
        for k in range(nfq):
            f[k] = fmod(fq[k], params[k0:])
        for iu in range(self.nU):
            res = 0
            for k in range(nfq):
                res += f[k] * self.cfq[k, iu]
            self.cU[iu] = res * norm


    cdef covariance_kernel_deriv(self, double[:] params, int ik):
        """Numerical differentiation of fmodel"""
        cdef:
            int iu, k0, k, nfq = self.nfq
            double res, norm = self.norm, h=1e-5
            double[:] p, df
            double* fq = &self.fq[0]
            double (*fmod) (double, double[:])
        fmod = self.fmodel
        k0 = 1 if self.do_sig else 0

        if self.do_sig and ik==0:
            self.cU[:] = 0.
        else:
            p = np.array(params[k0:])
            df = np.zeros(nfq, np.double)
            for k in range(nfq):
                df[k] = self.Dfmodel(fq[k], p, ik-k0)

            for iu in range(self.nU):
                res = 0
                for k in range(nfq):
                    res += df[k] * self.cfq[k, iu]
                self.cU[iu] = res * norm


    cdef dCovariance(self, double[:] params, int ik):
        """see @PLagBase.dCovariance"""
        cdef:
            int i, n = self.n
            double* dCov = &self.dCov[0,0]
            double* sig2 = &self.sig2[0]

        if self.do_sig and ik==0:
            self.dCov[:, :] = 0
            self.add_measurement_noise(params, dCov, 1)
        else:
            PLagBin.dCovariance(self, params, ik)


    def calculate_model(self, params):
        """Calculate model for the given parameters
            at the frequencies of used internally

        Called from python and not from here

        """
        cdef:
            int k0
            double[:] p
        k0 = 1 if self.do_sig else 0
        p = np.array(params[k0:])
        model = np.array([self.fmodel(self.fq[k], p)*self.norm
                    for k in range(self.nfq)])
        return np.asarray(self.fq), model


cdef class lagf(PLagBin):
    """Class for fitting functions directly to the cxd/lag"""

    # global parameters #
    cdef:
        double norm
        double (*cfmodel) (double, double[:])
        double (*cDfmodel) (double, double[:], int)
        double (*lfmodel) (double, double[:])
        double (*lDfmodel) (double, double[:], int)
        double[:] fq
        double[2] sig_fac
        int NFQ
        int n1, npar1
        psdf pm1, pm2
        double[::1,:] C1, C2


    def __init__(self, 
            list T, list Y, list Ye, 
            double dt, double[:] fqL, int inorm, int do_sig, 
            double[:] p1, double[:] p2,
            int[:] ifunc, int NFQ):
        """Model cxd/lag using pre-defined functions.
            The integration is approximated by NFQ bins.
            fqL here has two elements, taken as the limit
            of the integration

        Args:
            T: a list of two arrays of time axes for the two 
                light curves
            Y: a list of two arrays of rate values corresponding to T
            Ye: 1-sigma measurement error corresponding to Y
            dt: time sampling of the data
            fqL: frequency bin boundaries of length 2
            inorm: normalization type. 0:var, 1:leahy, 2:rms
            do_sig: if == 1, include a parameter that multiplies
                the measurement errors
            p1: best fit psd parameters for light curve 1
            p2: best fit psd parameters for light curve 2
            ifunc: 4 int indicating what functional form to use for
                psd1, psd2, cxd and lag
            NFQ: how many points to use to approximate the integration

        """
        cdef:
            int npar, i
            double[:] FQL
            np.ndarray t, y, ye
        FQL = np.logspace(np.log10(fqL[0]), np.log10(fqL[1]), NFQ)
        self.fq = 10**((np.log10(FQL[1:]) + np.log10(FQL[:-1]))/2.)

        # the psd part #
        self.pm1 = psdf(T[0], Y[0], Ye[0], dt, FQL, inorm, do_sig, ifunc[0], NFQ)
        self.pm2 = psdf(T[1], Y[1], Ye[1], dt, FQL, inorm, do_sig, ifunc[1], NFQ)
        self.n1  = self.pm1.n
        self.norm = (self.pm1.mu * self.pm2.mu)**(inorm*0.5)
        self.sig_fac[0] = exp(p1[0]) if do_sig else 1.0
        self.sig_fac[1] = exp(p2[0]) if do_sig else 1.0

        self.npar1 = _identify_function(ifunc[2], &self.cfmodel, &self.cDfmodel)
        npar = self.npar1 + _identify_function(ifunc[3], &self.lfmodel, &self.lDfmodel)
        
        
        t  = np.concatenate((T[0], T[1]))
        y  = np.concatenate((Y[0]-self.pm1.mu, Y[1]-self.pm2.mu))
        ye = np.concatenate((Ye[0], Ye[1]))
        do_sig = 0
        PLagBin.__init__(self, t, y, ye, dt, npar, FQL, do_sig)
        self.NFQ = NFQ
        self.do_sig = do_sig

        # the fixed part of the covariance matrix #
        self.pm1.Covariance(p1)
        self.pm2.Covariance(p2)
        self.C1 = np.array(self.pm1.Cov)
        self.C2 = np.array(self.pm2.Cov)


    cdef covariance_kernel(self, double[:] params):
        """
        params are: cfmodel and clmodel params
        """
        cdef:
            int iu, k, nfq = self.nfq, npar1 = self.npar1
            double res, dum, norm = self.norm
            double[:] cf, lf
            double* fq = &self.fq[0]
            double (*cfmod) (double, double[:])
            double (*lfmod) (double, double[:])
            double[:] cpar = params[:npar1], lpar = params[npar1:]
        cfmod = self.cfmodel
        lfmod = self.lfmodel
        cf = np.zeros(nfq, np.double)
        lf = np.zeros(nfq, np.double)
        for k in range(nfq):
            cf[k] = cfmod(fq[k], cpar)
            lf[k] = lfmod(fq[k], lpar)

        for iu in range(self.nU):
            res = 0
            for k in range(nfq):
                res += cf[k] * (self.cfq[k, iu]*cos(lf[k]) - 
                                self.sfq[k, iu]*sin(lf[k]))
            self.cU[iu] = res * norm

    
    cdef Covariance(self, double[:] params):
        """see @PLagBase.Covariance"""
        cdef int n1 = self.n1
        PLagBin.Covariance(self, params)
        self.Cov[:n1, :n1] = self.C1[:,:]
        self.Cov[n1:, n1:] = self.C2[:,:]
        self.add_measurement_noise(params, &self.Cov[0,0], 1)



    cdef covariance_kernel_deriv(self, double[:] params, int ik):
        """see @PLagBase.covariance_kernel_deriv"""
        cdef:
            int iu, k, nfq = self.nfq, npar1 = self.npar1
            double res, norm = self.norm
            double[:] dfc, dfl, fc, fl
            double* fq = &self.fq[0]
            double (*cfmod) (double, double[:])
            double (*lfmod) (double, double[:])
            double (*cDfmod) (double, double[:], int)
            double (*lDfmod) (double, double[:], int)
            double[:] cpar = params[:npar1], lpar = params[npar1:]
        cfmod  = self.cfmodel
        lfmod  = self.lfmodel
        cDfmod = self.cDfmodel
        lDfmod = self.lDfmodel


        dfc = np.zeros(nfq, np.double)
        dfl = np.zeros(nfq, np.double)
        fc  = np.zeros(nfq, np.double)
        fl  = np.zeros(nfq, np.double)
        for k in range(nfq):
            fc[k]  = cfmod(fq[k], cpar)
            fl[k]  = lfmod(fq[k], lpar)


        if ik<npar1:
            for k in range(nfq):
                dfc[k] = cDfmod(fq[k], cpar, ik)

            for iu in range(self.nU):
                res = 0
                for k in range(nfq):
                    res += dfc[k] * (
                        self.cfq[k, iu]*cos(fl[k]) - self.sfq[k, iu]*sin(fl[k]))
                self.cU[iu] = res * norm
        else:
            for k in range(nfq):
                dfl[k] = lDfmod(fq[k], lpar, ik - npar1)

            for iu in range(self.nU):
                res = 0
                for k in range(nfq):
                    res += fc[k] * dfl[k] * (
                        -self.cfq[k, iu]*sin(fl[k]) - self.sfq[k, iu]*cos(fl[k]))
                self.cU[iu] = norm * res


    cdef dCovariance(self, double[:] params, int ik):
        """see @PLagBase.dCovariance"""
        cdef:
            int i, j, n = self.n, n1 = self.n1
            double* dCov = &self.dCov[0,0]
        PLagBin.dCovariance(self, params, ik)

        # set both psd parts to zero #
        for i in range(n1):
            for j in range(n1): self.dCov[i, j] = 0
            for j in range(n1, n): self.dCov[i, j] = self.dCov[j, i]
        for i in range(n1,n):
            for j in range(n1,n): self.dCov[i, j] = 0


    cdef add_measurement_noise(self, double[:] params, double* arr, int sign):
        """see @PLagBase.add_measurement_noise
        """
        cdef:
            int i, n = self.n, n1 = self.n1
            double* sig2 = &self.sig2[0]
            double fac1 = self.sig_fac[0], fac2 = self.sig_fac[1]
        for i in range(n1): 
            arr[i*n+i] += sign*fac1*sig2[i]
        for i in range(n1, n): 
            arr[i*n+i] += sign*fac2*sig2[i]


    def calculate_model(self, params):
        """Calculate model for the given parameters
            at the frequencies of used internally

        Called from python and not from here

        """
        cdef:
            int npar1 = self.npar1
            double[:] pc = params[:npar1], pl = params[npar1:]
        cmod = np.array([self.cfmodel(self.fq[k], pc)*self.norm
                    for k in range(self.nfq)])
        lmod = np.array([self.lfmodel(self.fq[k], pl)*self.norm
                    for k in range(self.nfq)])
        return np.asarray(self.fq), cmod, lmod

