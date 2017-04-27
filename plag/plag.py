# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import _plag
import numpy as np


class PLag(object):
    """ plag.PLagCython container for 
        fitting multiple light curves
        This is the class that a user would initialize and call
    """


    def __init__(self, plag_type, T, Y, Ye, dt, *args):
        """A model container that fits multiple light curves
            at the same time. This is the class that a user
            would typically initialize and call.

        Args:
            plag_type: string of the name of the model implemented
                in _plag defined here as a subclass of PLagCython
                e.g psd, cxd or pcxd.

            T: a list of np.ndarray of time axis of size num_of_light_curves.
                If doing lags, this should be a list of lists of np.ndarrays,
                where the length of outer list is the number of light curves,
                while the length of the inner list is 2 for the reference light curve
                and the light curve of interest. 
            Y: similar to T but containing the rate arrays instead of time. 
            Ye: similar to T but containing the measurement errors instead of time.
            dt: time sampling
            *args: extra arguments to be bassed to plag.{plag_type}.
                see @plag.{plag_type} for details.

        """

        # number of models is the number of light curves #
        self.nmod = len(T)
        model = eval(plag_type)
        self.mods = [model(t, y, ye, dt, *args) for t,y,ye in zip(T, Y, Ye)]
        

    def logLikelihood(self, params):
        """Calculate the logLikelihood for the given 
            input parameters. When multiple light curves
            are used, the total LogLikelihood is the sum 
            of the individual logLikelihood values. each
            calculated using self.mod.logLikelihood
    
        Args:
            params: np.ndarray of the input model parameters

        Returns:
            Log of the Likelihood function.

        """
        l = np.sum([m.logLikelihood(params) for m in self.mods])
        return l


    def dLogLikelihood(self, params):
        """Calculates the derivative and hessian of the 
            likelihood function at the location of the 
            input parameters. When multiple light curves
            are used, the total LogLikelihood, total gradient
            and total Hessian are the sum of those from individual
            light curves. Each is calculated using 
            self.mod.dLogLikelihood
        
        Args:
            params: np.ndarray of the the input model parameters

        Returns:
            logLikelihood, gradient_array, Hessian_matrix

        """
        npar= len(params) 
        L, G, H = 0, np.zeros(npar), np.zeros((npar,npar))
        for i in range(self.nmod):
            l, g, h = self.mods[i].dLogLikelihood(params)
            if not np.isfinite(l): return l, g, h
            L += l
            G += g
            H += h
        return L, G, H


    def step_param(self, par, dpar):
        """A stepping criterion during the optimization. 
            During the search for best fitting parameters, the
            gradient and Hessian are calculated using 
            @dLogLikelihood. This gives a step in the search
            with direction defined by dpar. The simplest stepping
            function would be to move in that direction. i.e
            par = par + dpar. Sometime other constraints in the
            parameters are needed. They should be included here.
            This function religates the stepping to the individual
            models. Each would have some resonable stepping function


        Args:
            par: model parameters at the current position
            dpar: a suggested stepping vector.

        Returns:
            par: the position (i.e. values of parameters) for
                the next step. The simples is par + dpar

        """
        return self.mods[0].step_param(par, dpar)



class PLagCython(object):
    """A wrapper for the cython PLagBase.
        other classes that inherit from this don't have to
        redefine the same tasks.
    """

    def __init__(self, plag_type, t, y, ye, dt, *args):
        """A wrapper for the cython class _plag.PLagBase.
        All models implemented in cython need to have a python
        wrapper that inherit PLagCython

        Args:
            plag_type: string of the name of the model implemented
                in _plag. e.g. psd etc. The cython model is then
                accessed via _plag.{plag_type}
            t: np.ndarray of time axis or a list of size 2 for the case 
                    of cross-spec/lag
            y: np.ndarray of corresponding rates or a list of size 2 
                for the case of cross-spec/lag
            ye: np.ndarray of the 1-sigma measurement errors. or a list
                of size 2 for the case of cross-spec/lag
            dt: time sampling
            *args: extra arguments to be bassed to _plag.{plag_type}.
                see @_plag.{plag_type} for details.

        """
        # mod stores the underlying cython object #
        self.mod  = eval('_plag.{}(t, y, ye, dt, *args)'.format(plag_type))
        self.npar = self.mod.npar


    def Covariance(self, params):
        """Calculate the covariance matrix given the model
            parameters. This calls the cython object 
            self.mod.Covariance(params)

        """
        self.mod.Covariance(np.array(params, np.double))


    def _get_cov_matrix(self, pars=None):
        """Returns the covariance matrix stored in self.mod.Cov

        Args:
            pars: input parameters. If not given, return
                the covariance matrix in memory
        """
        if pars is not None:
            pars = np.array(pars, np.double)
        return self.mod._get_cov_matrix(pars)


    def logLikelihood(self, params, icov=1, inv=0):
        """Calculate the logLikelihood for the given 
            input parameters. The calculation is done 
            by self.mod.logLikelihood in the cython
            object
    
        Args:
            params: parameters of the model
            icov: if not 0, calculate the covariance matrix.
                Otherwise, assume it has already been calculated
                and stored in mod.Cov. It is not calculated only
                if called from dLogLikelihood
            inv: if > 0, also calculate the inverse of the
                covariance matrix. It is not needed for the
                logLikelihood, but it is needed when doing the
                derivative. If requested, it will be stored in 
                self.Cov.
                If == 0, just do the likelihood calculation.
                If < 0, store the residuals in the diagonal elements
                    of self.Cov

        Returns:
            Log of the Likelihood function.
        

        """
        return self.mod.logLikelihood(np.array(params, np.double), icov, inv)


    def dLogLikelihood(self, params):
        """Calculates the derivative of the likelihood function
            at the location of the input parameters. The actual
            calculation is done by self.mod.dLogLikelihood in
            the cython code.
        
        Args:
            params: np.ndarray of the the input model parameters

        Returns:
            logLikelihood, gradient_array, Hessian_matrix

        """
        return self.mod.dLogLikelihood(np.array(params, np.double))


    def step_param(self, par, dpar):
        """A stepping criterion during the optimization. 
            During the search for best fitting parameters, the
            gradient and Hessian are calculated using 
            @dLogLikelihood. This gives a step in the search
            with direction defined by dpar. The simplest stepping
            function would be to move in that direction. i.e
            par = par + dpar. Sometime other constraints in the
            parameters are needed. They should be included here.


        Args:
            par: model parameters at the current position
            dpar: a suggested stepping vector.

        Returns:
            par: the position (i.e. values of parameters) for
                the next step. The simples is par + dpar

        """
        return par + dpar



class psd(PLagCython):
    """ PSD at predefined frequencies
    """

    def __init__(self, t, y, ye, dt, fqL, norm='rms', fit_sigma=False):
        """A psd model to calculated psd at pre-defined
        frequencies. The normalization is defined by norm.
        Optionally, a sigma factor can be included as a fit parameter

        Args:
            t: np.ndarray of time axis
            y: np.ndarray of corresponding rates
            ye: np.ndarray of the 1-sigma measurement errors.
            dt: time sampling
            fqL: a list of array of frequency bin boundaries.
            norm: var|leahy|rms
            fit_sigma: include a sigma factor an additional free parameter

        """
        inorm = 2
        if norm == 'var': inorm = 0
        if norm == 'leahy': inorm = 1
        do_sig = 1 if fit_sigma else 0
        super(self.__class__, self).__init__(
            'psd', t, y, ye, dt, np.array(fqL, np.double), inorm, do_sig)


    def step_param(self, par, dpar):
        __doc__ = super(self.__class__, self).step_param.__doc__
        dpar = np.clip(dpar, -2, 2)
        p = par + dpar
        p = np.clip(p, -20, 20)
        return p

class psdf(PLagCython):
    """ PSD Using some pre-defined functions
    """

    def __init__(self, t, y, ye, dt, fqL, norm='rms', fit_sigma=False, ifunc=1, NFQ=50):
        """Model the psd with with a function form.
        The normalization is defined by norm.
        Optionally, a sigma factor can be included as a fit parameter

        Args:
            t: np.ndarray of time axis
            y: np.ndarray of corresponding rates
            ye: np.ndarray of the 1-sigma measurement errors.
            dt: time sampling
            fqL: a list of array of frequency bin boundaries.
            norm: var|leahy|rms
            fit_sigma: include a sigma factor an additional free parameter
            ifunc: a number indicating what function to use.
                1: powerlaw
                2: bending powerlaw
                3: lorentzian with 3 parameters
                13: PL + lorentzian
                22: bending PL + bending PL
                33: lorentzian + lorentzian
            NFQ: how many frequency bins to use internally to calculate the 
                integrals

        """
        inorm = 2
        if norm == 'var': inorm = 0
        if norm == 'leahy': inorm = 1
        do_sig = 1 if fit_sigma else 0
        super(self.__class__, self).__init__(
            'psdf', t, y, ye, dt, np.array(fqL, np.double), 
            inorm, do_sig, ifunc, NFQ)


    def step_param(self, par, dpar):
        __doc__ = super(self.__class__, self).step_param.__doc__
        dpar = np.clip(dpar, -2, 2)
        p = par + dpar
        p = np.clip(p, -20, 20)
        return p


def optimize(mod, p0, ip_fix=None, maxiter=500, tol=1e-4, verbose=1):
    """Simple optimization routine that uses the
        Quadaratic approximation. From the calculated gradient 
        and Hessian, a stepping vector is
        created and used to nagivate the parameters space.
        It seems to work better that the scipy.optimize.minimize,
        but this may depend on the problem at hand.

        WARNING :: This is my simple implementation and it is error-prone
            check the results, and use at you own risk.

    Args:
        mod: the model whose logLikelihood function is to be maximized.
            It is assumed of course that there is a function
            mod.dLogLikelihood to calculate the logLikelihood values,
            gradient and Hessian for a given set of prameters.
        p0: a list or array of the initial values of the model parameters.
            To ensure convergence, this need to be close enough to the real
            solution, particularly for the cross-spec/lag models.
        ip_fix: indices of parameters to keep fixed duing the optimization.
                This is 0-based.
        maxiter: maximum number of search iterations before it is considered
            a failor. The convergence usally happens in less than a few tens
            of iterations unless the model is very complicated and the data
            quality is not good.
        tol: what is the tolerance for convergence. We check three parameters
            if ANY of them has an absolote value < tol, we consider this a 
            convergence. The parameters are:
                -absmax: max of abs(dpar/par)
                -gmax: max of abs(gradient)
                -dfun: change in logLikelihood between iterations.
        verbose: print verbosity.


    Returns:
        p, pe, l: the best fit parameter values and their ESTIMATED errors
            from the Hessian matrix, and the value of the loglikelihood.
            The errors are NOT to be taken seriously. They are usually 
            underestimated. Run @error for a more correct error estimates
            or mcmc.

    """
    tmpp  = np.array(p0, np.double)
    dpar  = tmpp*1e-3
    prev  = [-np.inf, tmpp]
    gzero = 1e-5
    npar  = len(p0)
    nloop = 0

    hinv   = np.zeros((npar,npar)) + 1e16

    for niter in range(1, maxiter+1):
        l, g, h = mod.dLogLikelihood(tmpp)
        
        if not np.isfinite(l):
            dpar   /= 10.
            l, tmpp = prev
            tmpp    = mod.step_param(tmpp, dpar)
            continue

        ## -- handle constants -- ##
        ivar   = np.argwhere(np.abs(g)>gzero)[:,0]
        if len(ivar) == 0: ivar = np.arange(npar)
        if ip_fix is not None:
            if not isinstance(ip_fix, list): ip_fix = [ip_fix]
            ivar = [i for i in ivar if i not in ip_fix]
        jvar   = [[i] for i in ivar]
        hinv_  = np.linalg.inv(h[ivar, jvar])
        dpar_  = np.dot(hinv_, g[ivar])
        dpar   = np.zeros(npar)
        dpar[ivar] = dpar_
        hinv   = np.zeros((npar,npar)) + 1e16
        hinv[ivar, jvar] = hinv_
        ## ---------------------- ##


        absmax = np.max(np.abs(dpar/tmpp))
        gmax   = np.max(np.abs(g))
        dfun   = l-prev[0]

        if verbose:
            print('{:4} {:5.3e} {:5.3e} {:5.3e} | {:5.3e} | {}'.format(
                niter, absmax, gmax, dfun, l, 
                ' '.join(['{:5.3g}'.format(x) for x in tmpp])))
        
        if absmax<tol or gmax<tol or np.abs(dfun)<tol: break
        prev   = [l, tmpp]
        
        # if we are stuck in a loop, disturb the solution a bit #
        if dfun<0: nloop += 1
        if nloop>10: dpar /= 2.

        tmpp   = mod.step_param(tmpp, dpar)
    p, pe = tmpp, np.sqrt(np.diagonal(hinv))

    ## used when calculating errors ##
    if ip_fix is not None: return p, pe, l
    ## ---------------------------- ##
    if verbose:
        print('*'*20)
        print(' '.join(['{:4g}'.format(x) for x in p]))
        print(' '.join(['{:4g}'.format(x) for x in pe]))
        print(' '.join(['{:4g}'.format(x) for x in g]))
        print('*'*20)
    return p, pe, l


