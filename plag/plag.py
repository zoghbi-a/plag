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
        p = par + dpar
        p = np.clip(p, -20, 20)
        return p
