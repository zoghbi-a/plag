#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import az
import scipy.optimize as opt

import matplotlib.pylab as plt

from IPython import embed


import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import plag


sim_input = {
    'n'     : 4096,
    'dt'    : 1.0,
    'mean'  : 100.,
    'norm'  : 'var',
    'psdpar': ['powerlaw', [1., -2]],
    'seglen': 128,
    'infqL' : 6,
    'gnoise': 0.1,
    'fname' : 'sim.lc',
}


def _simulate_lc(seed=34789, return_sim=False):
    """Simulate light curves and split them
        to segments

    Args:
        seed: random seed
        return_sim: return SimLC object

    Returns:
        t, r, e: where each is a list of arrays

    """

    # input #
    for k in ['n', 'dt', 'mean', 'psdpar', 'norm', 'gnoise', 'seglen']:
        exec('{0} = sim_input["{0}"]'.format(k))


    sim = az.SimLc(seed=seed)
    sim.add_model(*psdpar)
    sim.simulate(16*n, dt/4, mean, norm)
    if return_sim: return sim
    r = sim.x[4*n:8*n].reshape((n, 4)).mean(1)
    t = sim.t[4*n:8*n].reshape((n, 4)).mean(1)
    if gnoise is None:
        # poisson noise #
        raise NotImplemented
    else:
        # gaussian noise #
        r += np.random.randn(n)*gnoise
        e = np.ones(n)*gnoise
    

    # split to segments #
    r, idx = az.misc.split_array(r, seglen, index=True)
    t = [t[i] for i in idx]
    e = [e[i] for i in idx] 

    return t, r, e


def _get_fqL(tseg):
    """What are the frequency bin boundaries
    
    Args:
        tseg: a list of time axis segments.

    Returns:
        fqL, fq: frequency bins boundaries, and frequency
            values at the geometric center.
    """

    # input #
    for k in ['dt', 'infqL']:
        exec('{0} = sim_input["{0}"]'.format(k))


    # limits from the data #
    tranges = np.array([[t[-1]-t[0]] for t in tseg])
    f1, f2, f3 = 1./(tranges.max()), 1./(tranges.min()), 0.5/dt

    fqL = infqL
    if not isinstance(infqL, (list, np.ndarray)):
        if infqL>0:
            fqL = np.logspace(np.log10(f2), np.log10(f3), infqL+5)
            fqL = fqL[[0, 2, 4] + range(5, infqL+4)]
        else:
            fqL = np.linspace(f2, f3, -infqL+5)
            fqL = fqL[[0, 2, 4] + range(5, -infqL+4)]
        fqL = np.concatenate(([0.5*f1], fqL[:-1], [2*f3]))

    fqL = np.array(fqL)
    
    nfq = len(fqL) - 1
    fq = 10**(np.log10( (fqL[:-1]*fqL[1:]) )/2.)
    return fqL, fq


def simulate_psd_cython():
    """Run a simple plag psd simulation. do_sig=0"""

    # input #
    for k in ['norm', 'dt', 'psdpar']:
        exec('{0} = sim_input["{0}"]'.format(k))

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    def neg_lnlike(p, m):
        return -m.logLikelihood(p, 1, 0)

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None)

        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        

        model = plag._plag.psd(T[0], R[0], E[0], dt, fqL, inorm, 0)
        p0 = np.ones(len(fqL)-1)
        res = opt.minimize(neg_lnlike, p0, args=(model,), method='L-BFGS-B',
                bounds=[(-20,20)]*len(fq))
        sims.append(res.x)

    sims = np.array(sims)
    sm, ss = sims.mean(0), sims.std(0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.psd_model[:,1:]
    ii = np.logical_and(fm>fq[0], fm<fq[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm))
    plt.errorbar(fq, sm, ss, fmt='o')
    plt.savefig('psd_cython.png')
    np.savez('psd_cython.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)
    

def simulate_psd_cython_2():
    """Run a simple plag psd simulation. do_sig=1"""

    # input #
    for k in ['norm', 'dt', 'psdpar']:
        exec('{0} = sim_input["{0}"]'.format(k))

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    def neg_lnlike(p, m):
        return -m.logLikelihood(p, 1, 0)

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None)

        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        

        model = plag._plag.psd(T[0], R[0], E[0], dt, fqL, inorm, 1)
        p0 = np.ones(len(fqL))
        res = opt.minimize(neg_lnlike, p0, args=(model,), method='L-BFGS-B',
                bounds=[(-2,2)]+[(-20,20)]*len(fq))
        sims.append(res.x)

    sims = np.array(sims)
    sm, ss = sims.mean(0), sims.std(0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.psd_model[:,1:]
    ii = np.logical_and(fm>fq[0], fm<fq[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm))
    plt.errorbar(fq, sm[1:], ss[1:], fmt='o')
    plt.title(r'$\sigma= {:4.4g}\pm{:4.4g}$'.format(sm[0], ss[0]))
    plt.savefig('psd_cython_2.png')
    np.savez('psd_cython_2.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)



if __name__ == '__main__':

    p = argparse.ArgumentParser(                                
        description="Run simulations for plag",            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('nsim', type=int, help='number of simulations')

    p.add_argument('--psd_cython', action='store_true', default=False,
            help='Simulate psd_cython with do_sig=0')
    p.add_argument('--psd_cython_2', action='store_true', default=False,
            help='Simulate psd_cython with do_sig=1')     


    args = p.parse_args()


    ## psd ##
    if args.psd_cython:
        simulate_psd_cython()

    if args.psd_cython_2:
        simulate_psd_cython_2()