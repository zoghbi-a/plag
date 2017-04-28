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
    'lag'   : 1.0,
    'phase' : True,
    'fname' : 'sim.lc',
}


def _simulate_lc(seed=34789, return_sim=False, dolag=False):
    """Simulate light curves and split them
        to segments

    Args:
        seed: random seed
        return_sim: return SimLC object
        dolag: do lags too?

    Returns:
        t, r, e: where each is a list of arrays

    """

    # input #
    for k in ['n', 'dt', 'mean', 'psdpar', 'norm', 'gnoise', 'seglen']:
        exec('{0} = sim_input["{0}"]'.format(k))


    sim = az.SimLc(seed=seed)
    sim.add_model(*psdpar)
    sim.simulate(16*n, dt/4, mean, norm)
    if dolag:
        lag, phase = sim_input['lag'], sim_input['phase']
        sim.add_model('constant', lag, lag=True)
        sim.apply_lag(phase)

    if return_sim: return sim

    r = sim.x[4*n:8*n].reshape((n, 4)).mean(1)
    t = sim.t[4*n:8*n].reshape((n, 4)).mean(1)
    if dolag:
        s = sim.y[4*n:8*n].reshape((n, 4)).mean(1)
    if gnoise is None:
        # poisson noise #
        raise NotImplemented
    else:
        # gaussian noise #
        r += np.random.randn(n)*gnoise
        e = np.ones(n)*gnoise
        if dolag:
            s += np.random.randn(n)*gnoise
    

    # split to segments #
    r, idx = az.misc.split_array(r, seglen, index=True)
    t = [t[i] for i in idx]
    e = [e[i] for i in idx] 
    if dolag: s = [s[i] for i in idx]


    if dolag:
        t = [list(x) for x in zip(t,t)]
        r = [list(x) for x in zip(r,s)]
        e = [list(x) for x in zip(e,e)]
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



def simulate_psdf_cython():
    """Run a simple plag psdf simulation. do_sig=0; ifunc=1"""

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
        fqL = fqL[[0,-1]]
        

        
        model = plag._plag.psdf(T[0], R[0], E[0], dt, fqL, inorm, 0, 1, 50)
        p0 = np.array([1., 1.])
        res = opt.minimize(neg_lnlike, p0, args=(model,), method='Powell')
        sims.append(res.x)
    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = smod[:,1].mean(0), smod[:,1].std(0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.psd_model[:,1:]
    ii = np.logical_and(fm>fqL[0], fm<fqL[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm), lw=4)
    plt.fill_between(fs, np.log(ms-ss), np.log(ms+ss), alpha=0.4)
    plt.savefig('psdf_cython.png')
    np.savez('psdf_cython.npz', sims=sims, fqL=fqL, sim_input=sim_input, smod=smod)


def simulate_psdf_cython_2():
    """Run a simple plag psdf simulation. do_sig=1; ifunc=1"""

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
        fqL = fqL[[0,-1]]
        

        
        model = plag._plag.psdf(T[0], R[0], E[0], dt, fqL, inorm, 1, 1, 50)
        p0 = np.array([0., .1, -2.])
        res = opt.minimize(neg_lnlike, p0, args=(model,), method='Powell')
        sims.append(res.x)
    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = smod[:,1].mean(0), smod[:,1].std(0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.psd_model[:,1:]
    ii = np.logical_and(fm>fqL[0], fm<fqL[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm), lw=4)
    plt.fill_between(fs, np.log(ms-ss), np.log(ms+ss), alpha=0.4)
    plt.savefig('psdf_cython_2.png')
    np.savez('psdf_cython_2.npz', sims=sims, fqL=fqL, sim_input=sim_input, smod=smod)


def simulate_psdf_cython_3():
    """Run a simple plag psdf simulation. do_sig=0; ifunc=13
        PL + LOR
    """

    # input #
    for k in ['n', 'dt', 'mean', 'norm', 'gnoise', 'seglen']:
        exec('{0} = sim_input["{0}"]'.format(k))

    psdpar = [['powerlaw', [1., -2]], ['lorentz', [50, 8e-2, 3e-2]]]
    sim = az.SimLc(seed=None)
    sim.add_model(*psdpar[0])
    sim.add_model(psdpar[1][0], psdpar[1][1], clear=False)


    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    def neg_lnlike(p, m):
        return -m.logLikelihood(p, 1, 0)

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        sim.simulate(16*n, dt/4, mean, norm)
        r = sim.x[4*n:8*n].reshape((n, 4)).mean(1)
        t = sim.t[4*n:8*n].reshape((n, 4)).mean(1)
        r += np.random.randn(n)*gnoise
        e = np.ones(n)*gnoise
        R, idx = az.misc.split_array(r, seglen, index=True)
        T = [t[i] for i in idx]
        E = [e[i] for i in idx]


        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        fqL = fqL[[0,-1]]
        

        model = plag._plag.psdf(T[0], R[0], E[0], dt, fqL, inorm, 0, 13, 50)
        p0 = np.array([0.1, -2, 3, -2.5, -3.5])
        res = opt.minimize(neg_lnlike, p0, args=(model,), method='Powell')
        sims.append(res.x)
    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = smod[:,1].mean(0), smod[:,1].std(0)

    fm, pm = sim.psd_model[:,1:]
    ii = np.logical_and(fm>fqL[0], fm<fqL[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm), lw=4)
    plt.fill_between(fs, np.log(ms-ss), np.log(ms+ss), alpha=0.4)
    plt.savefig('psdf_cython_3.png')
    np.savez('psdf_cython_3.npz', sims=sims, fqL=fqL, sim_input=sim_input, smod=smod)



def simulate_lag_cython():
    """Run a simple plag psd/lag simulation. do_sig=0"""

    # input #
    for k in ['norm', 'dt', 'psdpar', 'lag', 'phase']:
        exec('{0} = sim_input["{0}"]'.format(k))

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    def neg_lnlike(p, m):
        return -m.logLikelihood(p, 1, 0)

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None, dolag=1)

        # get frequncy bins #
        fqL, fq = _get_fqL([t[0] for t in T])
        

        p0 = np.ones(len(fqL)-1)
        pm1 = plag._plag.psd(T[0][0], R[0][0], E[0][0], dt, fqL, inorm, 0)
        pm2 = plag._plag.psd(T[0][1], R[0][1], E[0][1], dt, fqL, inorm, 0)
        res1 = opt.minimize(neg_lnlike, p0, args=(pm1,), method='L-BFGS-B',
                bounds=[(-20,20)]*len(fq))
        res2 = opt.minimize(neg_lnlike, res1.x, args=(pm2,), method='L-BFGS-B',
                bounds=[(-20,20)]*len(fq))

        c0 = np.concatenate(((res1.x+res2.x)*0.3, fq*0+0.1))
        cm  = plag._plag.lag(T[0], R[0], E[0], dt, fqL, inorm, 0, res1.x, res2.x)
        res = opt.minimize(neg_lnlike, c0, args=(cm,), method='Powell')
        
        sims.append(np.concatenate((res1.x, res2.x, res.x)))

    sims = np.array(sims)
    sm, ss = sims.mean(0), sims.std(0)
    sim = _simulate_lc(None, return_sim=1, dolag=1)
    fm, pm = sim.psd_model[:,1:]
    fm, lm = np.array(sim.lag_model)[:,1:]
    ii = np.logical_and(fm>fq[0], fm<fq[-1])
    fm, pm, lm = fm[ii], pm[ii], lm[ii]
    nfq = len(fq)

    ax = plt.subplot(1, 2, 1); ax.set_ylim([-3, 8])
    plt.semilogx(fm, np.log(pm))
    plt.errorbar(fq, sm[:nfq], ss[:nfq], fmt='o')
    plt.errorbar(fq, sm[nfq:2*nfq], ss[nfq:2*nfq], fmt='o')
    plt.errorbar(fq, sm[2*nfq:3*nfq], ss[2*nfq:3*nfq], fmt='o')

    ax = plt.subplot(1, 2, 2); ax.set_ylim([-3, 3])
    plt.semilogx(fm, lm)
    plt.errorbar(fq, sm[3*nfq:], ss[3*nfq:], fmt='o')

    plt.savefig('lag_cython.png')
    np.savez('lag_cython.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)



if __name__ == '__main__':

    p = argparse.ArgumentParser(                                
        description="Run simulations for plag",            
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('nsim', type=int, help='number of simulations')

    p.add_argument('--psd_cython', action='store_true', default=False,
            help='Simulate psd_cython with do_sig=0')
    p.add_argument('--psd_cython_2', action='store_true', default=False,
            help='Simulate psd_cython with do_sig=1')     
    p.add_argument('--psdf_cython', action='store_true', default=False,
            help='Simulate psdf_cython with do_sig=0')
    p.add_argument('--psdf_cython_2', action='store_true', default=False,
            help='Simulate psdf_cython with do_sig=1')
    p.add_argument('--psdf_cython_3', action='store_true', default=False,
            help='Simulate psdf_cython with do_sig=0, ifunc=13. PL + LOR')

    p.add_argument('--lag_cython', action='store_true', default=False,
            help='Simulate lag_cython with do_sig=0')

    args = p.parse_args()


    ## psd ##
    if args.psd_cython:
        simulate_psd_cython()

    if args.psd_cython_2:
        simulate_psd_cython_2()

    if args.psdf_cython:
        simulate_psdf_cython()

    if args.psdf_cython_2:
        simulate_psdf_cython_2()

    if args.psdf_cython_3:
        simulate_psdf_cython_3()


    if args.lag_cython:
        simulate_lag_cython()

