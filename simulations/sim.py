#!/usr/bin/env python

import argparse
import numpy as np
import aztools as az
import scipy.optimize as opt

import matplotlib as mpl
mpl.use('Agg')
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
    'psdpar': ['powerlaw', [8, -2]],
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
    n      = sim_input['n']
    dt     = sim_input['dt']
    mean   = sim_input['mean']
    psdpar = sim_input['psdpar']
    norm   = sim_input['norm']
    gnoise = sim_input['gnoise']
    seglen = sim_input['seglen']


    sim = az.SimLC(seed=seed)
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
        #plt.plot(sim.t, sim.x, sim.t, sim.y); plt.show();exit(0)
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
    dt    = sim_input['dt']
    infqL = sim_input['infqL'] 


    # limits from the data #
    tranges = np.array([[t[-1]-t[0]] for t in tseg])
    f1, f2, f3 = 1./(tranges.max()), 1./(tranges.min()), 0.5/dt

    fqL = infqL
    if not isinstance(infqL, (list, np.ndarray)):
        if infqL>0:
            fqL = np.logspace(np.log10(f2), np.log10(f3), infqL+5)
            fqL = fqL[[0, 2, 4] + list(range(5, infqL+4))]
        else:
            fqL = np.linspace(f2, f3, -infqL+5)
            fqL = fqL[[0, 2, 4] + range(5, -infqL+4)]
        fqL = np.concatenate(([0.5*f1], fqL[:-1], [2*f3]))

    fqL = np.array(fqL)
    
    nfq = len(fqL) - 1
    fq = 10**(np.log10( (fqL[:-1]*fqL[1:]) )/2.)
    return fqL, fq


def neg_lnlike(p, m):
    l = m.logLikelihood(p, 1, 0)
    return -l
def neg_dlnlike(p, m):
    l,g,h = m.dLogLikelihood(p)
    if not np.isfinite(l):
        l = -1e4
        g = np.ones_like(p)
    return -l, -g

def simulate_psd_cython():
    """Run a simple plag psd simulation. do_sig=0"""

    # input #
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar'] 

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None)

        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        

        model = plag._plag.psd(T[0], R[0], E[0], dt, fqL, inorm, 0)
        p0 = np.ones(len(fqL)-1)
        # res = opt.minimize(neg_lnlike, p0, args=(model,), method='L-BFGS-B',
        #         bounds=[(-15,15)]*len(fq))
        res = opt.minimize(neg_dlnlike, p0, args=(model,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15)]*len(fq))
        sims.append(res.x)

    sims = np.array(sims)
    sm, ss = np.median(sims, 0), np.std(sims, 0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.normalized_psd[:,1:]
    ii = np.logical_and(fm>fq[0], fm<fq[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm))
    plt.errorbar(fq, sm, ss, fmt='o')
    plt.savefig('psd_cython.png')
    np.savez('psd_cython.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)
    

def simulate_psd_cython_2():
    """Run a simple plag psd simulation. do_sig=1"""

    # input #
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar'] 

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None)

        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        

        model = plag._plag.psd(T[0], R[0], E[0], dt, fqL, inorm, 1)
        p0 = np.ones(len(fqL))
        #res = opt.minimize(neg_lnlike, p0, args=(model,), method='L-BFGS-B',
        #        bounds=[(-2,2)]+[(-20,20)]*len(fq))
        res = opt.minimize(neg_dlnlike, p0, args=(model,), jac=True, 
            method='L-BFGS-B', bounds=[(-2,2)]+[(-15,15)]*len(fq))
        sims.append(res.x)

    sims = np.array(sims)
    sm, ss = np.median(sims, 0), np.std(sims, 0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.normalized_psd[:,1:]
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
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar'] 


    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2

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
        #res = opt.minimize(neg_lnlike, p0, args=(model,), method='Powell')
        res = opt.minimize(neg_dlnlike, p0, args=(model,), jac=True, 
            method='L-BFGS-B', bounds=[(-5,5)]+[(-3,2)])
        sims.append(res.x)
    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = np.median(smod[:,1], 0), np.std(smod[:,1], 0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.normalized_psd[:,1:]
    ii = np.logical_and(fm>fqL[0], fm<fqL[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm), lw=4)
    plt.fill_between(fs, np.log(ms-ss), np.log(ms+ss), alpha=0.4)
    plt.savefig('psdf_cython.png')
    np.savez('psdf_cython.npz', sims=sims, fqL=fqL, sim_input=sim_input, smod=smod)


def simulate_psdf_cython_2():
    """Run a simple plag psdf simulation. do_sig=1; ifunc=1"""

    # input #
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar'] 

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2

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
        #res = opt.minimize(neg_lnlike, p0, args=(model,), method='Powell')
        res = opt.minimize(neg_dlnlike, p0, args=(model,), jac=True, 
            method='L-BFGS-B', bounds=[(-2,2), (-5,5)]+[(-3,2)])
        sims.append(res.x)
    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = smod[:,1].mean(0), smod[:,1].std(0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.normalized_psd[:,1:]
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
    n      = sim_input['n']
    dt     = sim_input['dt']
    mean   = sim_input['mean']
    norm   = sim_input['norm']
    gnoise = sim_input['gnoise'] 
    seglen = sim_input['seglen']

    psdpar = [['powerlaw', [0.08, -2]], ['lorentz', [50, 8e-2, 1e-2]]]
    sim = az.SimLC(seed=None)
    sim.add_model(*psdpar[0])
    sim.add_model(psdpar[1][0], psdpar[1][1], clear=False)
    
    # plt.ion();embed();exit(0)
    # sim.simulate(16*n, dt/4, mean, norm)
    # fm, pm = sim.psd_model[:,1:]



    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2

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
        #res = opt.minimize(neg_lnlike, p0, args=(model,), method='Powell')
        res = opt.minimize(neg_dlnlike, p0, args=(model,), jac=True, 
            method='L-BFGS-B', bounds=[(-5,5)]*5)
        sims.append(res.x)
    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = smod[:,1].mean(0), smod[:,1].std(0)

    fm, pm = sim.normalized_psd[:,1:]
    ii = np.logical_and(fm>fqL[0], fm<fqL[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm), lw=4)
    plt.fill_between(fs, np.log(ms-ss), np.log(ms+ss), alpha=0.4)
    plt.savefig('psdf_cython_3.png')
    np.savez('psdf_cython_3.npz', sims=sims, fqL=fqL, sim_input=sim_input, smod=smod)



def simulate_lag_cython():
    """Run a simple plag psd/lag simulation. do_sig=0"""

    # input #
    sim_input['psdpar'] = ['broken_powerlaw', [.2, -1, -2, 1e-3]]
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar']
    lag    = sim_input['lag']
    phase  = sim_input['phase']


    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    sim_input['infqL'] = 4

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
        res1 = opt.minimize(neg_dlnlike, p0, args=(pm1,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15)]*len(fq))
        res2 = opt.minimize(neg_dlnlike, p0, args=(pm2,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15)]*len(fq))

        c0 = np.concatenate(((res1.x+res2.x)*0.3, np.random.randn(len(fq))*0.01))
        cm  = plag._plag.lag(T[0], R[0], E[0], dt, fqL, inorm, 0, res1.x, res2.x)
        
        res = opt.minimize(neg_lnlike, c0, args=(cm,), method='Powell')
        res = opt.minimize(neg_dlnlike, res.x, args=(cm,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15)]*len(fq)+[(-np.pi,np.pi)]*len(fq))
        
        
        sims.append(np.concatenate((res1.x, res2.x, res.x)))

    sims = np.array(sims)
    sm, ss = np.median(sims, 0), np.std(sims, 0)
    sim = _simulate_lc(None, return_sim=1, dolag=1)
    fm, pm = sim.normalized_psd[:,1:]
    fm, lm = np.array(sim.normalized_lag)[:,1:]
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


def simulate_lag_cython_2():
    """Run a simple plag psd/lag simulation. do_sig=1"""

    # input #
    sim_input['psdpar'] = ['broken_powerlaw', [.2, -1, -2, 1e-3]]
    for k in ['norm', 'dt', 'psdpar', 'lag', 'phase']:
        exec('{0} = sim_input["{0}"]'.format(k))


    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    sim_input['infqL'] = 4

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None, dolag=1)

        # get frequncy bins #
        fqL, fq = _get_fqL([t[0] for t in T])
        
        p0 = np.ones(len(fqL))
        pm1 = plag._plag.psd(T[0][0], R[0][0], E[0][0], dt, fqL, inorm, 1)
        pm2 = plag._plag.psd(T[0][1], R[0][1], E[0][1], dt, fqL, inorm, 1)
        res1 = opt.minimize(neg_dlnlike, p0, args=(pm1,), jac=True, 
            method='L-BFGS-B', bounds=[(-10,2)]+[(-15,15)]*len(fq))
        res2 = opt.minimize(neg_dlnlike, p0, args=(pm2,), jac=True, 
            method='L-BFGS-B', bounds=[(-10,1)]+[(-15,15)]*len(fq))

        c0 = np.concatenate(((res1.x+res2.x)*0.3, np.random.randn(len(fq))*0.01))
        cm  = plag._plag.lag(T[0], R[0], E[0], dt, fqL, inorm, 1, res1.x, res2.x)
        
        res = opt.minimize(neg_lnlike, c0, args=(cm,), method='Powell')
        res = opt.minimize(neg_dlnlike, res.x, args=(cm,), jac=True, method='L-BFGS-B', 
            bounds=[(-10,1)]+[(-15,15)]*len(fq)+[(-np.pi,np.pi)]*len(fq))
        
        
        sims.append(np.concatenate((res1.x, res2.x, res.x)))
    sims = np.array(sims)
    sm, ss = np.median(sims, 0), np.std(sims, 0)
    sim = _simulate_lc(None, return_sim=1, dolag=1)
    fm, pm = sim.psd_model[:,1:]
    fm, lm = np.array(sim.lag_model)[:,1:]
    ii = np.logical_and(fm>fq[0], fm<fq[-1])
    fm, pm, lm = fm[ii], pm[ii], lm[ii]
    nfq = len(fq)

    ax = plt.subplot(1, 2, 1); ax.set_ylim([-3, 8])
    plt.semilogx(fm, np.log(pm))
    plt.errorbar(fq, sm[1:(nfq+1)], ss[1:(nfq+1)], fmt='o')
    plt.errorbar(fq, sm[(nfq+2):(2*nfq+2)], ss[(nfq+2):(2*nfq+2)], fmt='o')
    plt.errorbar(fq, sm[(2*nfq+3):(3*nfq+3)], ss[(2*nfq+3):(3*nfq+3)], fmt='o')

    ax = plt.subplot(1, 2, 2); ax.set_ylim([-3, 3])
    plt.semilogx(fm, lm)
    plt.errorbar(fq, sm[(3*nfq+3):], ss[(3*nfq+3):], fmt='o')

    plt.savefig('lag_cython_2.png')
    

def simulate_lagf_cython():
    """Run a simple plag psdf simulation. do_sig=0; ifunc=1"""

    # input #
    sim_input['psdpar'] = ['broken_powerlaw', [.2, -1, -2, 1e-3]]
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar']
    lag    = sim_input['lag']
    phase  = sim_input['phase']


    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None, dolag=1)

        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        fqL = fqL[[0,-1]]



        p0 = np.zeros(2)
        pm1 = plag._plag.psdf(T[0][0], R[0][0], E[0][0], dt, fqL, inorm, 0, 1, 50)
        pm2 = plag._plag.psdf(T[0][1], R[0][1], E[0][1], dt, fqL, inorm, 0, 1, 50)
        embed();exit(0)
        res1 = opt.minimize(neg_dlnlike, p0, args=(pm1,), jac=True, 
            method='L-BFGS-B', bounds=[(-5,5)]+[(-3,2)])

        res1 = opt.minimize(neg_dlnlike, p0, args=(pm1,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15), (-15,15)]*len(fq))
        res2 = opt.minimize(neg_dlnlike, p0, args=(pm2,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15)]*len(fq))

        c0 = np.concatenate(((res1.x+res2.x)*0.3, np.random.randn(len(fq))*0.01))
        cm  = plag._plag.lag(T[0], R[0], E[0], dt, fqL, inorm, 0, res1.x, res2.x)
        
        res = opt.minimize(neg_lnlike, c0, args=(cm,), method='Powell')
        res = opt.minimize(neg_dlnlike, res.x, args=(cm,), jac=True, 
            method='L-BFGS-B', bounds=[(-15,15)]*len(fq)+[(-np.pi,np.pi)]*len(fq))
        
        
        sims.append(np.concatenate((res1.x, res2.x, res.x)))




    sims = np.array(sims)
    smod = np.array([model.calculate_model(s) for s in sims])
    fs = smod[0,0]
    ms, ss = np.median(smod[:,1], 0), np.std(smod[:,1], 0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.normalized_psd[:,1:]
    ii = np.logical_and(fm>fqL[0], fm<fqL[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm), lw=4)
    plt.fill_between(fs, np.log(ms-ss), np.log(ms+ss), alpha=0.4)
    plt.savefig('psdf_cython.png')
    np.savez('psdf_cython.npz', sims=sims, fqL=fqL, sim_input=sim_input, smod=smod)



def simulate_psd():
    """Run a simple plag psd simulation. do_sig=0"""

    # input #
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar ']


    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None)

        # get frequncy bins #
        fqL, fq = _get_fqL(T)
        
        model = plag.PLag('psd', T, R, E, dt, fqL, norm, False)
        p0 = np.ones(len(fqL)-1)
        # res = opt.minimize(neg_dlnlike, p0, args=(model,), jac=True, 
        #     method='L-BFGS-B', bounds=[(-15,15)]*len(fq))
        r = plag.optimize(model, p0, verbose=0)
        if np.any(r[1]>1e4): continue
        sims.append(r[:2])
    sims = np.array(sims)
    sm, ss = np.median(sims[:,0], 0), np.std(sims[:,0], 0)
    se = np.mean(sims[:,1], 0)
    sim = _simulate_lc(None, return_sim=1)
    fm, pm = sim.psd_model[:,1:]
    ii = np.logical_and(fm>fq[0], fm<fq[-1])
    fm, pm = fm[ii], pm[ii]
    plt.semilogx(fm, np.log(pm))
    plt.errorbar(fq, sm, ss, fmt='o')
    plt.fill_between(fq, sm-se, sm+se)
    plt.savefig('psd.png')
    np.savez('psd.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)


def simulate_lag():
    """Run a simple plag psd/lag simulation. do_sig=0"""

    # input #
    sim_input['psdpar'] = ['broken_powerlaw', [.2, -1, -2, 1e-3]]
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar']
    lag    = sim_input['lag']
    phase  = sim_input['phase']

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    sim_input['infqL'] = 4
    sim_input['seglen'] = 64

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None, dolag=1)

        # get frequncy bins #
        fqL, fq = _get_fqL([t[0] for t in T])
        

        p0 = np.ones(len(fqL)-1)
        pm1 = plag.PLag('psd', [t[0] for t in T], 
            [r[0] for r in R], [e[0] for e in E], dt, fqL, norm, False)
        pm2 = plag.PLag('psd', [t[1] for t in T], 
            [r[1] for r in R], [e[1] for e in E], dt, fqL, norm, False)
        res1 = plag.optimize(pm1, p0, verbose=0)[0]
        res2 = plag.optimize(pm2, p0, verbose=0)[0]
    
        c0 = np.concatenate(((res1+res2)*0.3, np.random.randn(len(fq))*0.01))
        cm  = plag.PLag('lag', T, R, E, dt, fqL, res1, res2, norm, False)

        res = plag.optimize(cm, c0)[0]
        
        sims.append(np.concatenate((res1, res2, res)))
    sims = np.array(sims)
    sm, ss = np.median(sims, 0), np.std(sims, 0)
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

    plt.savefig('lag.png')
    np.savez('lag.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)


def simulate_psdlag():
    """Run a simple plag psdlag simulation. do_sig=0"""

    # input #
    sim_input['psdpar'] = ['broken_powerlaw', [.2, -1, -2, 1e-3]]
    norm   = sim_input['norm']
    dt     = sim_input['dt']
    psdpar = sim_input['psdpar']
    lag    = sim_input['lag']
    phase  = sim_input['phase']

    inorm = 0 if norm == 'var' else 1 if norm == 'leahy' else 2
    sim_input['infqL'] = 4
    sim_input['seglen'] = 64

    sims = []
    for isim in range(1, args.nsim+1):

        az.misc.print_progress(isim, args.nsim+1, isim==args.nsim)

        # simulate lc #
        T, R, E = _simulate_lc(None, dolag=1)

        # get frequncy bins #
        fqL, fq = _get_fqL([t[0] for t in T])
        

        p0 = np.ones(len(fqL)-1)
        pm1 = plag.PLag('psd', [t[0] for t in T], 
            [r[0] for r in R], [e[0] for e in E], dt, fqL, norm, False)
        pm2 = plag.PLag('psd', [t[1] for t in T], 
            [r[1] for r in R], [e[1] for e in E], dt, fqL, norm, False)
        res1 = plag.optimize(pm1, p0, verbose=0)[0]
        res2 = plag.optimize(pm2, p0, verbose=0)[0]
    
        c0 = np.concatenate(((res1+res2)*0.3, np.random.randn(len(fq))*0.01))
        cm  = plag.PLag('lag', T, R, E, dt, fqL, res1, res2, norm, False)
        res = plag.optimize(cm, c0)[0]

        c0 = np.concatenate((res1, res2, res))
        plm = plag.PLag('psdlag', T, R, E, dt, fqL, norm, False)
        Res = plag.optimize(plm, c0)[0]
        
        sims.append(np.concatenate((res1, res2, res, Res)))
    sims = np.array(sims)
    embed();exit(0)
    sm, ss = np.median(sims, 0), np.std(sims, 0)
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

    plt.savefig('lag.png')
    np.savez('lag.npz', sims=sims, fq=fq, fqL=fqL, sim_input=sim_input)

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
    p.add_argument('--lag_cython_2', action='store_true', default=False,
            help='Simulate lag_cython with do_sig=1')
    p.add_argument('--lagf_cython', action='store_true', default=False,
            help='Simulate lagf_cython with do_sig=0')

    p.add_argument('--psd', action='store_true', default=False,
            help='Simulate psd with do_sig=0')
    p.add_argument('--lag', action='store_true', default=False,
            help='Simulate lag with do_sig=0')
    p.add_argument('--psdlag', action='store_true', default=False,
            help='Simulate psdlag with do_sig=0')

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
    if args.lag_cython_2:
        simulate_lag_cython_2()
    if args.lagf_cython:
        simulate_lagf_cython()

    if args.psd:
        simulate_psd()

    if args.lag:
        simulate_lag()

    if args.psdlag:
        simulate_psdlag()

