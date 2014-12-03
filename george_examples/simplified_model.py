##!/usr/bin/env python3.3
## -*- coding: utf-8 -*-
"""this script is modified from
https://github.com/dfm/george/blob/master/docs/_code/model.py
"""

from __future__ import division, print_function

import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
import subprocess
import sys
import cPickle

import george
from george import kernels


def model(params, t):
    amp, loc, sig2 = params
    return amp * np.exp(-0.5 * (t - loc) ** 2 / sig2)


def lnlike_gp(p, t, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.ExpSquaredKernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y, t)


def lnprior_gp(p):
    lna, lntau = p[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf
    return 0.0


def lnprob_gp(p, t, y, yerr):
    lp = lnprior_gp(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, t, y, yerr)


def fit_gp(initial, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]

    # allow some flexibility in choice of starting parameters
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


def generate_data(params, N, rng=(-7, 7)):
    """
    N = integer, number of data points
    """
    hp = np.array(params)
    print("parameters for data generated from gp are : {0}".format(hp))
    print("using a ExpSquared kernel")
    gp = george.GP(hp[0] * kernels.ExpSquaredKernel(hp[1]))
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))

    #y = model(params, t)
    y = gp.sample(t)
    yerr = 1.e-5 #1 + 0.1 * np.random.randn(N)
    y += yerr

    return t, y, yerr


if __name__ == "__main__":
    #np.random.seed(1331)
    np.random.seed(2)

    truth = np.array([1., 1.])
    print("true parameters are {0}".format(truth))
    rng = (-7, 7)

    t, y, yerr = generate_data(truth, 50)

    # Fit assuming GP.
    print("Fitting GP")
    data = (t, y, yerr)

    # first two are hyperparameters to the kernel
    sampler = fit_gp(truth, data)

    samples = sampler.flatchain
    x = np.linspace(rng[0], rng[1], 500)
    pl.figure()
    pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

    for s in samples[np.random.randint(len(samples), size=24)]:
        gp = george.GP(np.exp(s[0]) * kernels.ExpSquaredKernel(np.exp(s[1])))
        gp.compute(t, yerr)
        m = gp.sample_conditional(y, x)
        pl.plot(x, m, color="#4682b4", alpha=0.3)
        #print(np.exp(s))

    pl.ylabel(r"$y$")
    pl.xlabel(r"$t$")
    pl.xlim(rng[0], rng[1])
    pl.title("results with {0}".format("ExpSquaredKernel"))
    pl.savefig("simplified_data.png")

    # Make the corner plot.
    log_labels = [r"$\ln$ $a^2$", r"$\ln$ $\tau$"]
    labels = [r"$a^2$", r"$\tau$"]

    #cPickle.dump(samples, open("simple1D_samples.pkl", "w"))
    #cPickle.dump(truth, open("simple1D_truth.pkl", "w"))

    #fig = triangle.corner(samples[:, 2:], truths=truth, labels=labels)

    # MCMC infers the hyperparameters in the natural log space
    # either exponentiate the chain values / take log of the true value

    # only plot the hyperparameters
    #fig = triangle.corner(samples, truths=np.log(truth), labels=log_labels)
    #fig.savefig("simplified_log_gp-corner.png", dpi=150)

    ## in the original scale
    #fig = triangle.corner(np.exp(samples), truths=truth, labels=labels)
    #fig.savefig("simplified_gp-corner.png", dpi=150)

    #subprocess.call(["open", "simplified_log_gp-corner.png"])
    #subprocess.call(["open", "simplified_gp-corner.png"])
    #subprocess.call(["open", "simplified_data.png"])


