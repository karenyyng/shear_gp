#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import triangle
import numpy as np
import cPickle
import matplotlib.pyplot as pl

import george
from george import kernels


def model(params, t):
    amp, loc, sig2 = params
    return amp * np.exp(-0.5 * (t - loc) ** 2 / sig2)


def lnprior_base(p):
    amp, loc, sig2 = p
    if not -10 < amp < 10:
        return -np.inf
    if not -5 < loc < 5:
        return -np.inf
    if not 0 < sig2 < 3.0:
        return -np.inf
    return 0.0


def fit_ind(initial, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ind, args=data)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


def lnlike_gp(p, t, y, yerr):
    # a and tau needs to be exponentiated ?
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model(p[2:], t))


def lnprior_gp(p):
    lna, lntau = p[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf
    return lnprior_base(p[2:])


def lnprob_gp(p, t, y, yerr):
    lp = lnprior_gp(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, t, y, yerr)


def fit_gp(initial, data, nwalkers=32):
    ndim = len(initial)

    # start chains at slightly different places in parameter space
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler


def generate_data(params, N, rng=(-5, 5)):
    gp = george.GP(params[0] * kernels.ExpSquaredKernel(params[1]))

    # initialize t for drawing the data points
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))

    ## modify the following
    y = gp.sample(t)
    y += model(params[2:], t)
    yerr = 0.05 + 0.05 * np.random.rand(N)
    y += yerr * np.random.randn(N)

    # y = model(params[2:], t)
    # yerr = gp.sample(t)
    # 0.05 + 0.05 * np.random.rand(N)
    #y += yerr * np.random.randn(N)

    return t, y, yerr


if __name__ == "__main__":
    np.random.seed(1234)

    #truth = [0.1, 1.0, 0, 0.1, 0.4]
    truth = [0.1, 3.3, -1.0, 0.1, 0.4]
    t, y, yerr = generate_data(truth, 50)
    pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    pl.ylabel(r"$y$")
    pl.xlabel(r"$t$")
    pl.xlim(-5, 5)
    pl.title("simulated data")
    pl.savefig("data.png", dpi=150)

    ## Fit assuming independent.
    #print("Fitting independent")
    #data = (t, y, 1.0 / yerr ** 2)
    #truth_ind = [0.0, 0.0] + truth
    #sampler = fit_ind(truth_ind, data)

    ## Plot the samples in data space.
    #print("Making plots")
    #samples = sampler.flatchain
    #x = np.linspace(-5, 5, 500)
    #for s in samples[np.random.randint(len(samples), size=24)]:
    #    pl.plot(x, model(s[2:], x)+s[0]*x+s[1], color="#4682b4", alpha=0.3)
    #pl.title("results assuming uncorrelated noise")
    #pl.savefig("ind-results.png", dpi=150)

    ## Make the corner plot.
    ##fig = triangle.corner(samples[:, 2:], truths=truth, labels=labels)
    #fig = triangle.corner(samples[:, :], truths=truth, labels=labels)
    #fig.savefig("ind-corner.png", dpi=150)

    # Fit assuming GP.
    print("Fitting GP")
    data = (t, y, yerr)
    truth_gp = truth + 1e-8 * np.random.randn(len(truth))  # [0.0, 0.0] + truth[2:]
    sampler = fit_gp(truth_gp, data)

    # Plot the samples in data space.
    print("Making plots")
    samples = sampler.flatchain
    x = np.linspace(-5, 5, 500)
    pl.figure()
    pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    for s in samples[np.random.randint(len(samples), size=24)]:
        gp = george.GP(np.exp(s[0]) * kernels.Matern32Kernel(np.exp(s[1])))
        gp.compute(t, yerrtruth)
        m = gp.sample_conditional(y - model(s[2:], t), x) + model(s[2:], x)
        pl.plot(x, m, color="#4682b4", alpha=0.3)
    pl.ylabel(r"$y$")
    pl.xlabel(r"$t$")
    pl.xlim(-5, 5)
    pl.title("results with Gaussian process noise model")
    pl.savefig("gp-results.png", dpi=150)

    # Make the corner plot.
    labels = [r"$\ln a^2$", r"$\ln \tau$", r"$\alpha$", r"$\ell$", r"$\sigma^2$"]
    #fig = triangle.corner(samples[:, 2:], truths=truth, labels=labels)

    # follow the original script
    truth[0] = np.log(truth[0])
    truth[1] = np.log(truth[1])

    cPickle.dump(truth, open("truth.pkl", "w"))
    cPickle.dump(samples, open("samples.pkl", "w"))

    # only plot the hyperparameters
    fig = triangle.corner(samples, truths=truth, labels=labels, size=30)
    fig.savefig("gp-corner.png", dpi=150)


