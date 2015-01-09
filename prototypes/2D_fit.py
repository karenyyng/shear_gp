#!/bin/python
"""
module for quickly testing our 2D fits
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
sys.path.append("../")
import george
from george import kernels

import sample_and_fit_gp as sam
import diag_plots as dplot

# np.random.seed(1)  # fix the random seed for reproducibility
spacing = 5.
rng = (0, 60.)  # 55 s
threads = 4  # set how many processors to use for the MCMC
gen_data = False
out_filename = "results.h5"
burn_in = 100
total_runs = 100
debug = False

if debug:
    total_runs = 1

# parametrization of the Sq Exp kernel in George is:
# \begin{equation}
# k(r_{ij}) = \theta_1 \exp\left(\frac{r_{ij}^2 }{ 2. \theta_2^2}\right)
# \end{equation}
#
# Our parametrization ha the restriction of:
run_total_results = pd.DataFrame()

for run_no in range(total_runs):
    # reading and writing the results for each truth value
    # allows us to terminate between runs
    print ("run {0}".format(run_no))
    if run_no != 0:
        run_total_results = pd.read_hdf(out_filename, "df")

    truth = np.random.rand(2)
    truth[0] = truth[0] * np.random.randint(1, 10)
    truth[1] = truth[1] * np.random.randint(1, 10)
    noise_amp = 1e-1 * truth[0]
    run_result = pd.DataFrame({'hp1': truth[0],
                               'hp2': truth[1],
                               'noise_amp': noise_amp},
                              index=[run_no])

    coords, psi, psi_err = sam.generate_2D_data(truth, spacing, rng, noise_amp)

    # check that our drawn data points are not crazy
    #dplot.plot_2D_gp_samples(psi, coords, figside=6,
    #                         truth=truth, range_No=rng[1])

    ## set prior in log scale!!! very important to be consistent
    lnHP_truth = list(np.log(truth[:2]))

    # try to initialize a broad prior
    lnprior_vals = np.array([[-10., 10.], [-10., 10.]])
    data = (coords, psi, psi_err)

    # make sure that I didn't put in unreasonable prior
    assert sam.lnprior_gp(truth, lnprior_vals=lnprior_vals) is not -np.inf, \
        "invalid prior choice"

    ## first use the results optimized by george as initial starting location
    sampler, p0 = sam.fit_gp(lnHP_truth, data, guess_dev_frac=1e-3,
                             lnprior_vals=lnprior_vals, nwalkers=8,
                             threads=threads, conver_chain_len=int(1e3),
                             burnin_chain_len=int(5e2))


    ## Compute likelihood of fit using George
    # draw random samples in the chains and compute the fit
    samples = sampler.flatchain  # still in log space

    # compute the credible intervals and estimated location in log space
    ax1 = plt.subplot(111)
    ax2 = plt.subplot(111)
    est1 = dplot.histplot1d_part(ax1, samples.transpose()[0])
    est2 = dplot.histplot1d_part(ax2, samples.transpose()[1])

    est1 = np.exp(est1)
    est2 = np.exp(est2)
    run_result["hp1_loc"] = est1[0]
    run_result["hp1_low68"] = est1[1]
    run_result["hp1_up68"] = est1[2]
    run_result["hp1_low95"] = est1[3]
    run_result["hp1_up95"] = est1[4]

    run_result["hp2_loc"] = est2[0]
    run_result["hp2_low68"] = est2[1]
    run_result["hp2_up68"] = est2[2]
    run_result["hp2_low95"] = est2[3]
    run_result["hp2_up95"] = est2[4]

    # compute the marginal likelihood of initialized by loc of chains as gp
    # params
    hp = np.exp(np.array([est1[0], est2[0]]))
    gp = george.GP(hp[0] * kernels.ExpSquaredKernel(hp[1], ndim=2))
    gp.compute(coords, psi_err)
    run_result["loc_lnlikelihood"] = gp.lnlikelihood(psi)

    ## try using the true params to compute the marginal likelihood instead
    hp = truth
    gp = george.GP(hp[0] * kernels.ExpSquaredKernel(hp[1], ndim=2))
    gp.compute(coords, psi_err)
    run_result["truth_lnlikelihood"] = gp.lnlikelihood(psi)

    # store the results
    print ("storing results of run {0}".format(run_no))
    run_total_results = pd.concat([run_total_results, run_result])
    run_total_results.to_hdf(out_filename, "df")
