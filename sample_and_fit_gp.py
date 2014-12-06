""" helper functions for sampling and fitting data from 2D Gaussian process
Author : Karen Ng  <karenyng@ucdavis.edu>
"""
from __future__ import division, print_function
import emcee
import numpy as np
import george
from george import kernels

# ------- helper functions for generating functions -------------


def to_george_param(p):
    """convert lambDa and rho to the parameters of George"""
    lambDa, rho = p
    assert rho >= 0. and rho <= 1., \
        "input value for rho is {0},".format(rho) + \
        " rho has to be 0 <= rho <= 1"

    return [1. / lambDa, -4. * np.log(rho)]


def from_george_param(p_ge):
    """gives lambDa and rho from the parameters of George"""
    return [1. / p_ge[0], np.exp(-p_ge[1]/ 4.)]


def char_dim(rho):
    """convert number from the george parametrization to our parametrization"""
    return np.sqrt(-1. / np.log(rho))


def generate_2D_data(truth, spacing, rng=(0., 60.), noise_amp=1e-4,
                     george_param=True):
    """
    :param:
    truth = list of floats, first two the hyperparameters for the GP
        the rest of the floats are for the model
    spacing = float, spacing between grid points
    rng = tuple of two floats, end points of the grid in each dimension
    george_param = bool, whether the parameterization was in the format
        required by george

    :return:
    coords = grid points
    psi = GP sample values in 1D
    psi_err = gaussian noise
    """
    # by default the mean vector from which the Gaussian data
    # is drawn is zero

    if george_param is False:
        truth = to_george_param(truth)

    gp = george.GP(truth[0] * kernels.ExpSquaredKernel(truth[1], ndim=2))

    # use regular grid
    xg = np.arange(rng[0], rng[1], spacing)
    yg = np.arange(rng[0], rng[1], spacing)
    coords = np.array([[x, y] for x in xg for y in yg])

    psi = gp.sample(coords)

    #psi += model(truth[2:], coords)

    # not sure if I am generating psi_err correctly
    # this is the independent Gaussian noise that I add
    psi_err = noise_amp + noise_amp * np.random.rand(len(psi))
    psi += psi_err   #  * np.random.randn(len(psi))

    return coords, psi, psi_err

#------- to be written ------------------------------------

def jacobian():
    return


def model(p, coords):
    """trivial (const) one for testing purpose"""
    return 0  #p[0] * np.ones(coords.shape[0])


def standardize_data(psi):
    """ psi needs to be flattened to one dimension"""
    return (psi - np.mean(psi)) / np.std(psi)

# -------- helper functions for calling emcee ---------------

def lnlike_gp(truth, coord, psi, psi_err):
    hp = truth[:2]
    p = truth[2:]
    # update kernel parameters
    gp = george.GP(hp[0] * kernels.ExpSquaredKernel(hp[1], ndim=2.))

    # this step is very revealing that we are fitting the psi_err
    # when we try to find the hyperparameters
    # psi_err is going to be added to the covariance matrix of the kernel
    # by george
    gp.compute(coord, psi_err)
    return gp.lnlikelihood(psi - model(p, coord))


def lnprior_gp(hp, prior_vals=None):
    prior_vals = np.array(prior_vals)

    if prior_vals is not None:
        assert prior_vals.shape[0] == len(hp), \
            "wrong # of rows in prior_vals {0}".format(prior_vals.shape[0]) + \
            " that do not match no of params {0}".format(len(hp))
        assert prior_vals.shape[1] == 2, \
            "wrong # of cols in prior_vals {0}".format(prior_vals.shape[2])
    else:
        prior_vals = [[-1, 1], [-1, 0.2]]

    # uniform in the log spacing - i didn't initialize this correctly before
    lna, lntau = hp[:2]
    if not prior_vals[0][0] < lna < prior_vals[0][1]:
        return -np.inf
    if not prior_vals[1][0] < lntau < prior_vals[1][1]:
        return -np.inf

    # not exactly right, should also add the lnprior_base terms
    # if our model is non-trivial
    return 0.0


def lnprob_gp(truth, coords, psi, psi_err, prior_vals=[[-1, 1], [-1, 0]]):
    hp = truth[:2]
    #p = truth[2:]  #
    lp = lnprior_gp(hp, prior_vals=prior_vals)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(truth, coords, psi, psi_err)


def fit_gp(initial, data, nwalkers=8, guess_dev_frac=1e-2,
           prior_vals=[[-1, 1], [-1, 0]]):
    """
    :param
    initial = list / array of initial guesses of the truth value of the hp
    data = tuple (t, y, yerr),
        t = numpy array of coord grid,
        y = flattened (1D) numpy array of data,
        yerr = flattened (1D) numpy array of data err
    """
    ndim = len(initial)
    p0 = [np.array(initial) +
          guess_dev_frac * np.array(initial) * np.random.randn(ndim)
          for i in xrange(nwalkers)]

    map(lambda x: print("initial guesses was {0}".format(x)), p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data,
                                    kwargs={"prior_vals": prior_vals})

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
    print("the optimized p0 is {0}".format(p0))

    return sampler, p0

