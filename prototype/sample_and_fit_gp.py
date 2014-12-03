""" helper functions for sampling and fitting data from 2D Gaussian process
Author : Karen Ng  <karenyng@ucdavis.edu>
"""
from __future__ import division


import emcee
import numpy as np
import george
from george import kernels
import matplotlib.pyplot as plt


def model(p, coords):
    """trivial (const) one for testing purpose"""
    return 0  #p[0] * np.ones(coords.shape[0])


def generate_2D_data(truth, spacing, rng=(0., 60.), noise_amp=1e-4):
    """
    :param:
    truth = list of floats, first two the hyperparameters for the GP
        the rest of the floats are for the model
    spacing = float, spacing between grid points
    rng = tuple of two floats, end points of the grid

    :return:
    coords = grid points
    psi = GP sample values in 1D
    psi_err = gaussian noise
    """
    # by default the mean vector from which the Gaussian data
    # is drawn is zero
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
    psi += psi_err #* np.random.randn(len(psi))

    return coords, psi, psi_err


def to_george_param(p):
    lambDa = p[0]
    rho = p[1]
    assert rho >= 0. and rho <= 1., \
        "input value for rho is {0},".format(rho) + \
        "needed rho has to be 0 <= rho <= 1"

    return [1. / lambDa, -4. * np.log(rho)]


def from_george_param(p_ge):
    """ gives lambDa and rho from the parameters of George """
    return [1. / p_ge[0], np.exp(-p_ge[1]/ 4.)]


def char_dim(rho):
    """convert number from the george parametrization to our parametrization"""
    return np.sqrt(-2. / np.log(rho))


def plot_2D_gp_samples(psi_s, coord_grid, figside, lambDa, rho, range_No):
    char_length = char_dim(rho)
    color = psi_s
    fig, ax = plt.subplots()
    im = plt.scatter(coord_grid.transpose()[1], coord_grid.transpose()[0],
                     s=35, cmap=plt.cm.jet, c=color)
    fig.set_figwidth(figside * 1.04)

    fig.set_figheight(figside)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(r"ExpSq kernel: $\lambda=$" +
                 "{0:.2f}, ".format(lambDa) + r"$\rho=$" +
                 "{0:.2f},".format(rho) +
                 r" $ l=$" + "{0:.2f}".format(char_length),
                 fontsize=20)
    ax.set_xlabel("arcsec ({0} arcsec per side)".format(range_No),
                  fontsize=20)
    ax.set_ylabel("arcsec ({0} arcsec per side)".format(range_No),
                  fontsize=20)
    plt.show()


def plot_2D_gp_contour(psi_s, coord_grid, figside, lambDa, rho, range_No,
                       spacing, unit="arbitrary unit"):
    """
    :params
    psi_s = flattened (1D) version of the psi_s data
    coord_grid = 2D numpy array of floats
    figside = float, in inches how big the figure should be
    lambDa = float
    rho = float
    """
,
    char_length = char_dim(rho)
    xg = np.arange(0, range_No, spacing)
    yg = xg
    fig = plt.figure()
    fig.set_figheight(figside)
    fig.set_figwidth(figside)

    ax = fig.add_subplot(111, aspect='equal')
    im = ax.contourf(xg, yg, psi_s)
    unit = "arbitrary unit"
    ax.set_xlabel("{0} ({1} {0} per side)".format(unit, range_No),
                fontsize=20)
    ax.set_ylabel("{0} ({1} {0} per side)".format(unit, range_No),
                fontsize=20)
    ax.set_title(r"ExpSq kernel: $\lambda=$" +
                "{0}, ".format(lambDa) + r"$\rho=$" +
                "{0:.2f},".format(rho) +
                r" $ l=$" + "{0:.2f}".format(char_length),
                fontsize=20)
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.show()
    return


def lnlike_gp(hp, p, coord, psi, psi_err):
    gp = george.GP(hp[0] * kernels.ExpSquaredKernel(hp[1], ndim=2.))
    # this step is very revealing that we are fitting the psi_err
    # when we try to find the hyperparameters
    gp.compute(coord, psi_err)
    return gp.lnlikelihood(psi - model(p, coord))


def lnprior_gp(hp):
    # uniform in the log spacing - i didn't initialize this correctly before
    lna, lntau = hp[:2]
    if not -5 < lna < 5:
        return -np.inf
    if not -5 < lntau < 5:
        return -np.inf

    # not exactly right, should also add the lnprior_base terms
    return 0.0


def lnprob_gp(hp, p, coords, psi, psi_err):
    lp = lnprior_gp(hp)
    # might also needed these
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(hp, p, coords, psi, psi_err)


def fit_gp(initial, data, nwalkers=8):
    ndim = len(initial)
    p0 = [np.array(initial) + .5 * np.random.randn(ndim)
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
    print "the optimized p0 is {0}".format(p0)

    return sampler, p0



if __name__ == "__main__":
    pass


