""" helper functions for sampling and fitting data from 2D Gaussian process
Author : Karen Ng  <karenyng@ucdavis.edu>
"""
from __future__ import division, print_function
import emcee
import numpy as np
import george
from george import kernels
from scipy.special import gamma

# ------- helper functions for generating functions -------------


def to_george_param(p):
    """convert lambDa and rho to the parameters of George
    :param p = tuple / list of two parameters
    """
    lambDa, rho = p
    assert rho >= 0. and rho <= 1., \
        "input value for rho is {0},".format(rho) + \
        " rho has to be 0 <= rho <= 1"

    return [1. / lambDa, -4. * np.log(rho)]


def from_george_param(p_ge):
    """gives lambDa and rho from the parameters of George"""
    return [1. / p_ge[0], np.exp(-p_ge[1] / 4.)]


def char_dim(rho):
    """convert number from the george parametrization to our parametrization"""
    return np.sqrt(-1. / np.log(rho))


def make_grid(rng, spacing):
    # use regular grid
    xg = np.arange(rng[0], rng[1], spacing)
    yg = np.arange(rng[0], rng[1], spacing)
    return np.array([[x, y] for x in xg for y in yg])


def generate_2D_data(truth, spacing, rng=(0., 60.), noise_amp=1e-4,
                     george_param=True):
    """
    :param:
    truth = list of floats, first two are the hyperparameters for the GP
        the rest of the floats are for the model
    spacing = float, spacing between grid points
    rng = tuple of two floats, end points of the grid in each dimension
    george_param = bool, whether the parameterization was in the format
        required by george

    :return:
    coords = 2D numpy array, grid points
    psi = numpy array, GP sample values in 1D
    psi_err = numpy array, gaussian noise
    """
    # by default the mean vector from which the Gaussian data
    # is drawn is zero

    if george_param is False:
        truth = to_george_param(truth)

    gp = george.GP(truth[0] * kernels.ExpSquaredKernel(truth[1], ndim=2))

    coords = make_grid(rng, spacing)

    psi = gp.sample(coords)

    #psi += model(truth[2:], coords)

    # not sure if I am generating psi_err correctly
    # this is the independent Gaussian noise that I add
    psi_err = noise_amp + noise_amp * np.random.rand(len(psi))
    psi += psi_err   #  * np.random.randn(len(psi))

    return coords, psi, psi_err


def draw_cond_pred(s_param, fine_coords, psi, psi_err, coords):
    """
    """
    gp = george.GP(s_param[0] *
                   kernels.ExpSquaredKernel(s_param[1], ndim=2))
    gp.compute(coords, psi_err)
    return gp.sample_conditional(psi, fine_coords)


#------- to be tested ------------------------------------

def jacobian():
    return


def model(p, coords):
    """trivial (const) one for testing purpose"""
    return 0  #p[0] * np.ones(coords.shape[0])


def standardize_data(coords):
    """ scales data to have mean zero and std. dev. of 1 ....
    Not the best for data with outliers / heavy tails

    """
    #psi = psi.copy()
    #return (psi - np.mean(psi.ravel())) / np.std(psi.ravel())
    return


def normalize_data(coords):
    """scale data coords between the range of [0, 1] and returns a copy of the
    data
    """
    #psi = psi.copy()
    #return (psi - psi.ravel().min()) / (psi.ravel().max() - psi.ravel().min())
    return


def invgamma_pdf(x, alpha, beta):
    """pdf of inv gamma dist. with wiki parametrization
    Suggested usage:
    ----------------
        prior of the inverse precision param

    Params:
    ------
        x (float / np.array): value to evalute at
        alpha (float): scale parameter, real number > 0
        beta (float): shape parameter, real number > 0

    Returns:
    -------
        float / an array of floats

    Stability:
    ---------
        passed one test

    .. math::
        f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}
            x^{(-\alpha-1)} \exp\left(-\beta / x \right)
    """
    return beta ** alpha / gamma(alpha) * \
               x ** (-alpha - 1.) * np.exp(-beta / x)


def beta_pdf():
    """
    work in progress
    """
    return



# -------- helper functions for calling emcee ---------------

def lnlike_gp(truth, coord, psi, psi_err=1e-10):
    """ we initialize the lnlike_gp to be the ln likelihood computed by
    george given the data points

    Parameters:
    -----------
        truth : list of floats
            expect a format of [ln_hp1, ln_hp2, p1, p2, ..., pn]
            where the first two hyperparameters for the kernel function for
            George are in log scale
        coord : 2D numpy array
            each row consists of the 2 coordinates of a grid point
        psi : numpy array
            this is in 1D after using the ravel() function for the 2D values
        psi_err : numpy array
            same shape and size as psi but for the errors

    Returns:
    -------
        likelihood value : float

    """
    # to be consistent with how we set up lnprior fnction with truth of hp
    # being in the log scale, we have to exponentiate this
    hp = np.exp(truth[:2])

    p = truth[2:]
    # update kernel parameters

    gp = george.GP(hp[0] * kernels.ExpSquaredKernel(hp[1], ndim=2.))

    # this step is very revealing that we are fitting the psi_err
    # when we try to find the hyperparameters
    # psi_err is going to be added to the covariance matrix of the kernel
    # by george
    if len(psi_err) == 1:
        psi_err = psi_err * np.ones(len(coord))
    gp.compute(coord, psi_err)
    return gp.lnlikelihood(psi - model(p, coord))


def lnprior_gp(ln_hp, lnprior_vals=None, verbose=False):
    """ function for the george lnprior

    Parameters :
    ----------
        ln_hp : list of two floats
            these are the hyperparameters for the kernel function
            theta1 and theta2, these are in log space
        lnprior_vals : list of list of prior values
            e.g. [[ln_prior_low_lim1, ln_prior_up_lim1],
                  [ln_prior_low_lim2, ln_prior_up_lim2],
                    ... ]

    Returns :
    --------
        float : prior value

    """

    if lnprior_vals is not None:
        lnprior_vals = np.array(lnprior_vals)
        assert lnprior_vals.shape[0] == len(ln_hp), \
            "wrong # of rows in lnprior_vals {0}".format(lnprior_vals.shape[0]) + \
            " that do not match no of params {0}".format(len(ln_hp))
        assert lnprior_vals.shape[1] == 2, \
            "wrong # of cols in lnprior_vals {0}".format(lnprior_vals.shape[2])
    else:
        lnprior_vals = [[-10, 10.], [-10., 10]]
        if verbose:
            print("No prior vals given, setting them to " +
                  "{0}".format(lnprior_vals))

    # uniform in the log spacing - i didn't initialize this correctly before
    lna, lntau = ln_hp[:2]
    if not lnprior_vals[0][0] < lna < lnprior_vals[0][1]:
        return -np.inf
    if not lnprior_vals[1][0] < lntau < lnprior_vals[1][1]:
        return -np.inf

    # not exactly right, should also add the lnprior_base terms
    # if our model is non-trivial
    return 0.0


def lnprob_gp(lnHP_truth, coords, psi, psi_err=1e-10,
              lnprior_vals=[[-10, 10], [-10, 10]]):
    """the log posterior prob that emcee is going to evaluate

    Params:
    -------
        lnHP_truth
    """
    ln_hp = lnHP_truth[:2]
    # p = truth[2:]
    lp = lnprior_gp(ln_hp, lnprior_vals=lnprior_vals)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(lnHP_truth, coords, psi, psi_err)


def draw_initial_guesses(initial, guess_dev_frac, ndim, nwalkers):
    return [np.array(initial) +
            guess_dev_frac * np.array(initial) * np.random.randn(ndim)
            for i in xrange(nwalkers)]


def fit_gp(initial, data, nwalkers=8, guess_dev_frac=1e-6,
           lnprior_vals=[[-10., 10.], [-10., 10]], burnin_chain_len=int(1e3),
           conver_chain_len=int(5e3), a=2.0, threads=1, pool=None):
    """
    Parameters
    ----------
        initial : list / array
            of initial guesses of the truth value of the **log** of hp
        data : tuple (t, y, yerr),
            t : numpy array of coord grid,
            y = flattened (1D) numpy array of data,
            yerr = flattened (1D) numpy array of data err
        nwalkers : integer,
            number of MCMC chains to use
        guess_dev_frac : float, has to be > 0 and around 1,
            initial values of each chain is
            (init_value * (1 + guess_dev_frac * rand_float)) where rand_float
            is drawn from a unit variance normal
        a : float, proposal scale parameter, see GW 10 or the emcee paper at
            http://arxiv.org/abs/1202.3665, increase value to decrease
            acceptance_fraction and vice versa
        threads : integer, number of threads to use for parallelization
        pool : integer, number of pool processes to use for parallelization
        sampler : allow the use of

    Returns
    ------
        sampler : emcee sampler object
        p0 : list of floats
            parameter values
    """
    ndim = len(initial)

    # initialize starting points and make sure that the initial guesses
    # are within the prior range
    count = 0
    p0 = draw_initial_guesses(initial, guess_dev_frac, ndim, nwalkers)

    # make sure prior values are reasonable
    while(np.sum(map(lambda x: lnprior_gp(x, lnprior_vals=lnprior_vals), p0))):
        p0 = draw_initial_guesses(initial, guess_dev_frac, ndim, nwalkers)
        count += 1
        if count > 1e3:
            raise ValueError("Cannot initialize reasonable chain values " +
                             "within prior range")

    map(lambda x: print("Initial guesses were {0}".format(np.exp(x))), p0)
    # needs a check here to make sure that the initial guesses are not
    # outside the prior range
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, a=a, args=data,
                                    kwargs={"lnprior_vals": lnprior_vals},
                                    threads=threads, pool=pool)

    print("Running burn-in with length {0:d}".format(burnin_chain_len))
    p0, lnp, _ = sampler.run_mcmc(p0, burnin_chain_len)
    sampler_acceptance_check(sampler)
    sampler.reset()

    print("Running second burn-in with length {0:d}".format(burnin_chain_len))
    p = p0[np.argmax(lnp)]
    p0 = [p + guess_dev_frac * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, burnin_chain_len)
    sampler_acceptance_check(sampler)
    sampler.reset()

    print("Running production chain with length {0}".format(conver_chain_len))
    p0, _, _ = sampler.run_mcmc(p0, conver_chain_len)
    print("the optimized p0 values are \n{0}".format(p0))

    return sampler, p0


def sampler_acceptance_check(sampler):
    if np.any(sampler.acceptance_fraction < 0.2):
        error_msg = "Initial guesses may be bad / model may be bad \n" + \
                    "Acceptance rate is < 0.2, currently at \n" + \
                    str(sampler.acceptance_fraction)
        raise ValueError(error_msg)
    return None


def Rubin_Gelman_Rsq_score():
    return
