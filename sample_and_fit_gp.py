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
        first float is inv lambda
        second float is beta / the metric
    """
    lambDa, rho = p
    assert rho >= 0. and rho <= 1., \
        "input value for rho is {0},".format(rho) + \
        " rho has to be 0 <= rho <= 1"

    return [1. / lambDa, -4. * np.log(rho)]


def from_george_param(p_ge):
    """gives lambDa and rho from the parameters of George
    want to vectorize this?
    """
    return [1. / p_ge[0], np.exp(-4 * p_ge[1])]


def char_dim(rho):
    """convert number from the george parametrization to our parametrization
    used in some plots
    """
    return np.sqrt(-1. / np.log(rho))


def make_grid(rng, data_pts):
    """
    :param rng: list / tuple of two floats
        denotes the lower and upper range of the range
    :param spacing: positive float
    """
    # use regular grid
    xg = np.linspace(rng[0], rng[1], data_pts)
    return np.array([[x, y] for x in xg for y in xg])


def generate_2D_data(truth, data_pts_no_per_side, kernels, rng=(0., 1.),
                     noise_amp=1e-6, george_param=True,
                     white_kernel_as_nugget=True):
    """
    Parameters
    =========
    truth : list of floats, first two are the hyperparameters for the GP
        the rest of the floats are for the model
    data_pts_no_per_side : int, number of data points on a side
    kernels : list of two george.kernels objects
    noise_amp : float, small number that denotes Gaussian
        uncertainties on the data points at coordinates ``x``.
        This is added in quadrature to the digaonal of the covariance matrix.
    rng : tuple of two floats, end points of the grid in each dimension
    george_param : bool, whether the parameterization was in the format
        required by george

    Returns
    =======
    coords = 2D numpy array, grid points
    psi = numpy array, GP sample values in 1D
    """
    # by default the mean vector from which the Gaussian data
    # is drawn is zero

    if george_param is False:
        truth = to_george_param(truth)

    coords = make_grid(rng, data_pts_no_per_side)
    ExpSquaredLikeKernel = kernels[0]

    if white_kernel_as_nugget:
        WhiteKernel = kernels[1]
        gp = george.GP(truth[0] *
                       ExpSquaredLikeKernel(np.ones(2) * truth[1], ndim=2) +
                       WhiteKernel(noise_amp ** 2, ndim=2))
        # need to compute before we can sample from the kernel!
        # since we made use of the WhiteKernel, we put yerr = 0
        gp.compute(coords, yerr=0)
    else:
        gp = george.GP(truth[0] *
                       ExpSquaredLikeKernel(np.ones(2) * truth[1], ndim=2))

        # use yerr for adding diagonal noise,
        # yerr is added in quadrature by George implicitly
        gp.compute(coords, yerr=noise_amp)

    psi = gp.sample(coords)

    mtx = gp.get_matrix(coords)
    if np.linalg.slogdet(mtx)[0]:
        print("Kernel matrix is positive definite.")
        print("Cond # = {0:.2e}".format(np.linalg.cond(mtx)))
    else:
        print("WARNING: Kernel matrix is NOT positive definite.")

    return coords, psi  # , psi_err


def draw_cond_pred(s_param, fine_coords, psi, psi_err, coords):
    """
    this should be sampling from conditional distribution
    with Schur Complement as the Covariance matrix
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
    raise NotImplementedError
    #psi = psi.copy()
    #return (psi - np.mean(psi.ravel())) / np.std(psi.ravel())


def normalize_2D_data(coords):
    """scale data coords between the range of [0, 1] along each spatial
    dimension and returns a copy of the data
    :param coords: 2D np.array
    """
    n_spat_dim = coords.shape[1]

    norm = np.array([coords.transpose()[i].max() - coords.transpose()[i].min()
                     for i in range(n_spat_dim)])

    return coords / norm


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
    """pdf of the beta function which is the conjugate prior
    of the correlation parameter

    :note: have to think about which form of this we will need
    in the MCMC ... since we sample beta instead of correlation
    """
    return

# -------- helper functions for calling emcee ---------------


def lnlike_gp(ln_param, kernels, coord, psi, george_param=False):
    """ we initialize the lnlike_gp to be the ln likelihood computed by
    george given the data points

    Parameters:
    -----------
    ln_param : list of floats
        expect a format of [ln_hp1, ln_hp2, ln_hp3, p1, p2, ..., pn]
        where the first two hyperparameters for the kernel function for
        George are in log s3cale
    kernels : list of george.kernels object,
        first one should have same parameterization as ExpSquaredKernel,
        second one is the WhiteKernel
    coord : 2D numpy array
        each row consists of the 2 coordinates of a grid point
    psi : numpy array
        this is in 1D after using the ravel() function for the 2D values

    Returns:
    -------
    likelihood value : float

    """
    # to be consistent with how we set up lnprior fnction with truth of hp
    # being in the log scale, we have to exponentiate this
    hp = np.exp(ln_param[:3])
    if not george_param:
        hp[1] = 1. / hp[1]  # george's parametrization is 1 / beta

    # update kernel parameters
    ExpSquaredLikeKernel, WhiteKernel = kernels

    # DerivKernel objects can only accept list of 2 floats as beta
    gp = george.GP(hp[0] * ExpSquaredLikeKernel([hp[1], hp[1]], ndim=2.) +
                   # George adds diagonal error term in quadrature
                   WhiteKernel(hp[2] ** 2, ndim=2))

    # if type(psi_err) == float or type(psi_err) == int:
    #     psi_err = psi_err * np.ones(len(coord))

    # compute last 2 terms of marginal log likelihood stated
    # in the Rasmussen GP book eqn. 2.3
    # since we have kernel already used a WhiteKernel,
    # we shouldn't need a nugget
    gp.compute(coord, yerr=0.)

    # this computes the data dependent fit term of eqn. 2.3
    return gp.lnlikelihood(psi)  # - model(p, coord))


def ln_transformed_lnlike_gp(ln_param, kernels, coord, psi):
    hp = [np.exp(i) for i in ln_param[:2]] + list([ln_param[2]])

    # get kernel and update kernel parameters
    ExpSquaredLikeKernel, WhiteKernel = kernels

    # DerivKernel objects can only accept list of 2 floats as beta
    gp = george.GP(hp[0] * ExpSquaredLikeKernel([hp[1], hp[1]], ndim=2) +
                   # George adds diagonal error term in quadrature
                   WhiteKernel(hp[2] ** 2, ndim=2))

    # compute last 2 terms of marginal log likelihood stated
    # in the Rasmussen GP book eqn. 2.3
    # since we have kernel already combined with a WhiteKernel,
    # we shouldn't need a nugget
    gp.compute(coord, yerr=0.)

    # `lnlikelihood` computes the data dependent fit term of eqn. 2.3
    # negative sign for 2nd term comes from $-ln L - ln |J|$
    return gp.lnlikelihood(psi) - np.sum(ln_param[:2])


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


def lnprob_gp(lnHP_truth, kernel, coords, psi, psi_err=1e-10,
              lnprior_vals=[[-10, 10], [-10, 10]]):
    """the log posterior prob that emcee is going to evaluate

    Params:
    -------
        lnHP_truth = tuple of two floats,
            values are log values of the guessed hyperparameters
        kernel = george.kernels obj
        coords = numpy array, feature grid
        psi = numpy array, variable to be predicted
        psi_err = float, uncertainty / gaussian noise at coordinate `coords`
    """
    ln_hp = lnHP_truth[:2]
    lp = lnprior_gp(ln_hp, lnprior_vals=lnprior_vals)

    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(lnHP_truth, kernel, coords, psi, psi_err)


def draw_initial_guesses(initial, guess_dev_frac, ndim, nwalkers):
    return [np.array(initial) +
            guess_dev_frac * np.array(initial) * np.random.randn(ndim)
            for i in xrange(nwalkers)]


def fit_gp(initial, kernel, data, nwalkers=8, guess_dev_frac=1e-6,
           lnprior_vals=[[-10., 10.], [-10., 10]], burnin_chain_len=int(1e3),
           conver_chain_len=int(5e3), a=2.0, threads=1, pool=None):
    """
    Parameters
    ----------
        initial : list / array
            of initial guesses of the truth value of the **log** of hp
        kernel: george.kernels object
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

    Returns
    ------
        sampler : emcee sampler object,
            these are in LOG space
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
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, a=a,
                                    args=(kernel, data[0], data[1], data[2]),
                                    kwargs={"lnprior_vals": lnprior_vals},
                                    threads=threads, pool=pool)

    if burnin_chain_len > 0:
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
    print("the optimized p0 values are \n{0}".format(np.exp(p0)))

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


def optimize_ln_likelihood(gp, ln_p, psi, coords):
    """
    :note: original code obtained from George's documentation
    """
    import scipy.optimize as op

    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        # Update the kernel parameters and compute the likelihood.
        gp.kernel[:] = ln_p
        ll = gp.lnlikelihood(psi, quiet=True)

        # The scipy optimizer doesn't play well with
        # infinities.
        return -ll if np.isfinite(ll) else 1e25

    # And the gradient of the objective function.
    def grad_nll(p):
        # Update the kernel parameters and compute the
        # likelihood.
        gp.kernel[:] = ln_p
        return -gp.grad_lnlikelihood(psi, quiet=True)

    # You need to compute the GP once before
    # starting the optimization.
    gp.compute(coords)

    # Print the initial ln-likelihood.
    print(gp.lnlikelihood(psi))

    # Run the optimization routine.
    p0 = gp.kernel.vector
    results = op.minimize(nll, p0,
                          jac=grad_nll,
                          method="L-BFGS-B")

    # Update the kernel and print the final
    # log-likelihood.
    gp.kernel[:] = results.x
    print(gp.lnlikelihood(y))

    return


def calculate_kernel_properties(data_pt_nos, rng, truth):
    spacing = (rng[1] - rng[0]) / data_pt_nos
    eff_spacing = 1 / data_pt_nos
    exponent = -0.5 * spacing ** 2 * truth[1]
    value_exp = np.exp(exponent)
    char_spacing = 1 / np.sqrt(2. * truth[1])
    print ("-------grid properties------------------")
    print ("spacing = {0:.2e}, ".format(spacing) +
           "spacing^2 = {0:.2e}".format(spacing**2))
    print ("eff spacing = {0:.2e}, ".format(eff_spacing) +
           "eff spacing^2 = {0:.2e}".format(eff_spacing ** 2))

    print ("Exp(-0.5 * {0:.2f} * {1:.2f}) ".format(
            truth[1], spacing ** 2) +
           "= Exp({0:.2e}) = {1:1.2e} ".format(exponent, value_exp))
    print ("{2} * Exp({0:.2f}) = {1:1.2e}".format(exponent,
                                                  truth[0] * value_exp,
                                                  truth[0]))
    print ("\n-------param properties-----------------")
    print ("char spacing = {0:.2e}".format(char_spacing))
    print ("Correlation = exp(-4 * {0}) = {1:.2e}".format(truth[1],
                                                          np.exp(-4 * truth[1])))
    return
