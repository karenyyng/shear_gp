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


def make_grid(rng, data_pts, fine_data_pts=None, regular=True):
    """
    :param rng: list / tuple of two floats
        denotes the lower and upper range of the range
    :param spacing: positive float
    :param regular: bool
        determines if regular grid is used or not

    :returns: 2D numpy array
        shape = (n_obs, 2)
    """
    xg = np.linspace(rng[0], rng[1], data_pts)
    if fine_data_pts is None:
        fine_data_pts = data_pts
    yg = np.linspace(rng[0], rng[1], fine_data_pts)
    if regular:
        return np.array([[x, y] for x in xg for y in yg])
    else:
        return np.random.rand(data_pts ** 2, 2) * (rng[1] - rng[0]) - rng[0]


def generate_2D_data(truth, data_pts_no_per_side, kernels, rng=(0., 1.),
                     noise_amp=1e-6, regular_grid=True,
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
        This is added in quadrature to the diagonal of the covariance matrix.
    rng : tuple of two floats, end points of the grid in each dimension
    george_param : bool, whether the parameterization was in the format
        required by george

    Returns
    =======
    coords = 2D numpy array, grid points
    psi = numpy array, GP sample values in 1D
    """
    coords = make_grid(rng, data_pts_no_per_side, regular=regular_grid)

    if white_kernel_as_nugget:
        gp = construct_gp_for_ExpSqlike_and_white_kernels(kernels,
                                                          (truth[0],
                                                           truth[1],
                                                           truth[1],
                                                           noise_amp))
        yerr = 0.

    else:
        gp, yerr = construct_gp_for_ExpSqlike_kernels(kernels, noise_amp)

    # use yerr for adding diagonal noise,
    # yerr is added in quadrature by George implicitly
    gp.compute(coords, yerr=yerr)

    psi = gp.sample(coords)

    mtx = gp.get_matrix(coords)
    if np.linalg.slogdet(mtx)[0]:
        print("Kernel matrix is positive definite.")
        print("Cond # = {0:.2e}".format(np.linalg.cond(mtx)))
    else:
        print("WARNING: Kernel matrix is NOT positive definite.")

    if white_kernel_as_nugget:
        return coords, psi
    else:
        return coords, psi, yerr


def draw_cond_pred(s_param, fine_coords, psi, psi_err, coords):
    """
    this should be sampling from conditional distribution
    with Schur Complement as the Covariance matrix
    """
    gp = george.GP(s_param[0] *
                   kernels.ExpSquaredKernel(s_param[1], ndim=2))
    gp.compute(coords, psi_err)
    return gp.sample_conditional(psi, fine_coords)


# ------- to be tested ------------------------------------

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


def log_prior(param):
    """:to do: check the 1 / sigma prior is indeed the Jeffrey's prior"""
    return 1. / param

# -------- helper functions for calling emcee ---------------


def construct_gp_for_ExpSqlike_and_white_kernels(kernels, hp):
    """
    :param kernels:
    :param hp: list or tuple or array of floats
        i.e. [inv_lambda, l_sq, l_sq, noise_amp]

    :returns: gp object
    """
    ExpSquaredLikeKernel, WhiteKernel = kernels
    # George adds diagonal error term in quadrature
    gp = george.GP(hp[0] * ExpSquaredLikeKernel([hp[1], hp[2]], ndim=2.) +
                   WhiteKernel(pow(hp[3], 2), ndim=2), mean=0.0)

    return gp


def construct_gp_for_ExpSqlike_kernels(kernels, hp, noise_amp):
    gp = george.GP(hp[0] * kernels[0]([hp[1], hp[2]], ndim=2.),
                   mean=0.0)
    psi_err = \
        noise_amp + noise_amp * np.random.randn(data_pt_no_per_side ** 2)
    return gp, psi_err


def lnlike_gp(ln_param, gp, coord, psi, yerr=0.0):
    """ we initialize the lnlike_gp to be the ln likelihood computed by
    george given the data points, this uses original parametrization

    Parameters:
    -----------
    ln_param : list of floats
        expect a format of [ln_hp1, ln_hp2, ln_hp3, p1, p2, ..., pn]
        where the first two hyperparameters for the kernel function for
        George are in log s3cale
    gp : george.gp object,
        gp should be a linear combination of two kernels,
        first have same parameterization as ExpSquaredKernel,
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
    # being in the log scale
    gp.kernel[:] = ln_param

    # compute last 2 terms of marginal log likelihood stated
    # in the Rasmussen GP book eqn. 2.3
    gp.compute(coord, yerr=yerr)

    # this computes the data dependent fit term of eqn. 2.3
    return gp.lnlikelihood(psi)


def ln10_transform_hp_fix_lambda(ln_param):
    """
    This upplies the hyperparameter values to `lnlike_gp`
    :param ln_param: list of floats
        [inv_lambda, log10(l_sq), log10(l_sq), log10(noise_amp)]
    :returns: list of floats
        these are the hyperparameters in correct scale for
        being used by `lnlike_gp`
    """
    # only l_sq and noise_amp
    ln_param = [ln_param[0], pow(10, ln_param[1]), pow(10, ln_param[2]),
                pow(10, ln_param[3]) ** 2]
    return np.log(ln_param)


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


def compute_ln_likelihood_surface(
        inv_lambda, l_sq, noise_amp, kernels,
        data_pt_nos_per_side=10, rng=(0, 1.),
        p0_rng=(0.1, 2.), p0_grid_pts=40,
        p1_rng=(1e-3, 1.), p1_grid_pts=40, ax=None):
    """plots the ln_likelihood surface in the default parametrization of George

    Parameters
    ----------
    inv_lambda : float
        value of inv lambda
    l_sq : float
        value of l_sq
    noise_amp : float
        this value is added **in quadrature** to evaluate the value
        to be added to the diagonal of the kernel matrix
    data_pt_nos : int
        how many data pt (psi) per side to generate
        total no. of data pt = (data_pt_nos) ** 2
    p0_grid_pts : int
        no. of p0 value to compute
        the likelihood surface at
    p1_grid_pts : int
        no. of p1 value to compute the likelihood surface at
    """

    truth = (inv_lambda, l_sq)

    # rng = (0, 1)  # make sure features are normalized ...
    print ("noise_amp = {0:.2e}".format(noise_amp))
    print ("Generating 2D data ...")
    coords, psi = \
        generate_2D_data(truth, data_pt_nos_per_side, kernels=kernels,
                         rng=rng, noise_amp=noise_amp,
                         white_kernel_as_nugget=True)

    # provide mean subtracted data - this is done by George
    # underneath the hood
    psi -= np.mean(psi)
    # psi /= np.std(psi)

    p0_grid = np.linspace(p0_rng[0], p0_rng[1], p0_grid_pts)
    p1_grid = np.linspace(p1_rng[0], p1_rng[1], p1_grid_pts)

    gp = construct_gp_for_ExpSqlike_and_white_kernels(
        kernels, (inv_lambda, p0_grid[0], p0_grid[0], p1_grid[0]))

    # initialize the param space to examine
    print ("Computing likelihood surface ...")
    lnlikelihood_surface = np.array(
        [[lnlike_gp((np.log(inv_lambda), np.log(p0), np.log(p0), np.log(p1)),
                    gp, coords, psi) for p0 in p0_grid] for p1 in p1_grid])

    return p0_grid, p1_grid, lnlikelihood_surface


def compute_log10_transformed_ln_likelihood_surface(
        inv_lambda, l_sq, noise_amp, kernels,
        data_pt_nos_per_side=10, rng=(0, 1.),
        p0_rng=(0.1, .2), p0_grid_pts=10,
        p1_rng=(1e-3, 1.), p1_grid_pts=10, ax=None, verbose=False,
        regular_grid=False):
    """plots the ln_likelihood surface in the default parametrization of George

    Parameters
    ----------
    inv_lambda : float
        value of inv lambda
    l_sq : float
        value of l_sq
    noise_amp : float
        this value is added **in quadrature** to evaluate the value
        to be added to the diagonal of the kernel matrix
    data_pt_nos : int
        how many data pt (psi) per side to generate
        total no. of data pt = (data_pt_nos) ** 2
    p0_grid_pts : int
        no. of p0 value to compute the likelihood surface at
    p1_grid_pts : int
        no. of p1 value to compute the likelihood surface at
    """
    truth = (inv_lambda, l_sq)

    # rng = (0, 1)  # make sure features are normalized ...
    print ("noise_amp = {0:.2e}".format(noise_amp))
    print ("Generating 2D data ...")
    coords, psi = \
        generate_2D_data(truth, data_pt_nos_per_side, kernels=kernels,
                         rng=rng, noise_amp=noise_amp,
                         white_kernel_as_nugget=True,
                         regular_grid=regular_grid)

    # linear in the log space!
    p0_grid = np.linspace(np.log10(p0_rng[0]), np.log10(p0_rng[1]),
                          p0_grid_pts)
    p1_grid = np.linspace(np.log10(p1_rng[0]), np.log10(p1_rng[1]),
                          p1_grid_pts)

    if verbose:
        print (p0_grid)
        print (p1_grid)

    gp = construct_gp_for_ExpSqlike_and_white_kernels(
        kernels,
        ln10_transform_hp_fix_lambda((inv_lambda, p0_grid[0],
                                     p0_grid[0], p1_grid[0]))
    )

    # initialize the param space to examine
    print ("Computing likelihood surface ...")
    lnlikelihood_surface = np.array([[
        lnlike_gp(ln10_transform_hp_fix_lambda((inv_lambda, p0, p0, p1)),
                  gp, coords, psi)
        for p0 in p0_grid] for p1 in p1_grid])

    return p0_grid, p1_grid, lnlikelihood_surface


def sampler_acceptance_check(sampler):
    if np.any(sampler.acceptance_fraction < 0.2):
        error_msg = "Initial guesses may be bad / model may be bad \n" + \
                    "Acceptance rate is < 0.2, currently at \n" + \
                    str(sampler.acceptance_fraction)
        raise ValueError(error_msg)
    return None


def Rubin_Gelman_Rsq_score():
    return


def calculate_kernel_properties(data_pt_nos, rng, truth):
    spacing = (rng[1] - rng[0]) / data_pt_nos
    eff_spacing = 1 / data_pt_nos
    exponent = -0.5 * spacing ** 2 / truth[1]
    value_exp = np.exp(exponent)
    char_spacing = np.sqrt(truth[1])
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
    print ("\n----------------------------------------")
    return

# ----------- optimization / initialization routines -------------


def optimize_likelihood_in_log10_space(initial_guess, dep_var, features,
                                       kernels, verbose=True):
    """A Scipy L-BFGS-B optimizer for George kernels
    I modified the built-in optimization function of `George`
    to get this

    :param initial_guess: list / tuple / array of floats
        in format of [inv_lambda, l_sq, l_sq, noise_amp ** 2.] in original
        scale
    :param dep_var: list / array of floats
        len(dep_var) = len(features)
    :param features: list / array of floats

    :return: george.gp object with the optimized parameters
    """
    import scipy.optimize as op
    assert len(initial_guess) == 4, \
        "Initial_guess should be in format of \n" + \
        "[inv_lambda, l_sq, l_sq, noise_amp]"

    assert initial_guess[1] == initial_guess[2], \
        "Two values for l_sq have to be the same"

    def negative_ln_likelihood_in_log10_space(log10_param, verbose=False):
        """
        Define the objective function (negative log-likelihood in this case).
        """
        # Update the kernel parameters and compute the likelihood.
        # kernel vector (of param) is in log scale
        # One last transformation to log space is due to how George stores
        # params in vectors.
        hp = np.log(
            [pow(10, i) for i in log10_param]
        )
        gp.kernel[:] = [hp[0], hp[1], hp[1], hp[2]]

        ll = gp.lnlikelihood(dep_var, quiet=True)
        if verbose:
            print ("New params : ", np.exp(gp.kernel.vector))
            print ("New lnlikelihood : ", gp.lnlikelihood(dep_var))

        # The scipy optimizer doesn't play well with infinities.
        return -ll if np.isfinite(ll) else 1e25

    def grad_negative_ln_likelihood_in_log10_space(log10_param):
        # And the gradient of the objective function.
        # Update the kernel parameters in log10 space and
        # compute the likelihood.
        # One last transformation to log space is due to how George stores
        # params in vectors.
        hp = np.log(
            [pow(10, i) for i in log10_param]
        )
        gp.kernel[:] = [hp[0], hp[1], hp[1], hp[2]]
        grad_ll = -gp.grad_lnlikelihood(dep_var, quiet=True)
        if ~np.isfinite(grad_ll):
            print ("Infinite gradient of log likelihood")
        return grad_ll

    gp = construct_gp_for_ExpSqlike_and_white_kernels(
        kernels, initial_guess)

    # You need to compute the GP once before starting the optimization.
    gp.compute(features)

    # Print the initial ln-likelihood.
    original_ll = gp.lnlikelihood(dep_var)

    guess = \
        (initial_guess[0], initial_guess[1],
         initial_guess[3] ** 2)
    rand = np.random.rand(3) * 5e-1
    guess = np.log10(guess + np.array([rand[0], rand[1], rand[2]]))
    if verbose:
        print ("Initial guess :", [pow(10, i) for i in guess])

    # Run the optimization routine in log10 space.
    results = op.minimize(negative_ln_likelihood_in_log10_space,
                          x0=guess, method="L-BFGS-B")  # ,
                          # jac=grad_negative_ln_likelihood_in_log10_space)

    # Update the kernel and print the final log-likelihood.
    hp = np.log(
        [pow(10, i) for i in results.x]
    )
    gp.kernel[:] = [hp[0], hp[1], hp[1], hp[2]]

    final_param = np.exp([hp[0], hp[1], hp[1], hp[2]])
    if verbose:
        print("Optimized lnlikelihood : ", gp.lnlikelihood(dep_var))

        print("\nInitial lnlikelihood : ", original_ll)
        print ("Optimized param : ", final_param)
    return gp
