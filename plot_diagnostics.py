"""contains code for all the diagnostic plots for shear_gp project
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
#from astrostats import biweightLoc, bcpcl
from astroML import density_estimation as de
from sample_and_fit_gp import char_dim
import sample_and_fit_gp as fit
import pandas as pd
from scipy.special import erf
from scipy.stats import norm


def find_bin_ix(binedges, loc):
    """find the index in the np array binedges that corresponds to loc"""
    find_loc_i = binedges < loc
    return np.sum(find_loc_i)


def comb_zip(ls1, ls2):
    return [(lb1, lb2) for lb1 in ls1 for lb2 in ls2]


def histplot1d_part(ax, x, prob=None, N_bins='knuth', histrange=None,
                    x_lim=None, y_lim=None):
    '''
    This take the additional value of an array axes. for use with subplots
    similar to histplot1d but for subplot purposes I believe
    '''
    # compare bin width to knuth bin width
    # if type(N_bins) is int:
    #    print "specified bin width is {0}, Knuth bin size is {1}".format(
    #        N_bins, knuth_N_bins)
    if N_bins == 'knuth':
        binwidth, bins = de.knuth_bin_width(x, return_bins=True)
        knuth_N_bins = bins.size - 1
        N_bins = knuth_N_bins

    hist, binedges, tmp = ax.hist(
        x, bins=N_bins, histtype='step', weights=prob, range=histrange,
        color='k', linewidth=1)

    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in np.arange(N_bins):
        if i == 0:
            x_binned = \
                np.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        elif np.size(x_binned) == 0:
            x_binned = \
                np.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        else:
            x_temp = \
                np.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
            x_binned = np.concatenate((x_binned, x_temp))
    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc, x_binned, 1)
    ll_95, ul_95 = bcpcl(loc, x_binned, 2)

    # Create location and confidence interval line plots
    # find the binedge that the location falls into
    # so that the line indicating the location only extends to top of
    # histogram
    loc_ix = find_bin_ix(binedges, loc)
    ll_68_ix = find_bin_ix(binedges, ll_68)
    ul_68_ix = find_bin_ix(binedges, ul_68)
    ll_95_ix = find_bin_ix(binedges, ll_95)
    ul_95_ix = find_bin_ix(binedges, ul_95)

    ax.plot((loc, loc), (0, hist[loc_ix - 1]), ls='--', lw=1, color="k")

    width = binedges[ll_68_ix + 1] - binedges[ll_68_ix]
    for i in range(ll_68_ix, ul_68_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.6)
    for i in range(ll_95_ix, ul_95_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.3)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return loc, ll_68, ul_68, ll_95, ul_95


def histplot2d_part(ax, x, y, prob=None, N_bins=100, histrange=None,
                    x_lim=None, y_lim=None):
    '''
    similar to histplot2d
    This take the additional value of an array axes. for use with subplots
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)]
        the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    # prevent masked array from choking up the 2d histogram function
    x = np.array(x)
    y = np.array(y)

    # Create the confidence interval plot
    assert prob is not None, "there is no prob given for weighting"

    if histrange is None:
        if prob is not None:
            H, xedges, yedges = \
                np.histogram2d(x, y, bins=N_bins, weights=prob)
        elif prob is None:
            H, xedges, yedges = np.histogram2d(x, y, bins=N_bins)
    else:
        if prob is not None:
            H, xedges, yedges = \
                np.histogram2d(x, y, bins=N_bins,
                               range=[[histrange[0], histrange[1]],
                                      [histrange[2], histrange[3]]],
                               weights=prob)
        elif prob is None:
            H, xedges, yedges = np.histogram2d(
                x, y, bins=N_bins, range=[[histrange[0], histrange[1]],
                                          [histrange[2], histrange[3]]])
    H = np.transpose(H)
    # Flatten H
    h = np.reshape(H, (N_bins ** 2))
    # Sort h from smallest to largest
    index = np.argsort(h)
    h = h[index]
    h_sum = np.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in np.arange(np.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum / h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum / h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]

    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    y = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    X, Y = np.meshgrid(x, y)

    # can use pcolor or imshow to show the shading instead
    ax.pcolormesh(X, Y, H, cmap=plt.cm.gray_r, shading='gouraud')
    ax.contour(X, Y, H, (h_2sigma, h_1sigma), linewidths=(2, 2),
               colors=((158 / 255., 202 / 255., 225 / 255.),
                       (49 / 255., 130 / 255., 189 / 255.)))

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)


def N_by_N_lower_triangle_plot(data, space, var_list, axlims=None,
                               truth=None, Nbins_2D=None, axlabels=None,
                               N_bins=None, xlabel_to_rot=None, histran=None,
                               figsize=6, fontsize=12, save=False, prefix=None,
                               suffix=".png", path="./"):
    """ create a N by N matrix of plots
    with the top plot of each row showing a density plot in 1D
    and the remaining plots being 2D contour plots

    Parameters:
    -----------
        data: numpy array / emcee.sampler.flatchain.transpose()
            that contain the posterior values of all the variables to be
            plotted, or dataframe of values
        space = float, px of space that is added between subplots
        var_list = list of integers / str - denotes the column header names
            that needs to be plotted
            or if data is not actually a df / dict with keys,
            use a list of integers instead
        axlims = dictionary, keys are the strings in var_list,
            each value is a tuple of (low_lim, up_lim) to denote the limit
            of values to be plotted
        truth = dictionary of floats, each correspond to a relevant entry in
            the df / dict / record array with same dimension as var_list
        Nbins_2D = dictionary, keys are in format of tuples of
            (x_col_str, y_col_str) to denote which subplot you are referring to
        axlabels = dictionary, keys correspond to the variable names
        xlabel_to_rot = array / dictionary,
            put array of angle degrees to be rotated if array was supplied as
            data
            if dictionary / dataframe is supplied as the data
            key is the the key for the labels to be rotated,
            value is the degree to be rotated
        histran = dictionary,
            some keys has to be the ones for the plots, value are in
            form of (lowerhist_range, upperhist_range)
        figsize = integer, figuares are squared this refers to the side length
        fontsize = integer, denotes font size of the labels
        save = boolean, denotes if plot should be saved or not
        prefix = string, prefix of the output plot file
        path = string, path of the output plot file
        suffix = string, file extension of the output plot file

    Returns:
    -------
        estimates: dict of estimates,
            each list of estiamte is in the form of
            [loc, ll_68, ul_68, ll_95, ul_95]

    stability: works but use at own risk
    """
    from matplotlib.ticker import MaxNLocator

    # begin checking if inputs make sense
    N = len(var_list)
    assert N <= len(axlabels), "length of axlabels is wrong"
    assert N >= 2, "lower triangular contour plots require more than 2\
        variables in the data"

    if truth is not None:
        assert len(truth) == N, "length of 'truth' values of variables " + \
            " needs to match the number of variables in var_list"

        if isinstance(truth, dict):
            for key in truth.keys():
                assert key in var_list, "key {0} in the list ".format(key) + \
                    "of 'truth' value not present in var_list"

    if isinstance(var_list, dict):
        for var in var_list:
            assert var in data.columns, "Variable: {0} to ".format(var) + \
                " be plotted not in df"

    if axlabels is None:
        axlabels = {key: key for key in var_list}

    if xlabel_to_rot is None:
        xlabel_to_rot = {key: 0 for key in var_list}

    if histran is None:
        histran = {key: None for key in var_list}

    if axlims is None:
        axlims = {key: (None, None) for key in var_list}

    if Nbins_2D is None:
        keys = comb_zip(var_list, var_list)
        Nbins_2D = {key: 50 for key in keys}

    if N_bins is None:
        N_bins = {key: 'knuth' for key in var_list}

    if save:
        assert prefix is not None, "prefix for output file cannot be none"

    # impossible for the matrix plot not to be squared in terms of dimensions
    # set each of the subplot to be squared with the figsize option
    f, axarr = plt.subplots(N, N, figsize=(figsize, figsize), sharex='col')
    f.subplots_adjust(wspace=space, hspace=space)

    # remove unwanted plots on the upper right
    plt.setp([a.get_axes() for i in range(N - 1)
              for a in axarr[i, i + 1:]], visible=False)

    # remove unwanted row axes tick labels
    plt.setp([a.get_xticklabels() for i in range(N - 1)
              for a in axarr[i, :]], visible=False)

    # remove unwanted column axes tick labels
    plt.setp([axarr[0, 0].get_yticklabels()], visible=False)
    plt.setp([a.get_yticklabels() for i in range(N - 1)
              for a in axarr[i + 1, 1:]], visible=False)

    # create axes labels
    if axlabels is not None:
        for j in range(1, N):
            axarr[j, 0].set_ylabel(axlabels[var_list[j]], fontsize=fontsize)
        for i in range(N):
            axarr[N - 1, i].set_xlabel(axlabels[var_list[i]],
                                       fontsize=fontsize)

    for n in range(N):
        # avoid overlapping lowest and highest ticks mark
        # print "setting x and y tick freq for {0}".format((n, n))
        ax2 = axarr[n, n]
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # print "setting x and y tick freq for {0}".format((i, j))
    for i in range(N):
        for j in range(N):  # range(i)
            ax2 = axarr[i, j]
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # rotate the xlabels appropriately
    if xlabel_to_rot is not None:
        match_ix = [var_list.index(item) for item in var_list]
        # ok to use for-loops for small number of iterations
        for ix in match_ix:
            labels = axarr[N - 1, ix].get_xticklabels()
            for label in labels:
                label.set_rotation(xlabel_to_rot[var_list[ix]])

    if isinstance(data, dict) or isinstance(data, pd.core.frame.DataFrame):
        prob = data["prob"]
    else:
        prob = np.ones(data.shape[1])

    # start plotting the diagonal and storing location and confidence level
    # estimates
    est = {}
    for i in range(N):
        print("N_bins = {0}".format(N_bins[var_list[i]]))
        est[var_list[i]] = histplot1d_part(axarr[i, i],
                                           np.array(data[var_list[i]]),
                                           prob, N_bins=N_bins[var_list[i]],
                                           histrange=histran[var_list[i]],
                                           x_lim=axlims[var_list[i]])

    # start plotting the lower triangle when row no > col no
    for i in range(N):
        for j in range(i):
            histplot2d_part(axarr[i, j], np.array(data[var_list[j]]),
                            np.array(data[var_list[i]]),
                            prob=prob,
                            N_bins=Nbins_2D[(var_list[j], var_list[i])],
                            x_lim=axlims[var_list[j]],
                            y_lim=axlims[var_list[i]])

    # start adding the truth label lines to the contours
    if truth is not None:
        for i in range(N):
            for j in range(i):
                # should plot the truth value of the x-axis
                axarr[i, j].axvline(truth[var_list[j]], color='r')
                # should plot the truth value of the y-axis
                axarr[i, j].axhline(truth[var_list[i]], color='r')

    if save:
        print("saving plot to {0}".format(path + prefix + suffix))
        plt.savefig(path + prefix + suffix, dpi=200, bbox_inches='tight')

    return est


def trace_plot(sampler, labels, truth=None, fontsize=14, chain_no=0):
    """visualize the MCMC steps to eyeball if burn in is sufficient / if
    convergence is achieved
    :params sampler: sampler object from emcee
    :params labels: list of strings, same length as the number of variables
    :params truth: list of floats,

    :stability: works
    :note: thinking of having different trace plots for each chain,
       right now we just glue the chains together, if the burn-in was
       insufficient, we 'd see some weird discontinuities
    """
    varNo = sampler.flatchain.shape[1]
    assert len(labels) >= varNo, \
        "list of labels should be of the same length or more as the \n" + \
        "number of variables"

    f, ax = plt.subplots(nrows=varNo, sharex=True)
    ax[0].set_title("acceptance percent is {0:.2f}%".format(
        np.mean(sampler.acceptance_fraction) * 100.), size=fontsize)

    # plot traceplot of each variable one by one
    for i in range(varNo):
        # ax[i].plot(sampler.flatchain[:, i], alpha=0.5, color='grey')
        ax[i].plot(sampler.chain[chain_no, :, i], alpha=0.5, color='grey')
        ax[i].set_ylabel(labels[i], size=fontsize)

        # add horizontal line to show truth value if available
        if truth is not None:
            ax[i].axhline(truth[i], color='r', label="true value")
            ax[i].legend(loc='best')

    ax[-1].set_xlabel('MCMC step number', size=fontsize)

    return None


def bcpcl(T, T_p, N_sigma):
    '''
    Author: W. A. Dawson
    Calculates the bias corrected percent confidence limits.
    -- Suppose that we have observed data (y1, y2, ..., yn) and use it to
    estimate a population parameter Q (e.g. Q could be the true mean of the
    entire population).
    -- T is a statistic that estimates Q. For example T could be an estimate
    of the true mean by calculating the mean of  (y1, y2, ..., yn).
    -- Suppose that we create m bootstrap samples (y_p_1j, y_p_2j, ...,j_p_nj)
    from observed sample  (y1, y2, ..., yn), where j is the jth bootstrap sample.
    -- Then T_p_j is the jth bootstrap observation of T.  For example this
    could be the mean of (y_p_1j, y_p_2j, ...,j_p_nj).

    T = [float] e.g. biweight Location for (y1, y2, ..., yn)
    T_p = [vector array] biwieght Locations for the bootstrap samples
    N_sigma = the number of sigma to report the confidence limits for
        e.g. for 95% confidence limits N_sigma=2
    Return (lower, upper) confidence limits
    '''

    assert len(T_p) > 0, "length of bootstrapped stat needs to be > 0" + \
        "if bootstrapped stat is computed from histogram, check if " + \
        "histogram values are > 1"

    # Number of bootstrap samples
    m = np.size(T_p)
    # Percentile confidence interval is defined as 100%(1-a), thus for 1sigma
    # a=0.32
    a = 1 - erf(N_sigma / np.sqrt(2))
    # order the bootstrap sample values smallest to largest
    index = np.argsort(T_p)
    T_p = T_p[index]
    # Calculate the bias correction term
    mask = T_p < T
    z_0 = norm.ppf(np.sum(mask) / m)
    # Calculate the a1 and a2 values
    a1 = norm.cdf(2 * z_0 + norm.ppf(a / 2))
    a2 = norm.cdf(2 * z_0 + norm.ppf(1 - a / 2))
    # Calculate the lower and upper indicies of lower and upper confidence
    # intervals
    id_L = np.int(m * a1) - 1
    id_U = np.int(m * a2)
    # Find the lower an upper confidence values
    T_L = T_p[id_L]
    T_U = T_p[id_U]
    return T_L, T_U


def biweightLoc(z, c=6):
    '''Biweight statistic Location (similar to the mean) eqn 5
    Author: W. A. Dawson
    '''
    M = np.median(z)
    MAD = np.median(np.abs(z - M))
    if MAD == 0:
        raise ZeroDivisionError
    u = (z - M) / (c * MAD)
    mask_u = np.abs(u) < 1
    z = z[mask_u]
    u = u[mask_u]
    Cbi = M + np.inner(z - M, (1 - u ** 2) ** 2) / \
        np.sum((1 - u ** 2) ** 2)
    return Cbi


def plot_log10_transformed_ln_likelihood_surface(
        truth, p0_grid, p1_grid, lnlikelihood_surface, kernel_name,
        data_pt_per_side, xlabel, ylabel, ax=None):
    """
    Parameters
    ----------
    p0: float
        value of a
    p1: float
        value of b
    noise_amp : float
        this value is added **in quadrature** to evaluate the value
        to be added to the diagonal of the kernel matrix
    data_pt_per_side : int
        how many data pt (psi) per side to generate
        total no. of data pt = (data_pt_per_side) ** 2
    p0_grid_pts : int
        no. of `a` value to compute
        the likelihood surface at
    p1_grid_pts : int
        no. of `b` value to compute the likelihood surface at
    """
    from matplotlib.colors import LogNorm
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # make all likelihood positive
    # lnlikelihood_surface -= np.min(lnlikelihood_surface)

    print ("Plotting likelihood surface ...")
    lvls = list([-1e-5]) + \
        list([np.max(lnlikelihood_surface) * 0.1 * i for i in range(11)])
    cs = ax.contourf(p0_grid, p1_grid, lnlikelihood_surface,
                     levels=lvls)   # , norm=LogNorm())
    ax.axvline(np.log10(truth[0]), color='r', lw=2, label='truth')
    ax.axhline(np.log10(truth[1]), color='r', lw=2)
    ax.set_title("lnlikelihood surface for {0}\n".format(kernel_name) +
                 r"data_pt_per_side ={0}, ".format(data_pt_per_side) +
                 r"no. of $a$={0}, ".format(len(p1_grid)) +
                 r"no. of $b$" + r"={0}".format(len(p0_grid)),
                 fontsize=14)
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    # print("Finished updating")
    return lnlikelihood_surface


def plot_ln_likelihood_surface(p0_grid, p1_grid, lnlikelihood_surface,
                               true_p0, true_p1, kernel_name, p0_label,
                               p1_label, data_pts_no_per_side, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    print ("Plotting likelihood surface ...")

    lvls = list([-1e-5]) + \
        list([np.max(lnlikelihood_surface) * 0.1 * i for i in range(11)])
    cs = ax.contourf(p0_grid, p1_grid, lnlikelihood_surface, levels=lvls)

    ax.axvline(true_p0, color='r', lw=2, label='truth')
    ax.axhline(true_p1, color='r', lw=2)

    ax.set_title("lnlikelihood surface for {0}\n".format(kernel_name) +
                 r"data_pt_no ={0}, ".format(data_pts_no_per_side) +
                 r"no. of $l^2 $={0}, ".format(len(p0_grid)) +
                 r"no. of $\lambda^{-1}$" +
                 r"={0}".format(len(p1_grid)),
                 fontsize=14)
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(p0_label)
    ax.set_ylabel(p1_label)
    ax.legend(frameon=True)
    return


def plot_2D_gp_samples(psi_s, coord_grid, figside, range_No, kernel_name,
                       truth=None,
                       fontsize=15, unit="arbitrary unit",
                       truth_label=[r"$\lambda^{-1}$", r"$\beta$"], ax=None):
    """
    params:
        psi_s (numpy array): flattened (1D) version of the psi_s data
        coord_grid (2D np array of floats):
        figside = float, in inches how big the figure should be
        truth = tuple of floats that denotes lambDa and rho
    """
    fig, ax = plt.subplots()
    range_No = coord_grid[-1, -1]
    color = psi_s
    im = plt.scatter(coord_grid.transpose()[1], coord_grid.transpose()[0],
                     s=35, cmap=plt.cm.gist_heat, c=color, linewidths=0.3)
    #fig.set_figwidth(figside * 1.04)

    fig.set_figheight(figside)
    fig.colorbar(im, ax=ax, fraction=0.04)

    if truth is not None:
        if len(truth) == 2:
            lambDa, rho = truth
        elif len(truth) == 3:
            lambDa, rho, _ = truth

        char_length = char_dim(rho)
        ax.set_title(r"{1} kernel: {0} =".format(truth_label[0], kernel_name) +
                     "{0:.2f}, ".format(lambDa) + truth_label[1] + "=" +
                     "{0:.2f},".format(rho) +
                     r" $ l=$" + "{0:.2f}".format(char_length),
                     fontsize=fontsize)

    ax.set_xlabel("{0} ({1} {0} per side)".format(unit, range_No),
                  fontsize=fontsize)
    ax.set_ylabel("{0} ({1} {0} per side)".format(unit, range_No),
                  fontsize=fontsize)
    spacing = coord_grid[1, 1] - coord_grid[0, 0]
    ax.set_xlim(coord_grid[0, 0] - spacing, coord_grid[-1, -1] + spacing)
    ax.set_ylim(coord_grid[0, 0] - spacing, coord_grid[-1, -1] + spacing)
    plt.show()


def plot_2D_gp_sample_contour(
        psi_s, coord_grid, figside,
        truth=None, unit="arbitrary unit", kernel_name="ExpSq",
        ax=None):
    """
    :params
    psi_s = flattened (1D) version of the psi_s data
    coord_grid = 2D np array of floats, needs to be coords
        of a regular grid
    figside = float, in inches how big the figure should be
    truth = tuple of floats that denotes lambDa and rho
    """
    # xg = np.arange(0, range_No, spacing)
    xg = np.unique(coord_grid[:, 0])
    yg = np.unique(coord_grid[:, 1])

    psi_s = psi_s.reshape(len(xg), len(yg))

    if ax is None:
        fig = plt.figure()
        fig.set_figheight(figside)
        fig.set_figwidth(figside)
        ax = fig.add_subplot(111, aspect='equal')

    # both contour and contourf function need to be transposed before use
    im = ax.contourf(xg, yg, psi_s)
    if truth is not None:
        lambDa, rho = truth
        ax.set_title(r"{0} kernel: $\lambda=$".format(kernel_name) +
                     "{0}, ".format(lambDa) + r"$l^2=$" +
                     "{0:.2f},".format(rho),
                     # + r" $ l=$" + "{0:.2f}".format(char_length),
                     fontsize=20)
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.show()
    return


# -- helper functions to visualize the conditional predictive check surface---

def posterior_predictive_surface_vs_truth():
    return


def posterior_predictive_surface_from_MCMC_chains(
        sampler, rng, spacing, fine_spacing=1.):
    samples = sampler.flatchain
    fine_coords = fit.make_grid(rng, fine_spacing)
    # use 10 realizations from the conditional distribution
    ix = [np.random.randint(len(samples)) for i in range(10)]
    cond_preds = map(lambda x:
                     fit.draw_cond_pred(samples[x], fine_coords,
                                        psi, psi_err, coords),
                     ix)

    x_coord = range(int(rng[0]), int(rng[1]), int(spacing))
    fine_x_coord = range(int(rng[0]), int(rng[1]), int(fine_spacing))
    fine_y_coord = fine_x_coord

    psi_2D = psi.reshape(rng[1] / spacing, rng[1] / spacing)
    psi_err_2D = \
        psi_err.reshape(rng[1] / spacing, rng[1] / spacing)

    cond_preds_2D = \
        [pred.reshape(rng[1] / fine_spacing, rng[1] / fine_spacing) for
         pred in cond_preds]
    return cond_preds_2D, psi_err_2D, psi_2D


def change_x_ix(x_ix):
    plt.title("examining column {0}".format(x_ix))
    plt.errorbar(y_coord, psi_2D.transpose()[x_ix] ,
                 yerr=psi_err_2D[:][x_ix],
             marker='.', ls="None", color='k')
    fine_x_ix = int(spacing / fine_spacing * x_ix)
    for i in range(10):
        plt.plot(fine_y_coord, cond_preds_2D[i].transpose()[fine_x_ix],
                 color='b', alpha=0.2)
    plt.ylabel(r"$\psi$", size=14)
    plt.xlabel("y", size=14)
    plt.xlim(rng[0]-spacing, rng[1])
    return


def change_y_ix(y_ix):
    plt.title("examining row {0}".format(y_ix))
    plt.errorbar(x_coord, psi_2D[y_ix][:] , yerr=psi_err_2D[y_ix][:],
             marker='.', ls="None", color='k')
    fine_y_ix = int(spacing / fine_spacing * y_ix)
    for i in range(10):
        plt.plot(fine_x_coord, cond_preds_2D[i][fine_y_ix][:],
                 color='b', alpha=0.2)
    plt.ylabel(r"$\psi$", size=16)
    plt.xlabel("x", size=14)
    plt.xlim(rng[0]-spacing, rng[1])
    return
