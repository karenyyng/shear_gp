"""contains code for all the diagnostic plots for shear_gp project
"""
import numpy as np
import matplotlib.pyplot as plt
from astrostats import biweightLoc, bcpcl
from astroML import density_estimation as de
from sample_and_fit_gp import char_dim
import pandas as pd

def find_bin_ix(binedges, loc):
    """find the index in the np array binedges that corresponds to loc"""
    find_loc_i = binedges < loc
    return np.sum(find_loc_i)


def comb_zip(ls1, ls2):
    return [(lb1, lb2) for lb1 in ls1 for lb2 in ls2]


def plot_2D_gp_samples(psi_s, coord_grid, figside, truth, range_No,
                       fontsize=15, unit="arbitrary unit"):
    """
    :params
    psi_s = flattened (1D) version of the psi_s data
    coord_grid = 2D np array of floats
    figside = float, in inches how big the figure should be
    truth = tuple of floats that denotes lambDa and rho
    """
    lambDa, rho = truth
    range_No = coord_grid[-1, -1]
    char_length = char_dim(rho)
    color = psi_s
    fig, ax = plt.subplots()
    im = plt.scatter(coord_grid.transpose()[1], coord_grid.transpose()[0],
                     s=35, cmap=plt.cm.jet, c=color)
    #fig.set_figwidth(figside * 1.04)

    fig.set_figheight(figside)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(r"ExpSq kernel: $\lambda=$" +
                 "{0:.2f}, ".format(lambDa) + r"$\rho=$" +
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


def plot_2D_gp_contour(psi_s, coord_grid, figside, truth, range_No,
                       spacing, unit="arbitrary unit", *args):
    """
    :params
    psi_s = flattened (1D) version of the psi_s data
    coord_grid = 2D np array of floats
    figside = float, in inches how big the figure should be
    truth = tuple of floats that denotes lambDa and rho
    """
    lambDa, rho = truth
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


def histplot1d_part(ax, x, prob=None, N_bins='knuth', histrange=None,
                    x_lim=None, y_lim=None):
    '''
    This take the additional value of an array axes. for use with subplots
    similar to histplot1d but for subplot purposes I believe
    '''
    # compare bin width to knuth bin width
    #if type(N_bins) is int:
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
    :params:
        df = dataframe / dictionary / record array
        that contain the data of all the variables to be plots
    space = float, px of space that is added between subplots
    var_list = list of strings - denotes the column header names
        that needs to be plotted
        or if df is not actually a df / dict, use a list of integers
        instead
    axlims = dictionary, keys are the strings in var_list,
        each value is a tuple of (low_lim, up_lim) to denote the limit
        of values to be plotted
    truth = dictionary of floats, each correspond to a relevant entry in
        the df / dict / record array with same dimension as var_list
    Nbins_2D = dictionary, keys are in format of tuples of
        (x_col_str, y_col_str) to denote which subplot you are referring to
    axlabels = dictionary, keys correspond to the variable names
    xlabel_to_rot = dictionary,
        key is the the key for the labels to be rotated,
        value is the degree to be rotated
    histran = dictionary,
        some keys has to be the ones for the plots, value are in
        form of (lowerhist_range, upperhist_range)
    figsize = integer, figuares are squared this refers to the side length
    fontsize = integer, denotes font size of the labels
    save = logical, denotes if plot should be saved or not
    prefix = string, prefix of the output plot file
    path = string, path of the output plot file
    suffix = string, file extension of the output plot file

    Stability: Not entirely tested, use at own risk
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

        if type(truth) == dict:
            for key in truth.keys():
                assert key in var_list, "key {0} in the list ".format(key) + \
                    "of 'truth' value not present in var_list"

    if type(var_list) is dict:
        for var in var_list:
            assert var in data.columns, "variable to be plotted not in df"

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
        #print "setting x and y tick freq for {0}".format((n, n))
        ax2 = axarr[n, n]
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    #print "setting x and y tick freq for {0}".format((i, j))
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

    if type(data) is dict or type(data) is pd.core.frame.DataFrame:
        prob = data["prob"]
    else:
        prob = np.ones(data.shape[1])

    # start plotting the diagonal
    for i in range(N):
        print "N_bins = {0}".format(N_bins[var_list[i]])
        histplot1d_part(axarr[i, i], np.array(data[var_list[i]]),
                        prob,
                        N_bins=N_bins[var_list[i]],
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
        print "saving plot to {0}".format(path + prefix + suffix)
        plt.savefig(path + prefix + suffix, dpi=200, bbox_inches='tight')

    return


def trace_plot(sampler):
    """
    """
    return
