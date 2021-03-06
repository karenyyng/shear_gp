from __future__ import division, print_function
import george
import numpy as np
from kern_deriv import *
from sample_and_fit_gp import (make_grid)
import matplotlib.pyplot as plt


def run_KappaKappaExpSquare(coords, beta=1.):
    """sanity check"""
    k1 = 1.0 * george.kernels.ExpSquaredKernel(metric=beta, ndim=2)
    print ("pars of original kernel are {0}".format(k1))
    gpExpSq = george.GP(k1)

    # by default the kappaKappaExpSquaredKernel has ndim = 2.0
    k = KappaKappaExpSquaredKernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(beta * k)

    gpExpSq.compute(coords, 1e-1)
    gpKKExpSq.compute(coords, 1e-1)

    Cov = gpExpSq.get_matrix(coords)
    KKCov = gpKKExpSq.get_matrix(coords)
    print ("gp exp sq kernel gives {0}\n".format(Cov))
    print ("\nKKExpSq kernel gives {0}\n".format(KKCov))

    return


def plotDerivCov(kernel, coords, beta=1., plot=False, debug=False):
    """
    :param kernel: python kernel wrapper object
        from `ker_deriv.py`
    :param coords: numpy array of floats
        shape = (nobs, 2)
    :param beta: float

    :return: np array
        kernel_matrix
    """

    k = kernel(beta, coords, ndim=2)
    gpKKExpSq = george.GP(1.0 * k)

    if debug:
        print ("-----------------------------------------------------")
        print ("print info about {0}".format(kernel.__name__))
    gpKKExpSq.compute(coords, 1e-5)
    if plot:
        k.plot1(spacing=spacing, save=False)
    plt.close()
    if debug:
        print ("----------------------------------------------------")
    return k.value(coords)


def get_Cython_kernels(k, coords, beta=1):
    gp = george.GP(k(beta, coords, ndim=2L))
    return gp.get_matrix(coords)


def plotExpSqCov(coords, plot1=False, plot2=True, save=False, l_sq=1.,
                 lambDa=1.):
    """
    :param kernel: python kernel wrapper object
        from `ker_deriv.py`
    :param coords: numpy array of floats
        shape = (nobs, 2)
    :param l_sq: float
        this really is `l_sq` since we did not modify George

    :return: np array
        cov matrix
    """

    k = george.kernels.ExpSquaredKernel(l_sq, ndim=2)
    gpExpSq = george.GP(lambDa * k)

    gpExpSq.compute(coords, 1e-5)

    Cov = gpExpSq.get_matrix(coords)
    if plot1:
        plotCovMatrix(Cov, kernel_name="ExpSquaredKernel")

    if plot2:
        plotFixedCov(Cov, l_sq, coords)
    return Cov


def plotCovMatrix(Cov, kernel_name):
    f, ax = plt.subplots(figsize=(12, 9))
    plt.axes().set_aspect('equal')
    cm = plt.pcolor(Cov, cmap=plt.cm.Blues)  # , vmin=0, vmax=2.5)
    ylim = plt.ylim()
    plt.ylim(ylim[::-1])
    plt.xticks(rotation=45)
    plt.title(kernel_name +
              r'for points on a line ' +
              'with spacing {0}'.format(spacing))
    plt.colorbar(cm)
    if save:
        plt.savefig("./plots/" + kernel_name + ".png", bbox_inches='tight')
    plt.show()
    plt.close()


def plotFixedCov(Cov, beta, coords):
    """
    :param Cov: dictionary of np arrays
    :param pos_definiteness: dictionary of tuples
    :param features: np array, should be 2D array of floats
    """
    i = range(Cov.shape[0])
    fixIx = 4
    CovArray = [Cov[ix, fixIx] for ix in i]

    pos_def = np.linalg.slogdet(Cov)
    if pos_def[0] == 1.0:
        color = (0, 0, beta)
    else:
        color = (beta, 0., 0.)

    plt.plot(i, CovArray,
             label=r"$\beta = ${0:.2f}, ".format(beta) +
             r"$\rho$ = " + "{0:.2f}".format(normalized_corr(beta, coords)),
             color=color)
    plt.xlabel("i")
    plt.ylabel("Cov[i, {0}] value".format(fixIx))

    return


if __name__ == "__main__":
    # grid_rng = (0., 10.)
    # spacing = 1.

    # should reverse calculate a reasonable range
    betas = np.arange(0.1, 1.1, 0.1)

    coords = np.array([[1., i] for i in np.linspace(0, 1, 10)])
    # coords = make_grid(grid_rng, spacing)
    # coords = normalize_2D_data(coords)
    # coords = np.arange(grid_extent, step=spacing)

    Cov = {}


    Cov["ExpSquaredKernel"] = [plotExpSqCov(coords, plot2=True, save=False,
                                            beta=beta)
                               for beta in betas]
    Cov["KappaKappa"] = [plotDerivCov(KappaKappaExpSquaredKernel,
                                      coords, beta=beta)
                         for beta in betas]
    Cov["KappaGamma1"] = [plotDerivCov(KappaGamma1ExpSquaredKernel,
                                       coords, beta=beta)
                          for beta in betas]
    Cov["KappaGamma2"] = [plotDerivCov(KappaGamma2ExpSquaredKernel,
                                       coords, beta=beta)
                          for beta in betas]
    Cov["Gamma1Gamma1"] = [plotDerivCov(Gamma1Gamma1ExpSquaredKernel,
                                        coords, beta=beta)
                           for beta in betas]
    Cov["Gamma1Gamma2"] = [plotDerivCov(Gamma1Gamma2ExpSquaredKernel,
                                        coords, beta=beta)
                           for beta in betas]
    Cov["Gamma2Gamma2"] = [plotDerivCov(Gamma2Gamma2ExpSquaredKernel,
                                        coords, beta=beta)
                           for beta in betas]

    for k in Cov.keys():
        for i in range(len(betas)):
            plotFixedCov(Cov[k][i], betas[i], coords)
            plt.legend(loc='best', frameon=False, fontsize=10)
        plt.title(k)
        plt.savefig("./plots/" + k + "_Cov.png", bbox_inches='tight')
        plt.close()



