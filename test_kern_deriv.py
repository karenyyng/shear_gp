import george
import numpy as np
from kern_deriv import *
from sample_and_fit_gp import (make_grid, normalize_2D_data)
import matplotlib.pyplot as plt


def test_KappaKappaExpSquare(coords, beta=1.):
    """sanity check"""
    k1 = 1.0 * george.kernels.ExpSquaredKernel(metric=beta, ndim=2)
    print "pars of original kernel are {0}".format(k1)
    gpExpSq = george.GP(k1)

    # by default the kappaKappaExpSquaredKernel has ndim = 2.0
    k = KappaKappaExpSquaredKernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(beta * k)

    gpExpSq.compute(coords, 1e-1)
    gpKKExpSq.compute(coords, 1e-1)

    Cov = gpExpSq.get_matrix(coords)
    KKCov = gpKKExpSq.get_matrix(coords)
    print "gp exp sq kernel gives {0}\n".format(Cov)
    print "\nKKExpSq kernel gives {0}\n".format(KKCov)

    return


def plotDerivCov(kernel, coords, beta=1.):
    coords = np.array([[i, 3] for i in np.arange(0, grid_extent, spacing)])

    k = kernel(beta, coords, ndim=2)
    gpKKExpSq = george.GP(1.0 * k)

    print "------------------------------------------------------------------"
    print "print info about {0}".format(kernel.__name__)
    gpKKExpSq.compute(coords, 1e-5)
    k.plot(spacing=spacing, save=False)
    plt.close()
    print "------------------------------------------------------------------"
    return k.value(coords)


def plotExpSqCov(grid_extent, spacing, plot=False, save=False, beta=1.):
    coords = np.array([[i, 3] for i in np.arange(0, grid_extent, spacing)])

    k = george.kernels.ExpSquaredKernel(1.0, ndim=2)
    gpExpSq = george.GP(beta * k)

    gpExpSq.compute(coords, 1e-5)

    Cov = gpExpSq.get_matrix(coords)
    if plot:
        f, ax = plt.subplots(figsize=(12, 9))
        plt.axes().set_aspect('equal')
        cm = plt.pcolor(Cov, cmap=plt.cm.Blues)  # , vmin=0, vmax=2.5)
        ylim = plt.ylim()
        plt.ylim(ylim[::-1])
        plt.xticks(rotation=45)
        # plt.xticks(np.arange(0, grid_extent, spacing), rotation=45)
        # plt.yticks(np.arange(grid_extent, 0, spacing))
        plt.title("ExpSquaredKernel " +
                  r'for points on a line ' +
                  'with spacing {0}'.format(spacing))
        plt.colorbar(cm)
        plt.savefig("./plots/ExpSqCov.png", bbox_inches='tight')
        plt.show()
        plt.close()


def plotPearsonCorr(Cov, pos_definiteness, betas):
    """
    :param Cov: dictionary of np arrays
    :param pos_definiteness: dictionary of tuples
    ::
    """
    return

if __name__ == "__main__":
    grid_rng = (0., 10.)
    spacing = 1.
    beta = .3

    coords = make_grid(grid_rng, spacing)
    coords = normalize_2D_data(coords)

    plotExpSqCov(coords, plot=True, save=False)

    # coords = np.arange(grid_extent, step=spacing)

    Cov = {}
    print "beta is {0}".format(beta)
    Cov["KappaKappaCov"] = plotDerivCov(KappaKappaExpSquaredKernel,
                                        grid_extent=grid_extent,
                                        spacing=spacing, beta=beta)
    Cov["KappaGamma1Cov"] = plotDerivCov(KappaGamma1ExpSquaredKernel,
                                         grid_extent=grid_extent,
                                         spacing=spacing, beta=beta)
    Cov["KappaGamma2Cov"] = plotDerivCov(KappaGamma2ExpSquaredKernel,
                                         grid_extent=grid_extent,
                                         spacing=spacing, beta=beta)
    Cov["Gamma1Gamma1Cov"] = plotDerivCov(Gamma1Gamma1ExpSquaredKernel,
                                          grid_extent=grid_extent,
                                          spacing=spacing, beta=beta)
    Cov["Gamma1Gamma2Cov"] = plotDerivCov(Gamma1Gamma2ExpSquaredKernel,
                                          grid_extent=grid_extent,
                                          spacing=spacing, beta=beta)
    Cov["Gamma2Gamma2Cov"] = plotDerivCov(Gamma2Gamma2ExpSquaredKernel,
                                          grid_extent=grid_extent,
                                          spacing=spacing, beta=beta)

    pos_definiteness = {k: np.linalg.slogdet(Cov[k]) for k in Cov.keys()}

