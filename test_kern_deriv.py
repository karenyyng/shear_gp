import george
import numpy as np
from kern_deriv import *
import matplotlib.pyplot as plt


def test_KappaKappaExpSquare(grid_extent=5, spacing=1):
    """sanity check"""
    k1 = 1.0 * george.kernels.ExpSquaredKernel(metric=1.0, ndim=2)
    print "pars of original kernel are {0}".format(k1)
    gpExpSq = george.GP(k1)
    coords = np.array([[i, 3] for i in np.arange(0, grid_extent, spacing)])

    # by default the kappaKappaExpSquaredKernel has ndim = 2.0
    k = KappaKappaExpSquaredKernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(1. * k)

    gpExpSq.compute(coords, 1e-1)
    gpKKExpSq.compute(coords, 1e-1)

    Cov = gpExpSq.get_matrix(coords)
    KKCov = gpKKExpSq.get_matrix(coords)
    print "gp exp sq kernel gives {0}\n".format(Cov)
    print "\nKKExpSq kernel gives {0}\n".format(KKCov)

    return


def plotDerivCov(kernel, grid_extent=5, spacing=1):
    coords = np.array([[i, 3] for i in np.arange(0, grid_extent, spacing)])

    k = kernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(1. * k)

    gpKKExpSq.compute(coords, 1e-5)
    k.plot(spacing=spacing)


def plotExpSqCov(grid_extent, spacing, plot=False):
    coords = np.array([[i, 3] for i in np.arange(0, grid_extent, spacing)])

    k = george.kernels.ExpSquaredKernel(1.0, ndim=2)
    gpExpSq = george.GP(1. * k)

    gpExpSq.compute(coords, 1e-5)

    Cov = gpExpSq.get_matrix(coords)
    if plot:
        f, ax = plt.subplots(figsize=(12, 9))
        plt.axes().set_aspect('equal')
        cm = plt.pcolor(Cov, cmap=plt.cm.Blues)
        ylim = plt.ylim()
        plt.ylim(ylim[::-1])
        plt.xticks(rotation=45)
        # plt.xticks(np.arange(0, grid_extent, spacing), rotation=45)
        # plt.yticks(np.arange(grid_extent, 0, spacing))
        plt.title("ExpSquaredKernel " +
                  r'for points on a line ' +
                  'with spacing {0}'.format(spacing))
        plt.colorbar(cm)
        plt.show()
        plt.close()


if __name__ == "__main__":
    #test_KappaKappaExpSquare()
    plotDerivCov(KappaKappaExpSquaredKernel, grid_extent=10, spacing=1,)
    plotDerivCov(KappaGamma1ExpSquaredKernel, grid_extent=10, spacing=1,)
    plotDerivCov(KappaGamma2ExpSquaredKernel, grid_extent=10, spacing=1,)
    plotDerivCov(Gamma1Gamma1ExpSquaredKernel, grid_extent=10, spacing=1,)
    plotDerivCov(Gamma1Gamma2ExpSquaredKernel, grid_extent=10, spacing=1,)
    plotDerivCov(Gamma2Gamma2ExpSquaredKernel, grid_extent=10, spacing=1,)
    plotExpSqCov(grid_extent=10, spacing=1, plot=True)
