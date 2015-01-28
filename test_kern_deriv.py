import george
import numpy as np
from kern_deriv import KappaKappaExpSquareKernel
import matplotlib.pyplot as plt


def test_KappaKappaExpSquare():
    """sanity check"""
    k1 = 1.0 * george.kernels.ExpSquaredKernel(metric=1.0, ndim=2)
    print "pars of original kernel are {0}".format(k1)
    gpExpSq = george.GP(k1)

    coords = np.array([[1, 3], [2, 5]])

    # by default the kappaKappaExpSquareKernel has ndim = 2.0
    k = KappaKappaExpSquareKernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(1. * k)

    gpExpSq.compute(coords, 1e-1)
    gpKKExpSq.compute(coords, 1e-1)

    Cov = gpExpSq.get_matrix(coords)
    KKCov = gpKKExpSq.get_matrix(coords)
    print "gp exp sq kernel gives {0}\n".format(Cov)
    print "\nKKExpSq kernel gives {0}\n".format(KKCov)

    return


def plotKKCov(grid_extent=2, spacing=0.05):
    coords = np.array([[i, j] for i in np.arange(0, grid_extent, spacing)
                       for j in np.arange(0, grid_extent, spacing)])

    k = KappaKappaExpSquareKernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(1. * k)

    gpKKExpSq.compute(coords, 1)

    KKCov = gpKKExpSq.get_matrix(coords)
    plt.imshow(KKCov, origin='upper', extent=[0, grid_extent, grid_extent, 0],
               vmin=0, vmax=1)
    plt.xticks(rotation=45)
    #plt.yticks(np.arange(0, 1, 0.05))
    plt.title(
        r'$Cov(\kappa, \kappa)$ as 4th deriv of ExpSq kernel on square grid' +
        ' of spacing {0}'.format(spacing))
    plt.colorbar()
    plt.savefig('./plots/covKappaKappa.png', bbox_inches='tight')
    plt.close()


def plotExpSqCov(grid_extent=2, spacing=0.05):
    coords = np.array([[i, j] for i in np.arange(0, 1, spacing)
                       for j in np.arange(0, 1, spacing)])

    k = george.kernels.ExpSquaredKernel(1.0, ndim=2)
    gpExpSq = george.GP(1. * k)

    gpExpSq.compute(coords, 1)

    Cov = gpExpSq.get_matrix(coords)
    plt.imshow(Cov, origin='upper', extent=[0, grid_extent,
                                            grid_extent, 0 ],
               vmin=0, vmax=1)
    plt.xticks(rotation=45)
    # plt.xticks(np.arange(0, 1, 0.05), rotation=45)
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.title(r'ExpSquared Kernel visualized on square grid of spacing ' +
              '{0}'.format(spacing))
    plt.colorbar()
    plt.savefig('./plots/covExpSqKern.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    #test_KappaKappaExpSquare()
    plotKKCov()
    plotExpSqCov()
