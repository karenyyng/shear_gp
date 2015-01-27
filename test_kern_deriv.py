import george
import numpy as np
from kern_deriv import KappaKappaExpSquareKernel


def test_KappaKappaExpSquare():
    k1 = 2.0 * george.kernels.ExpSquaredKernel(metric=1.0, ndim=2)
    print "pars of original kernel are {0}".format(k1)
    gpExpSq = george.GP(k1)


    coords = np.array([[0, 1], [1, 0]])

    # by default the kappaKappaExpSquareKernel has ndim = 2.0
    k = KappaKappaExpSquareKernel(1.0, coords, ndim=2)
    gpKKExpSq = george.GP(2. * k)

    gpExpSq.compute(coords, 1e-1)
    gpKKExpSq.compute(coords, 1e-1)

    print "gp exp sq kernel gives \n", gpExpSq.get_matrix(coords)
    print "\nKKExpSq kernel gives \n", gpKKExpSq.get_matrix(coords)

    return


if __name__ == "__main__":
    test_KappaKappaExpSquare()
