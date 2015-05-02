from __future__ import (print_function, division, unicode_literals)
from george.kernels import (KappaKappaExpSquaredKernel,
                            KappaGamma1ExpSquaredKernel,
                            KappaGamma2ExpSquaredKernel,
                            Gamma1Gamma1ExpSquaredKernel,
                            Gamma2Gamma2ExpSquaredKernel,
                            Gamma1Gamma2ExpSquaredKernel,
                            ExpSquaredKernel)
import george
import numpy as np
import sys
sys.path.append("../")
from kern_deriv import KappaKappaExpSquaredKernel as KKker
from kern_deriv import KappaGamma1ExpSquaredKernel as KG1ker
from kern_deriv import KappaGamma2ExpSquaredKernel as KG2ker
from kern_deriv import Gamma1Gamma1ExpSquaredKernel as G1G1ker
from kern_deriv import Gamma2Gamma2ExpSquaredKernel as G2G2ker
from kern_deriv import Gamma1Gamma2ExpSquaredKernel as G1G2ker
from plot_kern_deriv import plotDerivCov, plotExpSqCov

# --------------------- starting to test Cython Kernel values --------


def test_Cython_kappakappa_2_coords_fixed_beta():
    beta = 1.0
    ndim = 2L
    coords = np.array([[1., 2.], [4., 7.]])
    cythonGP = george.GP(1.0 * KappaKappaExpSquaredKernel(beta * np.ones(ndim),
                                                          ndim=ndim))
    cythonCov = cythonGP.get_matrix(coords)
    pythonCov = plotDerivCov(KKker, coords, beta)

    assert np.array_equal(cythonCov, pythonCov)
    return


def test_Cython_kappakappa_2_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1., 2.], [4., 7.]])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * KappaKappaExpSquaredKernel(beta * np.ones(ndim),
                                                            ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotDerivCov(KKker, coords, beta)
                 for beta in betas]

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])

    return cythonCov


def test_Cython_kappakappa_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * KappaKappaExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(KKker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])


def test_Cython_kappagamma1_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * KappaGamma1ExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(KG1ker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])


def test_Cython_kappagamma2_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * KappaGamma2ExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(KG2ker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])


def test_Cython_gamma1gamma1_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * Gamma1Gamma1ExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(G1G1ker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])


def test_Cython_gamma1gamma2_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * Gamma1Gamma2ExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(G1G2ker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])


def test_Cython_gamma2gamma2_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * Gamma2Gamma2ExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(G2G2ker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(cythonCov[i], pythonCov[i])


def test_Cython_sqExp_10_coords_vary_beta():
    ndim = 2L
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])

    cythonGPs = [george.GP(1.0 * ExpSquaredKernel(beta * np.ones(ndim),
                                                  ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotExpSqCov(coords, beta=beta, plot2=False)
                 for beta in betas]

    for i in range(len(betas)):
        assert np.array_equal(cythonCov[i], pythonCov[i])


def test_Cython_sqExp_10_coords_fixed_lambda():
    ndim = 2L
    inv_lambda = 0.5
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])

    cythonGPs = [george.GP(inv_lambda * ExpSquaredKernel(beta * np.ones(ndim),
                                                     ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotExpSqCov(coords, beta=beta, plot2=False)
                 for beta in betas]

    for i in range(len(betas)):
        assert np.array_equal(1 / inv_lambda * cythonCov[i], pythonCov[i])


def test_Cython_kappakappa_10_coords_fixed_lambda():
    inv_lambda = 2.0
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    ndim = 2L

    cythonGPs = [george.GP(inv_lambda * KappaKappaExpSquaredKernel(beta *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(KKker, coords, beta)
                 for beta in betas]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(betas)):
        assert np.allclose(1 / inv_lambda * cythonCov[i], pythonCov[i])


if __name__ == "__main__":
    cythonCov1 = test_Cython_kappakappa_2_coords_vary_beta()
    cythonCov2 = test_Cython_kappakappa_10_coords_vary_beta()
