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
import json

sys.path.append("../")
from kern_deriv import KappaKappaExpSquaredKernel as KKker
from kern_deriv import KappaGamma1ExpSquaredKernel as KG1ker
from kern_deriv import KappaGamma2ExpSquaredKernel as KG2ker
from kern_deriv import Gamma1Gamma1ExpSquaredKernel as G1G1ker
from kern_deriv import Gamma2Gamma2ExpSquaredKernel as G2G2ker
from kern_deriv import Gamma1Gamma2ExpSquaredKernel as G1G2ker
from plot_kern_deriv import plotDerivCov, plotExpSqCov

# --------------------- starting to test Cython Kernel values --------
rtol = np.finfo(np.float64).eps
print ("Numpy machine precision on this machine is ", rtol)


def prepare_dict_for_test_case(jdict, kernel_name, cythonCov, pythonCov, l_sq,
                               inv_lambda, coords):
    temp_dict = {}
    temp_dict["l_sq"] = l_sq
    temp_dict["inv_lambda"] = inv_lambda
    temp_dict["coords"] = coords.tolist()
    temp_dict["cythonCov"] = cythonCov.tolist()
    temp_dict["pythonCov"] = pythonCov.tolist()
    assert kernel_name not in jdict, \
        "kernel name {} already defined in jdict.".format(kernel_name)
    jdict[kernel_name] = temp_dict
    return


def write_test_fixtures(save):
    f = open("test_case.json", "w")
    jdict = {}
    test_Cython_kappakappa_2_coords_fixed_l_sq(save, jdict)
    json.dump(jdict, f)
    f.close()


def test_Cython_kappakappa_2_coords_fixed_l_sq(save=False, jdict=None):
    l_sq = 1.0
    inv_lambda = 1.0
    ndim = 2L
    coords = np.array([[1., 2.], [4., 7.]])
    cythonGP = \
        george.GP(inv_lambda *
                  KappaKappaExpSquaredKernel(l_sq * np.ones(ndim), ndim=ndim))
    cythonCov = cythonGP.get_matrix(coords)
    pythonCov = plotDerivCov(KKker, coords, 1. / l_sq)

    if not save:
        assert np.array_equal(cythonCov, pythonCov)
    else:
        assert type(jdict) is dict, "jdict is missing for writing to json file"
        kernel_name = "kappakappa_2"
        print ("Preparing test results for %s" % kernel_name)
        return prepare_dict_for_test_case(jdict, kernel_name, cythonCov,
                                          pythonCov, l_sq, inv_lambda, coords)


def test_Cython_kappakappa_2_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1., 2.], [4., 7.]])
    ndim = 2L

    cythonGPs = [george.GP(1.0 *
                           KappaKappaExpSquaredKernel(l_sq * np.ones(ndim),
                                                      ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords) for cythonGP in cythonGPs]

    # Python implementation used beta instead of l_sq
    pythonCov = [plotDerivCov(KKker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]

    for i in range(len(l_sqs)):
        assert np.allclose(cythonCov[i], pythonCov[i], rtol=rtol)
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol

    return cythonCov


def test_Cython_kappakappa_10_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.linspace(0.1, 1., 10)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * KappaKappaExpSquaredKernel(l_sq *
                                                            np.ones(ndim),
                                                            ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    # Python implementation used beta instead of l_sq
    pythonCov = [plotDerivCov(KKker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.allclose(cythonCov[i], pythonCov[i], rtol=rtol)
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_kappagamma1_10_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.linspace(0.1, 1.0, 10)])
    ndim = 2L

    cythonGPs = \
        [george.GP(1.0 *
                   KappaGamma1ExpSquaredKernel(l_sq * np.ones(ndim),
                                               ndim=ndim))
         for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    # Python implementation used beta instead of l_sq
    pythonCov = [plotDerivCov(KG1ker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.allclose(cythonCov[i], pythonCov[i], rtol=rtol)
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_kappagamma2_10_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * KappaGamma2ExpSquaredKernel(l_sq *
                                                            np.ones(ndim),
                                                            ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(KG2ker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.allclose(cythonCov[i], pythonCov[i], rtol=rtol)
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_gamma1gamma1_10_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * Gamma1Gamma1ExpSquaredKernel(l_sq *
                                                              np.ones(ndim),
                                                              ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(G1G1ker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.allclose(cythonCov[i], pythonCov[i], rtol=rtol)
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_gamma1gamma2_10_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * Gamma1Gamma2ExpSquaredKernel(l_sq *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(G1G2ker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.array_equal(cythonCov[i], pythonCov[i])
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_gamma2gamma2_10_coords_vary_l_sq():
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])
    ndim = 2L

    cythonGPs = [george.GP(1.0 * Gamma2Gamma2ExpSquaredKernel(l_sq *
                                                            np.ones(ndim),
                                                      ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(G2G2ker, coords, beta=1 / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.allclose(cythonCov[i], pythonCov[i], rtol=rtol)
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_kappakappa_10_coords_fixed_lambda():
    inv_lambda = 2.0
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])
    ndim = 2L

    cythonGPs = [george.GP(inv_lambda *
                           KappaKappaExpSquaredKernel(l_sq * np.ones(ndim),
                                                      ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    pythonCov = [plotDerivCov(KKker, coords, beta=1. / l_sq)
                 for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    for i in range(len(l_sqs)):
        assert np.allclose(1 / inv_lambda * cythonCov[i], pythonCov[i],
                           rtol=rtol)
        assert np.sum(1 / inv_lambda * cythonCov[i] - pythonCov[i]) < rtol * 5e2


def test_Cython_sqExp_10_coords_vary_l_sq():
    ndim = 2L
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])

    cythonGPs = [george.GP(1.0 * ExpSquaredKernel(l_sq * np.ones(ndim),
                                                  ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotExpSqCov(coords, l_sq=l_sq, plot2=False)
                 for l_sq in l_sqs]

    for i in range(len(l_sqs)):
        assert np.array_equal(cythonCov[i], pythonCov[i])
        assert np.sum(cythonCov[i] - pythonCov[i]) < rtol


def test_Cython_sqExp_10_coords_fixed_lambda():
    ndim = 2L
    inv_lambda = 0.5
    l_sqs = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 11.1, 1)])

    cythonGPs = [george.GP(inv_lambda * ExpSquaredKernel(l_sq * np.ones(ndim),
                           ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotExpSqCov(coords, l_sq=l_sq, plot2=False)
                 for l_sq in l_sqs]

    for i in range(len(l_sqs)):
        assert np.array_equal(1 / inv_lambda * cythonCov[i], pythonCov[i])
        assert np.sum(1 / inv_lambda * cythonCov[i] - pythonCov[i]) < rtol


def test_timing_Cython_kappakappa_n_coords_fixed_lambda(n=1000):
    inv_lambda = 2.0
    l_sqs = [0.5]  # np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.linspace(0.1, 11.1,  n)])
    ndim = 2L

    cythonGPs = [george.GP(inv_lambda *
                           KappaKappaExpSquaredKernel(l_sq * np.ones(ndim),
                                                      ndim=ndim))
                 for l_sq in l_sqs]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]
    # print("cythonCov:", cythonCov)

    #pythonCov = [plotDerivCov(KKker, coords, beta=1. / l_sq)
    #             for l_sq in l_sqs]
    # print("pythonCov:", pythonCov[0])

    # for i in range(len(l_sqs)):
    #     assert np.allclose(1 / inv_lambda * cythonCov[i], pythonCov[i],
    #                        rtol=rtol)
    #     assert np.sum(1 / inv_lambda * cythonCov[i] - pythonCov[i]) < rtol * 5e2


if __name__ == "__main__":
    save = True

    _ = test_timing_Cython_kappakappa_n_coords_fixed_lambda(2000)

    # write_test_fixtures(save)

    # cythonCov1 = test_Cython_kappakappa_2_coords_fixed_l_sq()
    # cythonCov2 = test_Cython_kappakappa_10_coords_vary_l_sq()
    # _ = test_Cython_kappakappa_10_coords_vary_l_sq()
    # _ = test_Cython_kappagamma1_10_coords_vary_l_sq()
    # _ = test_Cython_kappagamma2_10_coords_vary_l_sq()
    # _ = test_Cython_gamma1gamma1_10_coords_vary_l_sq()
    # _ = test_Cython_gamma1gamma2_10_coords_vary_l_sq()
    # _ = test_Cython_gamma2gamma2_10_coords_vary_l_sq()
