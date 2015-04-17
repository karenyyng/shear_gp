"""
test the Python implementation of the new kernels
"""
from __future__ import division, print_function
import pytest
import numpy as np
import sys
sys.path.append("../")

from kern_deriv import (KappaKappaExpSquaredKernel,
                        KappaGamma1ExpSquaredKernel,
                        KappaGamma2ExpSquaredKernel,
                        Gamma1Gamma1ExpSquaredKernel,
                        Gamma2Gamma2ExpSquaredKernel,
                        Gamma1Gamma2ExpSquaredKernel,
                        isotropic_norm,
                        normalized_corr)


# --------create fixtures ----------------------------

@pytest.fixture()
def two_coords_test_data():
    # only use fixtures if you foresee having to share test data across
    # different tests
    beta = 1.
    return beta, np.array([[1., 2.], [4., 7.]])


# --------test noramlization and parametrization --------

def test_isotropic_norm():
    def test_norm_do_not_change_normed_coords():
        coords = np.array([[0., 0.], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        assert isotropic_norm(coords) == 1.

    def test_norm_gets_unnormed_coords_normalized():
        coords = np.array([[0., 0.], [10, 0.]])
        assert isotropic_norm(coords) == 1.


def test_normalized_corr():
    coords = np.array([[0., 0.], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    assert normalized_corr(2., coords) - np.exp(-4. * 1. * 2.) < 1e-10


# --------- testing the kernels for non-kernel specific things -------------
@pytest.fixture(scope="class")
def kernels():
    beta, coords = two_coords_test_data()
    ker = {}
    ker["kappakappa"] = KappaKappaExpSquaredKernel(beta, coords, ndim=2)
    ker["kappagamma1"] = KappaGamma1ExpSquaredKernel(beta, coords, ndim=2)
    ker["kappagamma2"] = KappaGamma2ExpSquaredKernel(beta, coords, ndim=2)
    ker["gamma1gamma1"] = \
        Gamma1Gamma1ExpSquaredKernel(beta, coords, ndim=2)
    ker["gamma1gamma2"] = \
        Gamma1Gamma2ExpSquaredKernel(beta, coords, ndim=2)
    ker["gamma2gamma2"] = \
        Gamma2Gamma2ExpSquaredKernel(beta, coords, ndim=2)
    return ker


def test__X__(kernels):
    for k, v in kernels.iteritems():
        # __X__(m, n, ix)
        assert v.__X__(1, 0, 0) == 3.
        assert v.__X__(0, 1, 0) == -3.
        assert v.__X__(1, 0, 1) == 5.
        assert v.__X__(0, 1, 1) == -5.
        assert v.__X__(0, 0, 1) == 0


def test_term_A(kernels):
    for k, v in kernels.iteritems():
        assert v.__termA__([1, 1, 1, 1], 1, 0) - 625. < 1e-10
        assert v.__termA__([1, 1, 0, 0], 1, 0) - 225. < 1e-10
        assert v.__termA__([1, 0, 1, 0], 1, 0) - 225. < 1e-10
        assert v.__termA__([1, 1, 1, 0], 1, 0) - 375. < 1e-10
        assert v.__termA__([0, 0, 0, 0], 1, 0) - 81. < 1e-10


def test_term_B(kernels):
    for k, v in kernels.iteritems():
        assert v.__termB__([1, 1, 0, 1], 1, 0, [1, 1]) == 0
        assert v.__termB__([1, 1, 0, 0], 1, 0, [0, 1]) == 0  # test zero metric
        assert v.__termB__([1, 1, 0, 0], 1, 0, [1, 1]) == 25.
        assert v.__termB__([1, 1, 1, 1], 1, 0, [1, 1]) - 25. < 1e-10
        assert v.__termB__([1, 0, 0, 0], 1, 0, [1, 1]) - 15. < 1e-10
        assert v.__termB__([1, 0, 1, 0], 1, 0, [1, 1]) == 0


def test_term_C(kernels):
    for k, v in kernels.iteritems():
        assert v.__termC__([1, 1, 1, 1], [1, 2]) == 4.
        assert v.__termC__([0, 0, 1, 1], [1, 2]) == 2.
        assert v.__termC__([1, 1, 1, 1], [1, 1]) == 1.
        assert v.__termC__([1, 0, 1, 1], [1, 1]) == 0
        assert v.__termC__([1, 1, 0, 1], [1, 1]) == 0
        assert v.__termC__([1, 1, 0, 0], [1, 1]) == 1
        assert v.__termC__([1, 0, 1, 0], [1, 1]) == 0


def test_comb_B_ix(kernels):
    for k, v in kernels.iteritems():
        assert v.__comb_B_ix__([1, 1, 1, 1]) == [[1, 1, 1, 1],
                                                 [1, 1, 1, 1],
                                                 [1, 1, 1, 1],
                                                 [1, 1, 1, 1],
                                                 [1, 1, 1, 1],
                                                 [1, 1, 1, 1]]

        assert v.__comb_B_ix__([1, 1, 0, 0]) == [[1, 1, 0, 0],
                                                 [1, 0, 1, 0],
                                                 [1, 0, 1, 0],
                                                 [0, 0, 1, 1],
                                                 [1, 0, 1, 0],
                                                 [1, 0, 1, 0]]


def test_comb_C_ix(kernels):
    for k, v in kernels.iteritems():
        assert v.__comb_C_ix__([1, 1, 1, 1]) == [[1, 1, 1, 1],
                                                 [1, 1, 1, 1],
                                                 [1, 1, 1, 1]]

        assert v.__comb_C_ix__([1, 1, 0, 0]) == [[1, 1, 0, 0],
                                                 [1, 0, 1, 0],
                                                 [1, 0, 1, 0]]


def test_Sigma4thDeriv(kernels):
    # test individual elements of the kernel matrix
    for k, v in kernels.iteritems():
        assert v.__Sigma4thDeriv__([1, 1, 1, 1], 1, 0, [1, 1]) == \
            (625 - 6. * 25 + 3 * 1) / 4.

        # test the metric
        assert v.__Sigma4thDeriv__([1, 1, 1, 1], 1, 0, [1, 2]) == \
            (625 - 6. * 25 * 2 + 3 * 4) / 4.

        # test different indices
        assert v.__Sigma4thDeriv__([1, 1, 0, 0], 1, 0, [1, 1]) - \
            (225 - (25 - 9)  # 2 of the 6 terms are nonzero
             + 1) / 4. < 1e-10  # 1 of the 3 terms are nonzero

        assert v.__Sigma4thDeriv__([1, 1, 0, 0], 1, 0, [1, 1]) == \
            v.__Sigma4thDeriv__([0, 0, 1, 1], 1, 0, [1, 1])

        assert v.__Sigma4thDeriv__([1, 0, 1, 0], 1, 0, [1, 1]) - \
            (225. - (25 - 9) + 1) / 4. < 1e-10

        assert v.__Sigma4thDeriv__([1, 0, 1, 0], 1, 0, [1, 1]) == \
            v.__Sigma4thDeriv__([0, 1, 0, 1], 1, 0, [1, 1])

        assert v.__Sigma4thDeriv__([1, 1, 1, 0], 1, 0, [1, 1]) == \
            (375 - 3 * 15 + 0) / 4.


def test_compute_Sigma4derv_matrix(kernels):
    # tests most elements of the kernel matrix for 1 sets of 4 indices
    for k, v in kernels.iteritems():
        # the matrices are symmetric
        ker = v.__compute_Sigma4derv_matrix__([1, 1, 1, 1], [1, 1])

        # termA and Bs are all zero so we only calculate termCs only
        # on the diagonal, and we have 3 termCs
        assert ker[0, 0] == 3 * 1 / 4.
        assert ker[1, 1] == 3 * 1 / 4.

        # more terms off diagonal to consider
        assert ker[1, 0] == ker[0, 1]
        assert ker[1, 0] == (5 ** 4 - 6 * 5 ** 2 + 3 * 1) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 0, 0, 0], [1, 1])
        assert ker[0, 0] == 3 * 1 / 4.
        assert ker[1, 1] == 3 * 1 / 4.
        assert ker[1, 0] == (3 ** 4 - 6 * 3 ** 2 + 3 * 1) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 0, 1, 1], [1, 1])
        assert ker[0, 0] == 1 / 4.
        assert ker[1, 1] == 1 / 4.
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4.

        ker = v.__compute_Sigma4derv_matrix__([1, 1, 0, 0], [1, 1])
        assert ker[0, 0] == 1 / 4.
        assert ker[1, 1] == 1 / 4.
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4.

        # all other permutations of 1100 will take the following values
        ker = v.__compute_Sigma4derv_matrix__([1, 0, 1, 0], [1, 1])
        assert ker[0, 0] == 1 / 4.
        assert ker[1, 1] == 1 / 4.
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4.

        ker = v.__compute_Sigma4derv_matrix__([1, 1, 1, 0], [1, 1])
        assert ker[0, 0] == 0
        assert ker[1, 1] == 0

        # there is only (termA) - termB since termC = 0
        assert ker[1, 0] == ((3 * 5 ** 3) - (5 * 3 * 3)) / 4.

        # all permutations of 1000
        ker = v.__compute_Sigma4derv_matrix__([1, 0, 0, 0], [1, 1])
        assert ker[0, 0] == 0
        assert ker[1, 1] == 0

        # there is only (termA) - termB since termC = 0
        assert ker[1, 0] == ((5 * 3 ** 3) - (5 * 3 * 3)) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 1, 1, 1], [1, 1])
        assert ker[1, 0] == ((5 ** 3 * 3) - (5 * 3 * 3)) / 4.

# -----------kernel dependent tests! -------------------------------
def test_kappakappa_value(kernels):
    k = kernels["kappakappa"]
    ker_val = k.value()

    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1
    assert ker_val[0, 0] == (3 * 1 / 4. + 1 / 4. + 1 / 4. + 3 * 1 / 4.) * \
        orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. + 1 / 4. + 1 / 4. + 3 / 4.) * \
        orig_ker[1][1]

    assert ker_val[1, 0] - ((3 ** 4 - 6 * 3 ** 2 + 3 * 1) / 4. +
                            ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4. +
                            ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4. +
                            (5 ** 4 - 6 * 5 ** 2 + 3 * 1) / 4.) * \
        orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]

    print("kappakappa cov = ", ker_val)
    return ker_val


def test_kappagamma1_value(kernels):
    k = kernels["kappagamma1"]
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])

    # these ker_val[0, 0] and ker_val[1, 1] shouldn't be zero from the physics
    assert ker_val[0, 0] == (3 * 1 / 4. + 1 / 4. - 1 / 4. - 3 * 1 / 4.) * \
        orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. + 1 / 4. - 1 / 4. - 3 / 4.) * \
        orig_ker[1][1]

    assert ker_val[1, 0] - ((3 ** 4 - 6 * 3 ** 2 + 3 * 1) / 4. +
                            ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4. -
                            ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4. -
                            (5 ** 4 - 6 * 5 ** 2 + 3 * 1) / 4.) * \
        orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]


def test_kappagamma2_value(kernels):
    k = kernels["kappagamma2"]
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert ker_val[0, 0] == 0
    assert ker_val[1, 1] == 0
    assert ker_val[1, 0] == (((3 * 5 ** 3) - (5 * 3 * 3)) / 4 * 2 +
                             ((5 * 3 ** 3) - (5 * 3 * 3)) / 4 * 2) * \
        orig_ker[1][0]
    assert ker_val[1, 0] == ker_val[0, 1]


def test_gamma1gamma1_value(kernels):
    k = kernels["gamma1gamma1"]
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1

    assert ker_val[0, 0] == (3 * 1 / 4. - 1 / 4. - 1 / 4. + 3 * 1 / 4.) * \
        orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. - 1 / 4. - 1 / 4. + 3 / 4.) * \
        orig_ker[1][1]

    assert ker_val[1, 0] - ((3 ** 4 - 6 * 3 ** 2 + 3 * 1) / 4. -
                            ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4. -
                            ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) / 4. +
                            (5 ** 4 - 6 * 5 ** 2 + 3 * 1) / 4.) * \
        orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]


def test_gamma1gamma2_value(kernels):
    k = kernels["gamma1gamma2"]
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])
    assert ker_val[0, 0] == 0
    assert ker_val[1, 1] == 0
    assert ker_val[1, 0] == (((5 * 3 ** 3) - (5 * 3 * 3)) / 4 * 2 -
                             ((3 * 5 ** 3) - (5 * 3 * 3)) / 4 * 2) * \
        orig_ker[1][0]
    assert ker_val[1, 0] == ker_val[0, 1]


def test_gamma2gamma2_value(kernels):
    k = kernels["gamma2gamma2"]
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1
    assert ker_val[0, 0] == (1 / 4.) * 4. * orig_ker[0][0]
    assert ker_val[1, 1] == (1 / 4.) * 4. * orig_ker[0][0]

    assert ker_val[1, 0] == ((3 ** 2 * 5 ** 2) - (3 ** 2 + 5 ** 2) + 1) * \
        orig_ker[1][0]

    assert ker_val[0, 1] == ker_val[1, 0]



if __name__ == "__main__":
    kappakappaCov = test_kappakappa_value(kernels())
#
# def test_positive_definiteness():
#     assert np.linalg.slogdet()[0] == 1
# pass
