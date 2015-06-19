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

def two_coords_test_data(beta=1.0):
    """
    How to write test:
    * row number is m, n
    * column number (spatial_ix) is indicated by list of 4 integers
    * only differences of the same column will be computed
    """
    return beta, np.array([[1., 2.],
                           [4., 7.]])


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

# -----set up kernel classes --------------------------------------
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


@pytest.fixture(scope="class")
def kernels_beta_equals_pt_25():
    beta, coords = two_coords_test_data(0.25)
    print(beta)
    ker = {}
    ker["kappakappa"] = KappaKappaExpSquaredKernel(beta, coords, ndim=2,
                                                   verbose=True)
    ker["kappagamma1"] = KappaGamma1ExpSquaredKernel(beta, coords, ndim=2,
                                                     verbose=True)
    ker["kappagamma2"] = KappaGamma2ExpSquaredKernel(beta, coords, ndim=2,
                                                     verbose=True)
    ker["gamma1gamma1"] = \
        Gamma1Gamma1ExpSquaredKernel(beta, coords, ndim=2, verbose=True)
    ker["gamma1gamma2"] = \
        Gamma1Gamma2ExpSquaredKernel(beta, coords, ndim=2, verbose=True)
    ker["gamma2gamma2"] = \
        Gamma2Gamma2ExpSquaredKernel(beta, coords, ndim=2, verbose=True)
    return ker


# --------- testing the kernels for non-kernel specific things -----
def test__X__(kernels, kernels_beta_equals_pt_25):
    # __X__ is not affected by beta
    for kernels in (kernels, kernels_beta_equals_pt_25):
        for k, v in kernels.iteritems():
            # __X__(m, n, ix)
            assert v.__X__(1, 0, 0) == 3.
            assert v.__X__(0, 1, 0) == -3.
            assert v.__X__(1, 0, 1) == 5.
            assert v.__X__(0, 1, 1) == -5.
            assert v.__X__(0, 0, 1) == 0


def test_term_A(kernels, kernels_beta_equals_pt_25):
    for kernels in [kernels, kernels_beta_equals_pt_25]:
        for k, v in kernels.iteritems():
            assert v.__termA__([1, 1, 1, 1], 1, 0) - 625. < 1e-10
            assert v.__termA__([1, 1, 0, 0], 1, 0) - 225. < 1e-10
            assert v.__termA__([1, 0, 1, 0], 1, 0) - 225. < 1e-10
            assert v.__termA__([1, 1, 1, 0], 1, 0) - 375. < 1e-10
            assert v.__termA__([0, 0, 0, 0], 1, 0) - 81. < 1e-10


def test_term_B(kernels, kernels_beta_equals_pt_25):
    for kernels in [kernels, kernels_beta_equals_pt_25]:
        for k, v in kernels.iteritems():
            assert v.__termB__([1, 1, 0, 1], 1, 0, [1, 1]) == 0
            # test zero metric
            assert v.__termB__([1, 1, 0, 0], 1, 0, [0, 1]) == 0
            assert v.__termB__([1, 1, 0, 0], 1, 0, [1, 1]) == 25.
            assert v.__termB__([1, 1, 1, 1], 1, 0, [1, 1]) - 25. < 1e-10
            assert v.__termB__([1, 0, 0, 0], 1, 0, [1, 1]) - 15. < 1e-10
            assert v.__termB__([1, 0, 1, 0], 1, 0, [1, 1]) == 0


def test_term_C(kernels, kernels_beta_equals_pt_25):
    for kernels in [kernels, kernels_beta_equals_pt_25]:
        for k, v in kernels.iteritems():
            assert v.__termC__([1, 1, 1, 1], [1, 2]) == 4.
            assert v.__termC__([0, 0, 1, 1], [1, 2]) == 2.
            assert v.__termC__([1, 1, 1, 1], [1, 1]) == 1.
            assert v.__termC__([1, 0, 1, 1], [1, 1]) == 0
            assert v.__termC__([1, 1, 0, 1], [1, 1]) == 0
            assert v.__termC__([1, 1, 0, 0], [1, 1]) == 1
            assert v.__termC__([1, 0, 1, 0], [1, 1]) == 0


def test_comb_B_ix(kernels, kernels_beta_equals_pt_25):
    for kernels in [kernels, kernels_beta_equals_pt_25]:
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


def test_comb_C_ix(kernels, kernels_beta_equals_pt_25):
    for kernels in [kernels, kernels_beta_equals_pt_25]:
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


def test_Sigma4thDeriv_pt_25(kernels_beta_equals_pt_25):
    kernels = kernels_beta_equals_pt_25
    # test individual elements of the kernel matrix
    for k, v in kernels.iteritems():
        assert v.__Sigma4thDeriv__([1, 1, 1, 1], 1, 0, [1, 1]) == \
            (.25 ** 4 * 5 ** 4. -       # 1 permutation of term A
             .25 ** 3 * 6. * 5 ** 2. +  # 6 permutations of term B
             .25 ** 2 * 3 * 1) / 4.     # 3 permutations of term C

        # test the metric
        assert v.__Sigma4thDeriv__([1, 1, 1, 1], 1, 0, [1, 2]) == \
            (.25 ** 4. * 5 ** 4. - .25 ** 3 * 6. * 25 * 2 +
             .25 ** 2. * 3 * 4) / 4.

        # test different indices
        assert v.__Sigma4thDeriv__([1, 1, 0, 0], 1, 0, [1, 1]) - \
            (.25 ** 4. * 5 ** 2 * 3 ** 2 -    # 1 permutation of term A
             .25 ** 3. * (5 ** 2. + 3 ** 2.)  # 2 of the 6 termBs are nonzero
             + .25 ** 2. * 1) / 4. == 0.      # 1 of the 3 termCs are nonzero

        assert v.__Sigma4thDeriv__([1, 1, 0, 0], 1, 0, [1, 1]) == \
            v.__Sigma4thDeriv__([0, 0, 1, 1], 1, 0, [1, 1])

        assert v.__Sigma4thDeriv__([1, 0, 1, 0], 1, 0, [1, 1]) - \
            (.25 ** 4. * 5. ** 2 * 3. ** 2. -  # 1 permutation of term A
             .25 ** 3. * (5 ** 2. + 3 ** 2.) + # 2 of the 6 termBs are nonzero
             .25 ** 2. * 1) / 4. == 0.         # 1 of the 3 termCs are nonzero

        assert v.__Sigma4thDeriv__([1, 0, 1, 0], 1, 0, [1, 1]) == \
            v.__Sigma4thDeriv__([0, 1, 0, 1], 1, 0, [1, 1])

        assert v.__Sigma4thDeriv__([1, 1, 1, 0], 1, 0, [1, 1]) == \
            (.25 ** 4 * 5 ** 3. * 3 ** 1 -  # 1 permutation of term A
             .25 ** 3. * 3 * 15 + 0) / 4.   #


def test_compute_Sigma4derv_matrix_beta_equals_pt_25(kernels_beta_equals_pt_25):
    # tests most elements of the kernel matrix for 1 sets of 4 indices
    for k, v in kernels_beta_equals_pt_25.iteritems():
        # the matrices are symmetric
        ker = v.__compute_Sigma4derv_matrix__([1, 1, 1, 1], [1, 1])

        # termA and Bs are all zero so we only calculate termCs only
        # on the diagonal, and we have 3 termCs
        assert ker[0, 0] == 3 * 1 / 4. * .25 ** 2.
        assert ker[1, 1] == 3 * 1 / 4. * .25 ** 2.

        # more terms off diagonal to consider
        assert ker[1, 0] == ker[0, 1]
        assert ker[1, 0] == (5 ** 4 * 0.25 ** 4. -     # term A
                             # 6 permutations of term B
                             6 * 5. ** 2. * 0.25 ** 3. +
                             # 3 permutations of term C
                             3 * 1 * 0.25 ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 0, 0, 0], [1, 1])
        assert ker[0, 0] == 3 * 1 / 4. * .25 ** 2.
        assert ker[1, 1] == 3 * 1 / 4. * .25 ** 2.
        assert ker[1, 0] == (3 ** 4 * .25 ** 4.-
                             6 * 3 ** 2 * .25 ** 3. +
                             3 * 1 * .25 ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 0, 1, 1], [1, 1])
        assert ker[0, 0] == 1 / 4. * .25 ** 2.  # term C
        assert ker[1, 1] == 1 / 4. * .25 ** 2.  # term C
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2 * .25 ** 4.) -
                             ((3 ** 2 + 5 ** 2) * .25 ** 3.) +
                             1 * .25 ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([1, 1, 0, 0], [1, 1])
        assert ker[0, 0] == 1 / 4. * .25 ** 2.  # term C
        assert ker[1, 1] == 1 / 4. * .25 ** 2.  # term C
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2 * .25 ** 4.) -
                             ((3 ** 2 + 5 ** 2) * .25 ** 3.) +
                             1 * .25 ** 2.) / 4.

        # all other permutations of 1100 will take the following values
        ker = v.__compute_Sigma4derv_matrix__([1, 0, 1, 0], [1, 1])
        assert ker[0, 0] == 1 / 4. * .25 ** 2.
        assert ker[1, 1] == 1 / 4. * .25 ** 2.
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2) * .25 ** 4. -  # termA
                             (5 ** 2 + 3 ** 2.) * .25 ** 3.+  # termB
                             1 ** 2. * .25 ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([1, 1, 1, 0], [1, 1])
        assert ker[0, 0] == 0
        assert ker[1, 1] == 0

        # there is only (termA) - termB since termC = 0
        assert ker[1, 0] == ((3 * 5 ** 3 * .25 ** 4.) -
                             (5 * 3 * 3 * .25 ** 3.)) / 4.

        # all permutations of 1000
        ker = v.__compute_Sigma4derv_matrix__([1, 0, 0, 0], [1, 1])
        assert ker[0, 0] == 0
        assert ker[1, 1] == 0

        # there is only (termA) - termB since termC = 0
        assert ker[1, 0] == ((5 * 3 ** 3 * .25 ** 4.) -
                             (5 * 3 * 3 * .25 ** 3.)) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 1, 1, 1], [1, 1])
        assert ker[1, 0] == ((5 ** 3 * 3 * .25 ** 4.) -
                             (5 * 3 * 3 * .25 ** 3.)) / 4.

# -----------kernel dependent tests! -------------------------------
def test_kappakappa_value_beta_equals_pt_25(kernels_beta_equals_pt_25):
    # following tests only test when beta = .25
    # this test is supposed to fail
    k = kernels_beta_equals_pt_25["kappakappa"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # postive definite
    # check eqn. (2) from `ker_deriv.pdf`

    # only term C contributes
    assert ker_val[0, 0] == (3 * 1 / 4. +      # 1111
                             1 / 4. +          # only 1 term from 1122 survives
                             1 / 4. +          # only 1 term from 2211 survives
                             3 * 1 / 4.) * orig_ker[0][0] * .25 ** 2.  # 2222
    assert ker_val[1, 1] == (3 / 4. + 1 / 4. + 1 / 4. + 3 / 4.) * \
        orig_ker[1][1] * .25 ** 2.

    # all terms contribute for off diagonal terms
    assert ker_val[1, 0] - ((3 ** 4 * .25 ** 4. -             # term A 1111
                             6 * 3 ** 2 * .25 ** 3. +         # 6 terms B 1111
                             3 * 1 * .25 ** 2.)  +            # 3 terms C 1111
                            ((3 ** 2 * 5 ** 2) * .25 ** 4. -  # term A 1122
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +  # 2 terms B 1122
                             1 ** 2. * .25 ** 2)  +           # 1 terms C 1122
                            ((3 ** 2 * 5 ** 2) * .25 ** 4. -  # term A 2211
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +  # 2 term B 2211
                             1 ** 2. * .25 ** 2)  +           # term C 2211
                            (5 ** 4 * .25 ** 4. -             # term A 2222
                             6 * 5 ** 2 * .25 ** 3. +         # 6 terms B 2222
                             3 * 1 * .25 ** 2.)               # 3 terms C 2222
                            ) / 4. * orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]

    return ker_val


def test_kappagamma1_value_beta_equals_pt_25(kernels_beta_equals_pt_25):
    k = kernels_beta_equals_pt_25["kappagamma1"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    # these ker_val[0, 0] and ker_val[1, 1] shouldn't be zero due to termC
    # of the derivative

    # check eqn. (5) from `ker_deriv.pdf`
    # these two tests give zeros so doesn't matter
    assert ker_val[0, 0] == (3 * 1 / 4. + 1 / 4. - 1 / 4. - 3 * 1 / 4.) * \
        orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. + 1 / 4. - 1 / 4. - 3 / 4.) * \
        orig_ker[1][1]

    assert ker_val[1, 0] - ((3 ** 4 * .25 ** 4. -              # 1111 term A
                             6 * 3 ** 2 * .25 ** 3+            # 1111 term B
                             3 * 1 * .25 ** 2.) +              # 1111 term C
                            ((3 ** 2 * 5 ** 2 * .25 ** 4.) -   # 1122 term A
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +   # 1122 term B
                             1 * .25 ** 2.) -                  # 1122 term C
                            ((3 ** 2 * 5 ** 2) * .25 ** 4. -   # 2211 term A
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +   # 2211 term B
                             1 * .25 ** 2.) -                  # 2211 term C
                            (5 ** 4 * .25 ** 4. -              # 2222 term A
                             6 * 5 ** 2 * .25 ** 3. +          # 2222 term B
                             3 * 1 * .25 ** 2.)                # 2222 term C
                            ) / 4. * orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]


def test_kappagamma2_value_beta_equals_pt_25(kernels_beta_equals_pt_25):
    k = kernels_beta_equals_pt_25["kappagamma2"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert ker_val[0, 0] == 0
    assert ker_val[1, 1] == 0

    # term C is 0
    assert ker_val[1, 0] == ((3 * 5 ** 3 * .25 ** 4.) -  # 2221, 2212 term A
                             (5 * 3 * 3) * .25 ** 3. +   # 3 term B 2221,2212
                             (5 * 3 ** 3) * .25 ** 4. -  # 1112, 1121 term A
                             (5 * 3 * 3) * .25 ** 3.     # 3 term B 1112,1121
                             ) * 2. / 4. * orig_ker[1][0]
    assert ker_val[1, 0] == ker_val[0, 1]


def test_gamma1gamma1_value_beta_equals_pt_25(kernels_beta_equals_pt_25):
    k = kernels_beta_equals_pt_25["gamma1gamma1"]

    # original kernel value from ExpSquaredKernel
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # positive definite

    # these are all term Cs since termAs and termBs are zeros
    assert ker_val[0, 0] == (3 * 1 / 4. -      # 1111
                             1 / 4. -          # 1122
                             1 / 4. +          # 2211
                             3 * 1 / 4.        # 2222
                             ) * .25 ** 2. * orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. - 1 / 4. - 1 / 4. + 3 / 4.) * \
        orig_ker[1][1] * .25 ** 2.

    assert ker_val[1, 0] - ((3 ** 4 * .25 ** 4. -
                             6 * 3 ** 2 * .25 ** 3. +
                             3 * 1 * .25 ** 2.) -           # 1111
                            ((3 ** 2 * 5 ** 2 * .25 ** 4.) -
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +
                             1 * .25 ** 2.) -               # 1122
                            ((3 ** 2 * 5 ** 2 * .25 ** 4.) -
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +
                             1 * .25 ** 2.) +              # 2211
                            (5 ** 4 * .25 ** 4. -
                             6 * 5 ** 2 * .25 ** 3. +
                             3 * 1 * .25 ** 2.)             # 2222
                            ) / 4. * orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]


def test_gamma1gamma2_value_beta_equals_pt_25(kernels_beta_equals_pt_25):
    k = kernels_beta_equals_pt_25["gamma1gamma2"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.), 1]])
    # assert np.linalg.slogdet(ker_val)[0] == 1  # postive definite
    assert ker_val[0, 0] == 0
    assert ker_val[1, 1] == 0
    # term C is zero
    assert ker_val[1, 0] == (((5 * 3 ** 3 * .25 ** 4.) -
                              (5 * 3 * 3) * .25 ** 3.) -
                             ((3 * 5 ** 3 * .25 ** 4.) -
                              (5 * 3 * 3 * .25 ** 3.))
                             ) / 4 * 2 * orig_ker[1][0]
    assert ker_val[1, 0] == ker_val[0, 1]


def test_gamma2gamma2_value_inv_beta_equals_pt_25(kernels_beta_equals_pt_25):
    k = kernels_beta_equals_pt_25["gamma2gamma2"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-.25 * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # postive definite
    # only term Cs are non-zeros
    assert ker_val[0, 0] == (1 / 4.) * 4. * .25 ** 2. * orig_ker[0][0]
    assert ker_val[1, 1] == (1 / 4.) * 4. * .25 ** 2. * orig_ker[0][0]

    assert ker_val[1, 0] == ((3 ** 2 * 5 ** 2 * .25 ** 4.) -  # termA
                             (3 ** 2 + 5 ** 2) * .25 ** 3. +  # termB
                             1 * .25 ** 2.                    # termC
                             ) * orig_ker[1][0]

    assert ker_val[0, 1] == ker_val[1, 0]


# -----------kernel dependent varying beta (inv_beta) -----------------------
# following tests only test when inv_beta = 1.
def test_kappakappa_value(kernels):
    k = kernels["kappakappa"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-(3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-(3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # positive definite
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


def test_inv_beta():
    return


# if __name__ == "__main__":
#     kappakappaCov = test_kappakappa_value(kernels())
#
# def test_positive_definiteness():
#     assert np.linalg.slogdet()[0] == 1
# pass
