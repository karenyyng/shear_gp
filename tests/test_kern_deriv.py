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
def kernels_dict(beta=.5):
    beta, coords = two_coords_test_data(beta)
    print("Setting beta to ", beta)
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
    return ker, beta


# --------- testing the kernels for non-kernel specific things -----
def test__X__(kernels_dict):
    kernels, beta = kernels_dict
    # __X__ is not affected by beta
    for k, v in kernels.iteritems():
        # __X__(m, n, ix)
        assert v.__X__(1, 0, 0) == 3.
        assert v.__X__(0, 1, 0) == -3.
        assert v.__X__(1, 0, 1) == 5.
        assert v.__X__(0, 1, 1) == -5.
        assert v.__X__(0, 0, 1) == 0


def test_term_A(kernels_dict):
    kernels, beta = kernels_dict
    for k, v in kernels.iteritems():
        assert v.__termA__([1, 1, 1, 1], 1, 0) - 625. == 0
        assert v.__termA__([1, 1, 0, 0], 1, 0) - 225. == 0
        assert v.__termA__([1, 0, 1, 0], 1, 0) - 225. == 0
        assert v.__termA__([1, 1, 1, 0], 1, 0) - 375. == 0
        assert v.__termA__([0, 0, 0, 0], 1, 0) - 81.  == 0


def test_term_B(kernels_dict):
    kernels, beta = kernels_dict
    for k, v in kernels.iteritems():
        assert v.__termB__([1, 1, 0, 1], 1, 0, [1, 1]) == 0
        # test zero metric
        assert v.__termB__([1, 1, 0, 0], 1, 0, [0, 1]) == 0
        assert v.__termB__([1, 1, 0, 0], 1, 0, [1, 1]) - 25. == 0
        assert v.__termB__([1, 1, 1, 1], 1, 0, [1, 1]) - 25. == 0
        assert v.__termB__([1, 0, 0, 0], 1, 0, [1, 1]) - 15. == 0
        assert v.__termB__([1, 0, 1, 0], 1, 0, [1, 1]) == 0


def test_term_C(kernels_dict):
    kernels, beta = kernels_dict
    for k, v in kernels.iteritems():
        assert v.__termC__([1, 1, 1, 1], [1, 2]) == 4.
        assert v.__termC__([0, 0, 1, 1], [1, 2]) == 2.
        assert v.__termC__([1, 1, 1, 1], [1, 1]) == 1.
        assert v.__termC__([1, 0, 1, 1], [1, 1]) == 0.
        assert v.__termC__([1, 1, 0, 1], [1, 1]) == 0.
        assert v.__termC__([1, 1, 0, 0], [1, 1]) == 1.
        assert v.__termC__([1, 0, 1, 0], [1, 1]) == 0.


def test_comb_B_ix(kernels_dict):
    kernels, beta = kernels_dict
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


def test_comb_C_ix(kernels_dict):
    kernels, beta = kernels_dict
    for k, v in kernels.iteritems():
        assert v.__comb_C_ix__([1, 1, 1, 1]) == [[1, 1, 1, 1],
                                                [1, 1, 1, 1],
                                                [1, 1, 1, 1]]

        assert v.__comb_C_ix__([1, 1, 0, 0]) == [[1, 1, 0, 0],
                                                [1, 0, 1, 0],
                                                [1, 0, 1, 0]]


def test_Sigma4thDeriv(kernels_dict):
    kernels, beta = kernels_dict
    # test individual elements of the kernel matrix
    for k, v in kernels.iteritems():
        assert v.__Sigma4thDeriv__([1, 1, 1, 1], 1, 0, [1, 1]) == \
            (beta ** 4 * 5 ** 4. -       # 1 permutation of term A
             beta ** 3 * 6. * 5 ** 2. +  # 6 permutations of term B
             beta ** 2 * 3 * 1) / 4.     # 3 permutations of term C

        # test the metric
        assert v.__Sigma4thDeriv__([1, 1, 1, 1], 1, 0, [1, 2]) == \
            (beta ** 4. * 5 ** 4. - beta ** 3 * 6. * 25 * 2 +
             beta ** 2. * 3 * 4) / 4.

        # test different indices
        assert v.__Sigma4thDeriv__([1, 1, 0, 0], 1, 0, [1, 1]) - \
            (beta ** 4. * 5 ** 2 * 3 ** 2 -    # 1 permutation of term A
             beta ** 3. * (5 ** 2. + 3 ** 2.)  # 2 of the 6 termBs are nonzero
             + beta ** 2. * 1) / 4. == 0.      # 1 of the 3 termCs are nonzero

        assert v.__Sigma4thDeriv__([1, 1, 0, 0], 1, 0, [1, 1]) == \
            v.__Sigma4thDeriv__([0, 0, 1, 1], 1, 0, [1, 1])

        assert v.__Sigma4thDeriv__([1, 0, 1, 0], 1, 0, [1, 1]) - \
            (beta ** 4. * 5. ** 2 * 3. ** 2. -  # 1 permutation of term A
             beta ** 3. * (5 ** 2. + 3 ** 2.) + # 2 of the 6 termBs are nonzero
             beta ** 2. * 1) / 4. == 0.         # 1 of the 3 termCs are nonzero

        assert v.__Sigma4thDeriv__([1, 0, 1, 0], 1, 0, [1, 1]) == \
            v.__Sigma4thDeriv__([0, 1, 0, 1], 1, 0, [1, 1])

        assert v.__Sigma4thDeriv__([1, 1, 1, 0], 1, 0, [1, 1]) == \
            (beta ** 4 * 5 ** 3. * 3 ** 1 -  # 1 permutation of term A
             beta ** 3. * 3 * 15 + 0) / 4.   #


def test_compute_Sigma4derv_matrix(kernels_dict):
    kernels, beta = kernels_dict
    # tests most elements of the kernel matrix for 1 sets of 4 indices
    for k, v in kernels.iteritems():
        # the matrices are symmetric
        ker = v.__compute_Sigma4derv_matrix__([1, 1, 1, 1], [1, 1])

        # termA and Bs are all zero so we only calculate termCs only
        # on the diagonal, and we have 3 termCs
        assert ker[0, 0] == 3 * 1 / 4. * beta ** 2.
        assert ker[1, 1] == 3 * 1 / 4. * beta ** 2.

        # more terms off diagonal to consider
        assert ker[1, 0] == ker[0, 1]
        assert ker[1, 0] == (5 ** 4 * beta ** 4. -     # term A
                             # 6 permutations of term B
                             6 * 5. ** 2. * beta ** 3. +
                             # 3 permutations of term C
                             3 * 1 * beta ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 0, 0, 0], [1, 1])
        assert ker[0, 0] == 3 * 1 / 4. * beta ** 2.
        assert ker[1, 1] == 3 * 1 / 4. * beta ** 2.
        assert ker[1, 0] == (3 ** 4 * beta ** 4. -
                             6 * 3 ** 2 * beta ** 3. +
                             3 * 1 * beta ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 0, 1, 1], [1, 1])
        assert ker[0, 0] == 1 / 4. * beta ** 2.  # term C
        assert ker[1, 1] == 1 / 4. * beta ** 2.  # term C
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2 * beta ** 4.) -
                             ((3 ** 2 + 5 ** 2) * beta ** 3.) +
                             1 * beta ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([1, 1, 0, 0], [1, 1])
        assert ker[0, 0] == 1 / 4. * beta ** 2.  # term C
        assert ker[1, 1] == 1 / 4. * beta ** 2.  # term C
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2 * beta ** 4.) -
                             ((3 ** 2 + 5 ** 2) * beta ** 3.) +
                             1 * beta ** 2.) / 4.

        # all other permutations of 1100 will take the following values
        ker = v.__compute_Sigma4derv_matrix__([1, 0, 1, 0], [1, 1])
        assert ker[0, 0] == 1 / 4. * beta ** 2.
        assert ker[1, 1] == 1 / 4. * beta ** 2.
        assert ker[1, 0] == ((3 ** 2 * 5 ** 2) * beta ** 4. -  # termA
                             (5 ** 2 + 3 ** 2.) * beta ** 3.+  # termB
                             1 ** 2. * beta ** 2.) / 4.

        ker = v.__compute_Sigma4derv_matrix__([1, 1, 1, 0], [1, 1])
        assert ker[0, 0] == 0
        assert ker[1, 1] == 0

        # there is only (termA) - termB since termC = 0
        assert ker[1, 0] == ((3 * 5 ** 3 * beta ** 4.) -
                             (5 * 3 * 3 * beta ** 3.)) / 4.

        # all permutations of 1000
        ker = v.__compute_Sigma4derv_matrix__([1, 0, 0, 0], [1, 1])
        assert ker[0, 0] == 0
        assert ker[1, 1] == 0

        # there is only (termA) - termB since termC = 0
        assert ker[1, 0] == ((5 * 3 ** 3 * beta ** 4.) -
                             (5 * 3 * 3 * beta ** 3.)) / 4.

        ker = v.__compute_Sigma4derv_matrix__([0, 1, 1, 1], [1, 1])
        assert ker[1, 0] == ((5 ** 3 * 3 * beta ** 4.) -
                             (5 * 3 * 3 * beta ** 3.)) / 4.

# -----------kernel dependent tests! -------------------------------
def test_kappakappa_value(kernels_dict):
    kernels, beta = kernels_dict
    k = kernels["kappakappa"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # postive definite
    # check eqn. (2) from `ker_deriv.pdf`

    # only term C contributes
    assert ker_val[0, 0] == (3 * 1 / 4. +      # 1111
                             1 / 4. +          # only 1 term from 1122 survives
                             1 / 4. +          # only 1 term from 2211 survives
                             3 * 1 / 4.) * orig_ker[0][0] * beta ** 2.  # 2222
    assert ker_val[1, 1] == (3 / 4. + 1 / 4. + 1 / 4. + 3 / 4.) * \
        orig_ker[1][1] * beta ** 2.

    # all terms contribute for off diagonal terms
    assert ker_val[1, 0] - ((3 ** 4 * beta ** 4. -             # term A 1111
                             6 * 3 ** 2 * beta ** 3. +         # 6 terms B 1111
                             3 * 1 * beta ** 2.)  +            # 3 terms C 1111
                            ((3 ** 2 * 5 ** 2) * beta ** 4. -  # term A 1122
                             (3 ** 2 + 5 ** 2) * beta ** 3. +  # 2 terms B 1122
                             1 ** 2. * beta ** 2)  +           # 1 terms C 1122
                            ((3 ** 2 * 5 ** 2) * beta ** 4. -  # term A 2211
                             (3 ** 2 + 5 ** 2) * beta ** 3. +  # 2 term B 2211
                             1 ** 2. * beta ** 2)  +           # term C 2211
                            (5 ** 4 * beta ** 4. -             # term A 2222
                             6 * 5 ** 2 * beta ** 3. +         # 6 terms B 2222
                             3 * 1 * beta ** 2.)               # 3 terms C 2222
                            ) / 4. * orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]

    return ker_val


def test_kappagamma1_value(kernels_dict):
    kernels, beta = kernels_dict
    k = kernels["kappagamma1"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    # these ker_val[0, 0] and ker_val[1, 1] shouldn't be zero due to termC
    # of the derivative

    # check eqn. (5) from `ker_deriv.pdf`
    # these two tests give zeros so doesn't matter
    assert ker_val[0, 0] == (3 * 1 / 4. + 1 / 4. - 1 / 4. - 3 * 1 / 4.) * \
        orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. + 1 / 4. - 1 / 4. - 3 / 4.) * \
        orig_ker[1][1]

    assert ker_val[1, 0] - ((3 ** 4 * beta ** 4. -              # 1111 term A
                             6 * 3 ** 2 * beta ** 3+            # 1111 term B
                             3 * 1 * beta ** 2.) +              # 1111 term C
                            ((3 ** 2 * 5 ** 2 * beta ** 4.) -   # 1122 term A
                             (3 ** 2 + 5 ** 2) * beta ** 3. +   # 1122 term B
                             1 * beta ** 2.) -                  # 1122 term C
                            ((3 ** 2 * 5 ** 2) * beta ** 4. -   # 2211 term A
                             (3 ** 2 + 5 ** 2) * beta ** 3. +   # 2211 term B
                             1 * beta ** 2.) -                  # 2211 term C
                            (5 ** 4 * beta ** 4. -              # 2222 term A
                             6 * 5 ** 2 * beta ** 3. +          # 2222 term B
                             3 * 1 * beta ** 2.)                # 2222 term C
                            ) / 4. * orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]


def test_kappagamma2_value(kernels_dict):
    kernels, beta = kernels_dict
    k = kernels["kappagamma2"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert ker_val[0, 0] == 0
    assert ker_val[1, 1] == 0

    # term C is 0
    assert ker_val[1, 0] == ((3 * 5 ** 3 * beta ** 4.) -   # 2221, 2212 term A
                             (5 * 3 * 3) * beta ** 3. +    # 3 term B 2221,2212
                             (5 * 3 ** 3) * beta ** 4. -   # 1112, 1121 term A
                             (5 * 3 * 3) * beta ** 3.      # 3 term B 1112,1121
                             ) * 2. / 4. * orig_ker[1][0]
    assert ker_val[1, 0] == ker_val[0, 1]


def test_gamma1gamma1_value(kernels_dict):
    kernels, beta = kernels_dict
    k = kernels["gamma1gamma1"]

    # original kernel value from ExpSquaredKernel
    ker_val = k.value()
    orig_ker = np.array([[1, np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # positive definite

    # these are all term Cs since termAs and termBs are zeros
    assert ker_val[0, 0] == (3 * 1 / 4. -      # 1111
                             1 / 4. -          # 1122
                             1 / 4. +          # 2211
                             3 * 1 / 4.        # 2222
                             ) * beta ** 2. * orig_ker[0][0]
    assert ker_val[1, 1] == (3 / 4. - 1 / 4. - 1 / 4. + 3 / 4.) * \
        orig_ker[1][1] * beta ** 2.

    assert ker_val[1, 0] - ((3 ** 4 * beta ** 4. -
                             6 * 3 ** 2 * beta ** 3. +
                             3 * 1 * beta ** 2.) -           # 1111
                            ((3 ** 2 * 5 ** 2 * beta ** 4.) -
                             (3 ** 2 + 5 ** 2) * beta ** 3. +
                             1 * beta ** 2.) -               # 1122
                            ((3 ** 2 * 5 ** 2 * beta ** 4.) -
                             (3 ** 2 + 5 ** 2) * beta ** 3. +
                             1 * beta ** 2.) +              # 2211
                            (5 ** 4 * beta ** 4. -
                             6 * 5 ** 2 * beta ** 3. +
                             3 * 1 * beta ** 2.)             # 2222
                            ) / 4. * orig_ker[1][0] == 0

    assert ker_val[0, 1] == ker_val[1, 0]


def test_gamma1gamma2_value(kernels_dict):
    kernels, beta = kernels_dict
    k = kernels["gamma1gamma2"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.), 1]])
    # assert np.linalg.slogdet(ker_val)[0] == 1  # postive definite
    assert ker_val[0, 0] == 0
    assert ker_val[1, 1] == 0
    # term C is zero
    assert ker_val[1, 0] == (((5 * 3 ** 3 * beta ** 4.) -
                              (5 * 3 * 3) * beta ** 3.) -
                             ((3 * 5 ** 3 * beta ** 4.) -
                              (5 * 3 * 3 * beta ** 3.))
                             ) / 4 * 2 * orig_ker[1][0]
    assert ker_val[1, 0] == ker_val[0, 1]


def test_gamma2gamma2_value(kernels_dict):
    kernels, beta = kernels_dict
    k = kernels["gamma2gamma2"]
    ker_val = k.value()

    # original kernel value from ExpSquaredKernel
    orig_ker = np.array([[1, np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.)],
                         [np.exp(-beta * (3 ** 2 + 5 ** 2.) / 2.), 1]])

    assert np.linalg.slogdet(ker_val)[0] == 1  # postive definite
    # only term Cs are non-zeros
    assert ker_val[0, 0] == (1 / 4.) * 4. * beta ** 2. * orig_ker[0][0]
    assert ker_val[1, 1] == (1 / 4.) * 4. * beta ** 2. * orig_ker[0][0]

    assert ker_val[1, 0] == ((3 ** 2 * 5 ** 2 * beta ** 4.) -  # termA
                             (3 ** 2 + 5 ** 2) * beta ** 3. +  # termB
                             1 * beta ** 2.                    # termC
                             ) * orig_ker[1][0]

    assert ker_val[0, 1] == ker_val[1, 0]


# if __name__ == "__main__":
#     kappakappaCov = test_kappakappa_value(kernels())
#
# def test_positive_definiteness():
#     assert np.linalg.slogdet()[0] == 1
# pass
