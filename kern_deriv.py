r"""
all functions as wrapper around George for our kernel derivatives

see: https://github.com/karenyyng/shear_gp/blob/master/notes/ker_deriv.pdf
for the exact mathematical expressions

Read https://github.com/dfm/george/blob/master/george/kernels.py
for how the kernels are implemented in george

stability : untested

warning : in George, `y` refers to the variable to be predicted
in my notes, `y` refers to an alternative way of calling the spatial location
of the data points, and `psi` refers to the variable to be predicted

"""
from george.kernels import ExpSquaredKernel, RadialKernel, Kernel
import numpy as np


class KernelDerivatives:
    def __init__(self):
        # pick 2 pairs from 4 objects so we have 4C2 combinations
        self.__pairsOfBIndices__ = \
            [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
             [2, 3, 0, 1], [1, 3, 0, 2], [1, 2, 0, 3]]

        # pick 2 pairs from 4 objects and it doesn't matter if
        # we swap the pairs, i.e. can swap pair A and B
        # so we have 4C2 / 2 combinations = 3 """
        self.__pairsOfCIndices__ = \
            [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]

    # def indices_sanity_check(self, ix):
    #     assert ix == 0 or ix == 1, \
    #         "index has to be either 0 or 1"
    #     return
    def X(self, coords, m, n, spat_ix):
        return coords[m, spat_ix] - coords[n, spat_ix]

    def termA(self, coords, ix, m, n):
        r""" term 1 in equation (24) without leading factors of $\beta^4$

        :params coords: numpy array,
            shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        :params ix: list of 4 integers,
            denote the spatial subscripts for the ker deriv

        :beta: float,
            inverse length

        .. math:
            X_i X_j X_h X_k

        """
        term = 1
        for i in ix:
            term *= self.X(coords, m, n, i)

        return term

    def termB(self, coords, ix, m, n, metric):
        r""" term 2 in equation (24) without leading factors of $\beta^3$

        :param coords: numpy array,
            shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        :param ix: list of 4 integers,
            denote the spatial subscripts for the ker deriv,
            assumes to take the form [a, b, c, d]

        :param m: integer
            denote the row index of covariance matrix, or obs_no

        :params n: integer
            denote the col index of covariance matrix, or obs_no

        :param metric: list of floats
            should be of dimension 2, we assume diagonal metric here

        .. math:
            X_a X_b D_{cd} \delta_{cd}
        """
        if ix[2] != ix[3]:
            return 0

        return self.X(coords, m, n, ix[0]) * self.X(coords, m, n, ix[1]) * \
            metric[ix[2]]

    def termC(self, coords, ix, m, n, metric):
        r""" term 3 in equation (24) without leading factor of $\beta^2$

        :param coords: numpy array,
            shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        :param ix: list of 4 integers,
            denote the spatial subscripts for the ker deriv,
            assumes to take the form [a, b, c, d]

        :param m: integer
            denote the row index of covariance matrix, or obs_no

        :params n: integer
            denote the col index of covariance matrix, or obs_no

        :param metric: list of floats
            should be of dimension 2, we assume diagonal metric here

        .. math:
            D_{ab} D_{cd} \delta_{ab} \delta_{cd}
        """
        if ix[0] != ix[1]:
            return 0

        if ix[2] != ix[3]:
            return 0

        return metric[ix[2]] * metric[ix[0]]

    def __Sigma4thDeriv__(self, coords, ix, m, n, metric):
        r"""gather the 10 terms for the 4th derivative of each Sigma
        given the ix for each the derivatives are taken w.r.t.
        """
        beta = self._par[0]

        allTermBs = 0
        combBix = \
            [ix[i] for i in self.__pairsOfBIndices__[k] for k in range(6)]
        for i in range(6):
            allTermBs += self.termB(coords, combBix[i], m, n, metric)

        allTermCs = 0
        combCix = \
            [ix[i] for i in self.__pairsOfCIndices__[j] for j in range(3)]
        for i in range(3):
            allTermCs += self.termC(coords, combCix[i], m, n, metric)

        return beta ** 4 * termA(coords, ix, m, n, metric) + \
               beta ** 3 * allTermBs + \
               beta ** 2 * allTermCs


class KappaKappaExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    r"""
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords: 2D numpy array
        with shape = (n_obs, 2)

    .. math::
        eqn (2) from kern_deriv.pdf

    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if type(metric) == float or type(metric) == int:
            self.__metric__ = metric * np.ones(ndim)

        # python indices are zeroth indexed
        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]

    def __compute_Sigma4derv__(self):
        """ compute the coefficients due to the derivatives - this
        should result in a symmetric N x N matrix where N is the
        number of observations
        """
        print self.__Sigma4thDeriv__(self.__coords__, self.__ix_list__[0],
                                     1, 1, self.__metric__)
        return

    def value(self, x1, x2=None):
        """ the child class's method overrides the parent class's method
        to multiple our kernel with appropriate coefficients """
        return super(KappaKappaExpSquareKernel, self).value(x1, x2) * 200

    def debug_value(self, x1, x2=None):
        """ for debugging purpose this calls the original values
        for the computed matrix """
        return super(KappaKappaExpSquareKernel, self).value(x1, x2)



class Gamma1Gamma1ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    r"""
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:
    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # have to think about how to account for the negative sign in eqn (3)
        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]


class Gamma2Gamma2ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    r"""inherits from the ExpSquareKernel class and multiplies it with
    appropriate coefficients

    :params metric: a list of 2 integers
    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        self.__ix_list__ = [[1, 2, 1, 2],
                            [1, 2, 2, 1],
                            [2, 1, 1, 2],
                            [2, 1, 2, 1]]

        self.__terms_signs__ = [1, 1, 1, 1]


class KappaGamma1ExpSquareKernel(KernelDerivatives, ExpSquaredKernel,
                                 RadialKernel):
    r""" inherits from the ExpSquareKernel class and multiplies it with
    appropriate coefficients
    :params coords:

    .. math:: eqn (5) from kern_deriv.pdf
    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])
        self.__ix_list__ = [[1, 1, 1, 1],
                            [1, 1, 2, 2],
                            [2, 2, 1, 1],  # negative
                            [2, 2, 2, 2]]  # negative

        self.__terms_signs__ = [1, 1, -1, -1]


class KappaGamma2ExpSquareKernel(KernelDerivatives, ExpSquaredKernel,
                                 RadialKernel):
    r"""
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:

    .. math::
        eqn (6) from kern_deriv.pdf

    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])
        self.__ix_list__ = [[1, 1, 1, 2],
                            [1, 1, 2, 1],
                            [2, 2, 1, 2],
                            [2, 2, 2, 1]]

        self.__terms_signs__ = [1, 1, 1, 1]


class Gamma1Gamma2ExpSquareKernel(KernelDerivatives, ExpSquaredKernel,
                                  RadialKernel):
    r"""inherits from the ExpSquareKernel class and multiplies it with
    appropriate coefficients

    :params coords:

    .. math::
        eqn (6) from kern_deriv.pdf

    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])
        self.__ix_list__ = [[1, 1, 1, 2],
                            [1, 1, 2, 1],
                            [2, 2, 1, 2],  # negative
                            [2, 2, 2, 1]]  # negative

        self.__terms_signs__ = [1, 1, -1, -1]
