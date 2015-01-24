r"""
all functions as wrapper around George for our kernel derivatives

see: https://github.com/karenyyng/shear_gp/blob/master/notes/ker_deriv.pdf
for the exact mathematical expressions

Read https://github.com/dfm/george/blob/master/george/kernels.py
for how the kernels are implemented in george

stability : untested
"""
from george.kernels import ExpSquaredKernel, RadialKernel


class kernelDerivatives():
    def indices_sanity_check(self, ix):
        assert ix == 0 or ix == 1, \
            "index has to be either 0 or 1"
        return

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
        map(indices_sanity_check, ix)

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


class kappaKappaExpSquareKernel(kernelDerivatives, ExpSquaredKernel,
                                RadialKernel):
    r"""
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:

    .. math::
        eqn (2) from kern_deriv.pdf

    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])
        self.__ix_list__ = [[1, 1, 1, 1],
                            [1, 1, 2, 2],
                            [2, 2, 1, 1],
                            [2, 2, 2, 2]]

        self.__terms_signs__ = [1, 1, 1, 1]

        #self.__term__ =
        ## there should be a method here that grabs the kernel and multiply it with
        ## suitable coefficients


class gamma1Gamma1ExpSquareKernel(kernelDerivatives, ExpSquaredKernel,
                                  RadialKernel):
    r"""
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:
    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # have to think about how to account for the negative sign in eqn (3)
        self.__ix_list__ = [[1, 1, 1, 1],
                            [1, 1, 2, 2],
                            [2, 2, 1, 1],
                            [2, 2, 2, 2]]

        self.__terms_signs__ = [1, 1, 1, 1]


class gamma2Gamma2ExpSquareKernel(kernelDerivatives, ExpSquaredKernel,
                                  RadialKernel):
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


class kappaGamma1ExpSquareKernel(kernelDerivatives, ExpSquaredKernel,
                                 RadialKernel):
    r"""
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:

    .. math::
        eqn (5) from kern_deriv.pdf

    """
    def __init__(self, metric, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])
        self.__ix_list__ = [[1, 1, 1, 1],
                            [1, 1, 2, 2],
                            [2, 2, 1, 1],  # negative
                            [2, 2, 2, 2]]  # negative

        self.__terms_signs__ = [1, 1, -1, -1]


class kappaGamma2ExpSquareKernel(kernelDerivatives, ExpSquaredKernel,
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


class Gamma1Gamma2ExpSquareKernel(kernelDerivatives, ExpSquaredKernel,
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
