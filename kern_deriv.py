"""
all functions as wrapper around George for our kernel derivatives

see: https://github.com/karenyyng/shear_gp/blob/master/notes/ker_deriv.pdf
for the exact mathematical expressions

Read https://github.com/dfm/george/blob/master/george/kernels.py
for how the kernels are implemented in george

:stability: untested but runs without errors

:warning: in George, `y` refers to the variable to be predicted
    in my notes, `y` refers to an alternative way of calling the spatial
    location of the data points, and `psi` refers to the variable to
    be predicted
"""
from __future__ import division
import george
from george.kernels import ExpSquaredKernel, Kernel
import matplotlib.pyplot as plt
import numpy as np


class KernelDerivatives(Kernel):
    """
    this is intended to be a `base` / `virtual` class and not to meant to be
    instantiated directly
    """
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

    def __X__(self, coords, m, n, spat_ix):
        return coords[m, spat_ix] - coords[n, spat_ix]

    def __termA__(self, coords, ix, m, n):
        """
        Compute term 1 in equation (24) without leading factors of $\beta^4$

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
        #print "ix in termA is ", ix
        for i in ix:
            term *= self.__X__(coords, m, n, i)

        return term

    def __termB__(self, coords, ix, m, n, metric):
        """
        Compute term 2 in equation (24) without leading factors of $\beta^3$

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

        return self.__X__(coords, m, n, ix[0]) * \
            self.__X__(coords, m, n, ix[1]) * \
            metric[ix[2]]

    def __termC__(self, coords, ix, m, n, metric):
        """
        Compute term 3 in equation (24) without leading factor of $\beta^2$

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

    def __Sigma4thDeriv__(self, corr, coords, ix, m, n, metric, debug=False):
        """
        Gather the 10 terms for the 4th derivative of each Sigma
        given the ix for each the derivatives are taken w.r.t.

        :params corr: float
            value of the correlation parameter in the ExpSquaredKernel

        :params coords: 2D numpy array
            with shape (nObs, ndim)

        :params ix: list of 4 integers
        """

        if type(corr) == np.ndarray and len(corr) == 1:
            beta = corr[0]
        elif len(corr) == 2:
            beta = corr[1]
        else:
            raise ValueError("pars of {0} has unexcepted shape".format(corr))

        allTermBs = 0
        combBix = \
            [[ix[i] for i in self.__pairsOfBIndices__[j]] for j in range(6)]

        # combBix is the subscript indices combination for B terms
        for i in range(6):
            allTermBs += self.__termB__(coords, combBix[i], m, n, metric)

        allTermCs = 0
        combCix = \
            [[ix[i] for i in self.__pairsOfCIndices__[j]] for j in range(3)]

        for i in range(3):
            allTermCs += self.__termC__(coords, combCix[i], m, n, metric)

        termA = self.__termA__(coords, ix, m, n)

        if debug:
            print "combBix is ", combBix
            print "combCix is ", combCix
            print "terms are {0}, {1}, {2}".format(termA, allTermBs, allTermCs)

        return (beta ** 4. * termA +
            beta ** 3. * allTermBs +
            beta ** 2. * allTermCs) / 4.

    def __compute_Sigma4derv_matrix__(self, x, par, ix, metric):
        """
        Compute the coefficients due to the derivatives - this
        should result in a symmetric N x N matrix where N is the
        number of observations

        moved from KappaKappaExpSquareKernel to here

        :params par: theta_2^2 according to George parametrization
        :params ix: list of 4 integers to indicate derivative subscripts
        """

        return np.array([[self.__Sigma4thDeriv__(par, x, ix, m, n, metric,
                                                 debug=False)
                         for m in range(x.shape[0])]
                         for n in range(x.shape[0])
                         ])

    def value(self, x1, ix_list, pars, terms_signs, metric, x2=None):
        """
        This child class's method overrides the parent class's method
        to multiple our kernel with appropriate coefficients

        compute equation (27) by multiplying the coefficients due to
        derivatives and the actual kernel matrix

        :note:
            not sure what x2 is for, my guess is that x1 is the training
            data, x2 is the test data according to `gp.predict`

        """
        # print "calling KernelDerivatives class value method"
        # use parent class, Kernel.value method to parse values

        mat = np.zeros((x1.shape[0], x1.shape[0]))
        for i in range(len(self.__ix_list__)):
            # self.__compute_Sigma4derv_matrix__(i, x1)
            # new implementation this should call the KernelDerivatives
            # method
            mat += terms_signs[i] * \
                self.__compute_Sigma4derv_matrix__(x1, pars, ix_list[i],
                                                   metric)

        cov_mat = super(KernelDerivatives, self).value(x1, x2)

        # return the Schur product of the matrix
        return mat * cov_mat  # * 33 #* x1


    def debug_value(self):
        print "DEBUGGING THE INHERITANCE"
        return


class KappaKappaExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    """
    Inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords: 2D numpy array
        with shape = (n_obs, 2)

    .. math::
        eqn (2) from kern_deriv.pdf
    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(KappaKappaExpSquareKernel, self).__init__()

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if type(metric) == float or type(metric) == int:
            self.__metric__ = metric * np.ones(ndim)

        # python arrays are zeroth indexed
        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]


        self.__kernel__ = \
            super(KappaKappaExpSquareKernel, self).value(
            coords, ix_list=self.__ix_list__,
            pars=self.pars, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=None)


        print("Kernel value = {0}".format(self.__kernel__))
        plt.clf()
        plt.title("{0}".format(self))
        plt.imshow(self.__kernel__, origin='upper')  #, cmap=plt.cmap.winter)
        plt.show()

    def value(self, x1, x2=None):
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print "calling new kernel value method"
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return self.__kernel__
           # super(
           #  KappaKappaExpSquareKernel, self).value(
           #  x1, ix_list=self.__ix_list__,
           #  pars=self.pars, terms_signs=self.__terms_signs__,
           #  metric=self.__metric__, x2=x2)

    # def plotDerivCov(kernel, grid_extent=2, spacing=0.05,):
    #     coords = np.array([[0, i] for i in np.arange(0, grid_extent, spacing)])
    #     #coords = np.array([[i, j] for i in np.arange(0, grid_extent, spacing)
    #     #                   for j in np.arange(0, grid_extent, spacing)])

    #     k = kernel(1.0, coords, ndim=2)
    #     gpKKExpSq = george.GP(1. * k)

    #     gpKKExpSq.compute(coords, 1e-5)

    #     KKCov = gpKKExpSq.get_matrix(coords)
    #     print "KKCov is {0}".format(KKCov)
    #     plt.imshow(KKCov, origin='upper', extent=[0, grid_extent,
    #                                               grid_extent, 0])
    #             #vmin=0, vmax=1)
    #     plt.xticks(rotation=45)
    #     # plt.yticks(np.arange(0, 1, 0.05))
    #     plt.title(
    #         #r'$Cov(\kappa, \kappa)$ as 4th deriv of ExpSq kernel
    #         kernel.__name__ + ' visualized on a line of coords of spacing' +
    #         ' {0}'.format(spacing))
    #     plt.colorbar()
    #     plt.savefig('./plots/' + kernel.__name__ + '.png', bbox_inches='tight')
    #     plt.close()

        # # calling the value this way is correct
        # return KernelDerivatives.value(
        #     x1, ix_list=self.__ix_list__,
        #     pars=self.pars, terms_signs=self.__terms_signs__,
        #     metric=self.__metric__, x2=x2)

    # def debug_value(self, x1, x2=None):
    #     """
    #     for debugging purpose this calls the original values
    #     for the computed matrix
    #     """
    #     return super(KappaKappaExpSquareKernel, self).value(x1, x2)


class KappaGamma1ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    """
    inherits from the ExpSquareKernel class and multiplies it with
    appropriate coefficients
    :params coords:

    .. math:: eqn (5) from kern_deriv.pdf
    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(KappaGamma1ExpSquareKernel, self).__init__()

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if type(metric) == float or type(metric) == int:
            self.__metric__ = metric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],  # negative
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1  # negative

        self.__terms_signs__ = [1, -1, 1, -1]

        self.__kernel__ = \
            super(KappaGamma1ExpSquareKernel, self).value(
            coords, ix_list=self.__ix_list__,
            pars=self.pars, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=None)


        plt.clf()
        print "Kernel is {0}".format(self.__kernel__)
        plt.imshow(self.__kernel__, origin='upper')  #, cmap=plt.cmap.winter)
        plt.show()

    def value(self, x1, x2=None):
        return self.__kernel__
        #super(
        #    KappaGamma1ExpSquareKernel, self).value(
        #    self, x1, ix_list=self.__ix_list__,
        #    pars=self.pars, terms_signs=self.__terms_signs__,
        #    metric=self.__metric__, x2=x2)


class KappaGamma2ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    """
    inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:

    .. math::
        eqn (6) from kern_deriv.pdf

    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(KappaGamma2ExpSquareKernel, self).__init__()

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if type(metric) == float or type(metric) == int:
            self.__metric__ = metric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 1, 1, 2],
                                     [1, 1, 2, 1],
                                     [2, 2, 1, 2],
                                     [2, 2, 2, 1]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]

    def value(self, x1, x2=None):
        return super(
            KappaGamma1ExpSquareKernel, self).value(
            self, x1, ix_list=self.__ix_list__,
            pars=self.pars, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)


class Gamma1Gamma1ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    """
    Inherits from the ExpSquareKernel class and multiplies it with appropriate
    coefficients

    :params coords:
    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(Gamma1Gamma1ExpSquareKernel, self).__init__()

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        # have to think about how to account for the negative sign in eqn (3)
        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1

        self.__terms_signs__ = [1, -1, -1, 1]

    def value(self, x1, x2=None):
        return super(
            Gamma1Gamma1ExpSquareKernel, self).value(
            self, x1, ix_list=self.__ix_list__,
            pars=self.pars, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)


class Gamma2Gamma2ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    """
    inherits from the ExpSquareKernel class and multiplies it with
    appropriate coefficients

    :params metric: a list of 2 integers
    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(Gamma2Gamma2ExpSquareKernel, self).__init__()

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if type(metric) == float or type(metric) == int:
            self.__metric__ = metric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 2, 1, 2],
                                     [1, 2, 2, 1],
                                     [2, 1, 1, 2],
                                     [2, 1, 2, 1]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]

    def value(self, x1, x2=None):
        return super(
            Gamma2Gamma2ExpSquareKernel, self).value(
            self, x1, ix_list=self.__ix_list__,
            pars=self.pars, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)


class Gamma1Gamma2ExpSquareKernel(KernelDerivatives, ExpSquaredKernel):
    """
    inherits from the ExpSquareKernel class and multiplies it with
    appropriate coefficients

    :params coords:

    .. math::
        eqn (6) from kern_deriv.pdf

    """
    def __init__(self, metric, coords, ndim=2, dim=-1, extra=[]):
        super(ExpSquaredKernel, self).__init__(metric, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(Gamma1Gamma2ExpSquareKernel, self).__init__()

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if type(metric) == float or type(metric) == int:
            self.__metric__ = metric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 1, 1, 2],
                                     [1, 1, 2, 1],
                                     [2, 2, 1, 2],  # negative
                                     [2, 2, 2, 1]]) - 1  # negative

        self.__terms_signs__ = [1, 1, -1, -1]

    def value(self, x1, x2=None):
        return super(
            KappaGamma1ExpSquareKernel, self).value(
            self, x1, ix_list=self.__ix_list__,
            pars=self.pars, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)
