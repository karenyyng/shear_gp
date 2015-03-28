"""
prototype for the Cython code for implementing the derivatives of the kernels

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
# import george
from george.kernels import ExpSquaredKernel
import matplotlib.pyplot as plt
import numpy as np


class KernelDerivatives(ExpSquaredKernel):

    """
    this is intended to be a `abstract` / `virtual` class and
    not to meant to be instantiated directly

    need to inherit the `value` method of ExpSquaredKernel

    :params beta:
    """

    def __init__(self, beta, ndim=2, dim=-1, extra=[], rMetric=1.):
        # the following corresponds to Kernel.__init__
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])

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

    def __termA__(self, coords, ix, m, n, debug=False):
        """
        # the constructor also needs the coordinates
        Compute term 1 in equation (27) without leading factors of $\beta^4$

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
        if debug:
            print "indices of term A = {0}".format(ix)
            print "type of indices of term A = ", len(ix)
        # print "ix in termA is ", ix
        for i in ix:
            term *= self.__X__(coords, m, n, i)

        return term

    def __termB__(self, coords, ix, m, n, metric, debug=False):
        """
        Compute term 2 in equation (27) without leading factors of $\beta^3$

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
        if debug is True:
            print "metric = {0}".format(metric)
            print "indices of term B = {0}".format(ix)

        if ix[2] != ix[3]:
            return 0

        return self.__X__(coords, m, n, ix[0]) * \
            self.__X__(coords, m, n, ix[1]) * \
            metric[ix[2]]

    def __termC__(self, coords, ix, metric, debug=False):
        """
        Compute term 3 in equation (27) without leading factor of $\beta^2$

        :param coords: numpy array,
            shape is (obs_no, 2) where 2 denotes the 2 spatial dimensions

        :param ix: list of 4 integers,
            denote the spatial subscripts for the ker deriv,
            assumes to take the form [a, b, c, d]

        :param metric: list of floats
            should be of dimension 2, we assume diagonal metric here

        .. math:
            D_{ab} D_{cd} \delta_{ab} \delta_{cd}
        """
        if debug:
            print "metric = {0}".format(metric)
            print "indices of term C = {0}".format(ix)

        if ix[0] != ix[1]:
            return 0

        if ix[2] != ix[3]:
            return 0

        return metric[ix[2]] * metric[ix[0]]

    def __Sigma4thDeriv__(self, beta, coords, ix, m, n, metric, debug=False):
        """
        Gather the 10 terms for the 4th derivative of each Sigma
        given the ix for each the derivatives are taken w.r.t.

        :params beta: float
            value of the beta parameter in the ExpSquaredKernel

        :params coords: 2D numpy array
            with shape (nObs, ndim)

        :params ix: list of 4 integers

        :returns: components of the $\Gamma$ term in eqn (27) without the signs
        :note: see eqn (2) etc. for the factor of 1 / 4.
        """
        assert isinstance(beta, float) or isinstance(beta, int), \
            "beta param has to be a number"

        allTermBs = 0
        combBix = \
            [[ix[i] for i in self.__pairsOfBIndices__[j]] for j in range(6)]

        # combBix is the subscript indices combination for B terms
        for i in range(6):
            allTermBs += self.__termB__(coords, combBix[i], m, n, metric,
                                        debug=debug)

        allTermCs = 0
        combCix = \
            [[ix[i] for i in self.__pairsOfCIndices__[j]] for j in range(3)]

        for i in range(3):
            allTermCs += self.__termC__(coords, combCix[i], metric,
                                        debug=debug)

        termA = self.__termA__(coords, ix, m, n)

        if debug:
            print "combBix is ", combBix
            print "combCix is ", combCix
            print "beta is ", beta
            print "terms are {0}, {1}, {2}".format(termA, allTermBs, allTermCs)

        return (beta ** 4. * termA -
                beta ** 3. * allTermBs +
                beta ** 2. * allTermCs) / 4.

    def __compute_Sigma4derv_matrix__(self, x, par, ix, metric, debug=False):
        """
        Compute the coefficients due to the derivatives - this
        should result in a symmetric N x N matrix where N is the
        number of observations

        moved from KappaKappaExpSquaredKernel to here

        :params par: theta_2^2 according to George parametrization
        :params ix: list of 4 integers to indicate derivative subscripts
        """

        return np.array([[self.__Sigma4thDeriv__(par, x, ix, m, n, metric,
                                                 debug=debug)
                         for m in range(x.shape[0])]
                         for n in range(x.shape[0])
                         ])

    def value(self, x1, ix_list, pars, terms_signs, metric, x2=None,
              debug=False):
        """
        This child class's method overrides the parent class's method
        to multiple our kernel with appropriate coefficients

        compute equation (27) by multiplying the coefficients due to
        derivatives and the actual kernel matrix

        :param terms_signs: list of four 1 of -1,
            these corresponds to the signs of the 4 terms in eqn (2) - (7)

        :param pars: float, this is beta

        :note:
            not sure what x2 is for, my guess is that x1 is the training
            data, x2 is the test data according to `gp.predict`

        """
        if debug:
            print "calling KernelDerivatives class value method"
            print "pars for beta is {0}".format(pars)
        # use parent class, Kernel.value method to parse values

        mat = np.zeros((x1.shape[0], x1.shape[0]))
        for i in range(len(ix_list)):
            # self.__compute_Sigma4derv_matrix__(i, x1)
            # new implementation this should call the KernelDerivatives
            # method
            mat += terms_signs[i] * \
                self.__compute_Sigma4derv_matrix__(x1, pars,
                                                   ix_list[i],
                                                   metric)

        cov_mat = super(KernelDerivatives, self).value(x1, x2)

        # return the Schur product of the matrix
        # print "kern deriv coeff is {0}\n".format(mat)
        # print "original cov_mat is {0}\n".format(cov_mat)
        # print "Schur product is {0}\n".format(mat * cov_mat)
        return mat * cov_mat

    def plot_kernel_mtx(self, spacing, save=False, fig="./plots/", name=None):
        f, ax = plt.subplots(figsize=(12, 9))
        plt.axes().set_aspect('equal')
        cm = plt.pcolor(self.__kernel__, cmap=plt.cm.Blues)
        # vmin=0., vmax=2.5)

        # y axis should be flipped to match matrix indices
        ylim = plt.ylim()
        plt.ylim(ylim[::-1])
        plt.xticks(rotation=45)
        plt.title(name)
        plt.colorbar(cm)
        if save:
            plt.savefig(fig + name + '.png', bbox_inches='tight')

        plt.show()
        plt.close()
        return


class KappaKappaExpSquaredKernel(KernelDerivatives, ExpSquaredKernel):

    """
    Inherits from the ExpSquaredKernel class and multiplies it with appropriate
    coefficients

    :params coords: 2D numpy array
        with shape = (n_obs, 2)

    .. math::
        eqn (2) from kern_deriv.pdf
    """

    def __init__(self, beta, coords, ndim=2, dim=-1, extra=[], rMetric=1.):
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])
        self.__beta__ = beta

        # this should call KernelDerivatives.__init__()
        super(KappaKappaExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                                         dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if isinstance(rMetric, float) or isinstance(rMetric, int):
            self.__metric__ = rMetric * np.ones(ndim)

        # python arrays are zeroth indexed
        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]

        self.__kernel__ = self.value(coords)
        print "Kernel is {0}".format(self.__kernel__)
        print "stored version of pars in George is {0}".format(self.pars)

    def value(self, x1, x2=None, param=None):
        """ this won't be called by George betaectly """
        return super(KappaKappaExpSquaredKernel, self).value(
            x1, ix_list=self.__ix_list__,
            pars=self.__beta__, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)

    def plot1(self, spacing, save=False, fig="./plots",
              name='KappaKappaExpSquaredKernel'):
        self.plot_kernel_mtx(spacing, save=save, fig=fig, name=name)


class KappaGamma1ExpSquaredKernel(KernelDerivatives, ExpSquaredKernel):

    """
    inherits from the ExpSquaredKernel class and multiplies it with
    appropriate coefficients
    :params coords:

    .. math:: eqn (5) from kern_deriv.pdf
    """

    def __init__(self, beta, coords, ndim=2, dim=-1, extra=[], rMetric=1.):
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])

        self.__beta__ = beta

        # this should call KernelDerivatives.__init__()
        super(KappaGamma1ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                                          dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__coords__ = coords

        if isinstance(rMetric, float) or isinstance(rMetric, int):
            self.__metric__ = rMetric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],  # negative
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1  # negative

        self.__terms_signs__ = [1, -1, 1, -1]

        self.__kernel__ = self.value(self.__coords__)

    def value(self, x1, x2=None):
        return super(KappaGamma1ExpSquaredKernel, self).value(
            x1, ix_list=self.__ix_list__,
            pars=self.__beta__, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)

    def plot1(self, spacing, save=False, fig="./plots",
              name='KappaGamma1ExpSquaredKernel'):
        self.plot_kernel_mtx(spacing, save=save, fig=fig, name=name)


class KappaGamma2ExpSquaredKernel(KernelDerivatives, ExpSquaredKernel):

    """
    inherits from the ExpSquaredKernel class and multiplies it with appropriate
    coefficients

    :params coords:

    .. math::
        eqn (6) from kern_deriv.pdf

    """

    def __init__(self, beta, coords, ndim=2, dim=-1, extra=[], rMetric=1.):
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(KappaGamma2ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                                          dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__beta__ = beta
        self.__coords__ = coords

        if isinstance(rMetric, float) or isinstance(rMetric, int):
            self.__metric__ = rMetric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 1, 1, 2],
                                     [1, 1, 2, 1],
                                     [2, 2, 1, 2],
                                     [2, 2, 2, 1]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]

        self.__kernel__ = self.value(self.__coords__)

    def value(self, x1, x2=None):
        return super(
            KappaGamma2ExpSquaredKernel, self).value(
            x1, ix_list=self.__ix_list__,
            pars=self.__beta__, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)

    def plot1(self, spacing, save=False, fig="./plots",
              name="KappaGamma2ExpSquaredKernel"):
        self.plot_kernel_mtx(spacing, save=save, fig=fig, name=name)


class Gamma1Gamma1ExpSquaredKernel(KernelDerivatives, ExpSquaredKernel):

    """
    Inherits from the ExpSquaredKernel class and multiplies it with appropriate
    coefficients

    :params coords:
    """

    def __init__(self, beta, coords, ndim=2, dim=-1, extra=[], rMetric=1.):
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(Gamma1Gamma1ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                                           dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        if isinstance(rMetric, float) or isinstance(rMetric, int):
            self.__metric__ = rMetric * np.ones(ndim)

        # have to think about how to account for the negative sign in eqn (3)
        self.__ix_list__ = np.array([[1, 1, 1, 1],
                                     [1, 1, 2, 2],
                                     [2, 2, 1, 1],
                                     [2, 2, 2, 2]]) - 1

        self.__terms_signs__ = [1, -1, -1, 1]
        self.__beta__ = beta
        self.__coords__ = coords

        self.__kernel__ = self.value(self.__coords__)

    def value(self, x1, x2=None):
        return super(Gamma1Gamma1ExpSquaredKernel, self).value(
            x1, ix_list=self.__ix_list__,
            pars=self.__beta__, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)

    def plot1(self, spacing, save=False, fig="./plots",
              name='Gamma1Gamma1ExpSquaredKernel'):
        self.plot_kernel_mtx(spacing, save=save, fig=fig, name=name)


class Gamma2Gamma2ExpSquaredKernel(KernelDerivatives, ExpSquaredKernel):

    """
    inherits from the ExpSquaredKernel class and multiplies it with
    appropriate coefficients

    :params metric: a list of 2 integers
    """

    def __init__(self, beta, coords, ndim=2, dim=-1, extra=[], rMetric=1.):
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(Gamma2Gamma2ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                                           dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__beta__ = beta
        self.__coords__ = coords

        if isinstance(rMetric, float) or isinstance(rMetric, int):
            self.__metric__ = rMetric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 2, 1, 2],
                                     [1, 2, 2, 1],
                                     [2, 1, 1, 2],
                                     [2, 1, 2, 1]]) - 1

        self.__terms_signs__ = [1, 1, 1, 1]

        self.__kernel__ = self.value(coords)

    def value(self, x1, x2=None):
        return super(
            Gamma2Gamma2ExpSquaredKernel, self).value(
            x1, ix_list=self.__ix_list__,
            pars=self.__beta__, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)

    def plot1(self, spacing, save=False, fig="./plots",
              name='Gamma2Gamma2ExpSquaredKernel'):
        self.plot_kernel_mtx(spacing, save=save, fig=fig, name=name)


class Gamma1Gamma2ExpSquaredKernel(KernelDerivatives, ExpSquaredKernel):

    """
    inherits from the ExpSquaredKernel class and multiplies it with
    appropriate coefficients

    :params coords:

    .. math::
        eqn (6) from kern_deriv.pdf

    """

    def __init__(self, beta, coords, ndim=2, dim=-1, extra=[], rMetric=1.):
        super(ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                               dim=-dim, extra=[])

        # this should call KernelDerivatives.__init__()
        super(Gamma1Gamma2ExpSquaredKernel, self).__init__(beta, ndim=ndim,
                                                           dim=-dim, extra=[])

        assert len(coords.shape) == 2 and coords.shape[1] == 2, \
            "dimension of the coord array is not compatible with kernel\n" + \
            "needs numpy array of shape (n_obs, 2)"

        self.__beta__ = beta
        self.__coords__ = coords

        if isinstance(rMetric, float) or isinstance(rMetric, int):
            self.__metric__ = rMetric * np.ones(ndim)

        self.__ix_list__ = np.array([[1, 1, 1, 2],
                                     [1, 1, 2, 1],
                                     [2, 2, 1, 2],  # negative
                                     [2, 2, 2, 1]]) - 1  # negative

        self.__terms_signs__ = [1, 1, -1, -1]

        self.__kernel__ = self.value(coords)

    def value(self, x1, x2=None):
        return super(Gamma1Gamma2ExpSquaredKernel, self).value(
            x1, ix_list=self.__ix_list__,
            pars=self.__beta__, terms_signs=self.__terms_signs__,
            metric=self.__metric__, x2=x2)

    def plot1(self, spacing, save=False, fig="./plots",
              name='Gamma1Gamma2ExpSquaredKernel'):
        self.plot_kernel_mtx(spacing, save=save, fig=fig, name=name)
