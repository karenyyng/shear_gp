"""This creates matrix fixtures for use with the test_kernels.cpp /
test_lens_fields.cpp files in Catch.
"""
from __future__ import (division, print_function)
import sys
import george
import numpy as np
from george.kernels import (KappaKappaExpSquaredKernel,
                            KappaGamma1ExpSquaredKernel,
                            KappaGamma2ExpSquaredKernel,
                            Gamma1Gamma1ExpSquaredKernel,
                            Gamma2Gamma2ExpSquaredKernel,
                            Gamma1Gamma2ExpSquaredKernel,
                            WhiteKernel
                            )
from collections import OrderedDict


def express_matrix_in_eigen_format(mtx, filename="lens_field_fixture.txt",
                                   write=True, verbose=True):
    """ Eigen matrix can be initialized with
    mtx = ele1, ele2
    :param mtx: 2D numpy array
    :returns: None
    """
    if write:
        f = open(filename, "w")

    for mtx_row in mtx:
        line = ", ".join(["{:.8g}".format(mtx_ele) for mtx_ele in mtx_row])
        print (line)
        f.write(line + "\n")

    f.close()


def output_kernel_value(GPKer, kern_name, x_coords, l_sq, gp_prec, gp_err_prec,
                        ndim=2L, verbose=False):
    """
    @param GPKer : the GP derivative kernel from george
    @param x : numpy array, with shape (n, 2)
    """
    gp = george.GP(1. / gp_prec * GPKer(l_sq * np.ones(ndim), ndim=ndim) +
                   (WhiteKernel(1./gp_err_prec, ndim=ndim)))

    # Printing the outputs in a format so that we can copy the outputs
    # to instantiate Eigen3 matrix directly.
    mtx = gp.get_matrix(x_coords)

    if verbose:
        print_matrix_in_eigen_format(mtx)

    return mtx


def cut_mtx_apart(mtx, n_grid, n_gal, ker_name):
    """Split up matrix to conform to the Michael Schneider way of laying out
    the covariance matrix

    :mtx: 2D numpy array that represents the matrix.
    :n_grid: integer
    :n_gal: integer
    :ker_name: string
    :returns: a dictionary of the split-up matrix
    """
    from collections import OrderedDict

    mtxDict = OrderedDict([
        ('grid', mtx[:n_grid, :n_grid]),
        ('gal', mtx[-n_gal:, -n_gal:]),
        ('grid_by_gal', mtx[:n_grid, :n_gal])
    ])

    assert np.array_equal(mtxDict['grid'].shape, [n_grid, n_grid])
    return mtxDict


def kern_type_order():
    return [['kk', 'kg1', 'kg2'],
            ['kg1', 'g1g1', 'g1g2'],
            ['kg2', 'g1g2', 'g2g2']
            ]


def print_glue_order():
    """This prints the order for which different matrices are glued together
    in `stitch_matrix`
    """
    all_rows = []
    mtx_type = ['grid', 'grid_by_gal']
    for kern_type in kern_type_order():
        left_mtx = np.hstack([k + '_' + mtx_type[0] for k in kern_type])
        right_mtx = np.hstack([k + '_' + mtx_type[1] for k in kern_type])
        glued_mtx = np.hstack([left_mtx, right_mtx])
        all_rows.append(glued_mtx)

    mtx_type = ['grid_by_gal', 'gal']
    for kern_type in kern_type_order():
        left_mtx = np.hstack([k + '_' + mtx_type[0] + ".T" for k in kern_type])
        right_mtx = np.hstack([k + '_' + mtx_type[1] for k in kern_type])
        glued_mtx = np.hstack([left_mtx, right_mtx])
        all_rows.append(glued_mtx)

    cov_kernels = np.vstack(all_rows)
    print (cov_kernels)


def stitch_matrix(CovMtx, verbose=True):
    """
    :dictionary:
    :returns: a symmetric matrix that has the following layout,
        each submatrix is also symmetric

          |    n_grid         |  ngals            |
          =========================================
          || kk | kg1  | kg2  |                   |
    ngrid ||    | g1g1 | g1g2 |                   |
          ||           | g2g2 |                   |
    ------||--------------------------------------
    ngals ||                  | kk  | kg1  | kg2  |
          ||                  |     | g1g1 | g1g2 |
          ||                  |            | g2g2 |
            ---------------------------------------
    """

    all_rows = []
    mtx_type = ['grid', 'grid_by_gal']
    for kern_type in kern_type_order():
        # Upper left submatrix
        left_mtx = np.hstack([CovMtx[k][mtx_type[0]] for k in kern_type])
        right_mtx = np.hstack([CovMtx[k][mtx_type[1]] for k in kern_type])
        glued_mtx = np.hstack([left_mtx, right_mtx])
        all_rows.append(glued_mtx)

    mtx_type = ['grid_by_gal', 'gal']
    for kern_type in kern_type_order():
        # Lower left submatrix
        left_mtx = np.hstack([CovMtx[k][mtx_type[0]].transpose()
                              for k in kern_type])
        # Lower right submatrix
        right_mtx = np.hstack([CovMtx[k][mtx_type[1]] for k in kern_type])
        glued_mtx = np.hstack([left_mtx, right_mtx])
        all_rows.append(glued_mtx)

    cov_kernels = np.vstack(all_rows)

    if verbose:
        print_glue_order()

    return cov_kernels


def set_up_fixture(l_sq, gp_err, gp_err_prec):
    """create one possible fixture
    """
    x_grid = np.array([[2.5, 2.5],
                       [7.5, 2.5],
                       [2.5, 7.5],
                       [7.5, 7.5]])
    x_galaxies = np.array([[1., 2.],
                           [4., 7.]])

    n_grid = x_grid.shape[0]
    n_gal = x_galaxies.shape[0]

    x_coords = np.concatenate([x_grid, x_galaxies])

    GravLensFieldKernels = OrderedDict(
        [('kk', KappaKappaExpSquaredKernel),
         ('kg1', KappaGamma1ExpSquaredKernel),
         ('kg2', KappaGamma2ExpSquaredKernel),
         ('g1g1', Gamma1Gamma1ExpSquaredKernel),
         ('g1g2', Gamma1Gamma2ExpSquaredKernel),
         ('g2g2', Gamma2Gamma2ExpSquaredKernel)]
    )

    return x_coords, n_grid, n_gal, GravLensFieldKernels


def main(l_sq, gp_err, gp_err_prec):
    """
    :l_sq: float
    :gp_err: float
    :gp_err_prec: float
    :returns: TODO
    """
    x_coords, n_grid, n_gal, GravLensFieldKernels = \
        set_up_fixture(l_sq, gp_err, gp_err_prec)

    CovMtx = \
        {kern_name: cut_mtx_apart(
            output_kernel_value(
                ker, kern_name, x_coords, l_sq, gp_err, gp_err_prec),
            n_grid, n_gal, kern_name
            )
         for kern_name, ker in GravLensFieldKernels.iteritems()}

    return CovMtx


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: ./test_thresher_sampling.py l_sq gp_prec gp_err_prec")

    else:
        CovMtx = main(*[float(arg) for arg in sys.argv[1:]])

    cov_kernels = stitch_matrix(CovMtx)

    express_matrix_in_eigen_format(cov_kernels)
