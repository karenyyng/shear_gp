"""Tests for understanding George 's computation of the likelihood function."""
from __future__ import (division, print_function)
import pytest
import sys
sys.path.append("../")
import sample_and_fit_gp as fit
import numpy as np
import george
from george.kernels import ExpSquaredKernel, WhiteKernel
from GP import ExpSquared as ExpSq


@pytest.fixture
def TwoD_test_data_w_WhiteKernel(truth=[1., 0.3, 0.3, 1e-2],
                                 data_pts_no_per_side=2):
    kernels = (ExpSquaredKernel, WhiteKernel)

    return fit.generate_2D_data(truth[:2], data_pts_no_per_side, kernels,
                                noise_amp=truth[3])


def test_kernel_matrix_values(params=[1, .3, .3, 1e-2], coords=None):
    """
    This tests if the kernel matrix values are as consistent with
    the mathematical expression

    :math: \lambda^{-1} * \exp(r^2 / (2 * l_sq)) + noise_amp^2 I
    """
    if coords is None:
        coords = fit.make_grid((0, 1), 2)

    kernels = [ExpSquaredKernel, WhiteKernel]
    gp = \
        fit.construct_gp_for_ExpSqlike_and_white_kernels(
            kernels, params)

    gp.compute(coords, yerr=0.)

    george_kernel_matrix = gp.get_matrix(coords)

    py_ExpSq = ExpSq(*params)
    py_kernel = py_ExpSq.get_kernel(coords)

    assert np.array_equal(py_kernel, george_kernel_matrix)

    return


def test_ln_likelihood1(truth=[1., .3, .3, 1e-2]):
    coords, psi = TwoD_test_data_w_WhiteKernel(truth)

    kernels = [ExpSquaredKernel, WhiteKernel]
    gp = \
        fit.construct_gp_for_ExpSqlike_and_white_kernels(
            kernels, truth)

    # lnlike_gp actually changes the GP object to new param values
    lnlikelihood = fit.lnlike_gp(np.log((truth[0], truth[1], truth[2],
                                         truth[3] ** 2)), gp, coords, psi)

    py_ExpSq = ExpSq(*truth)
    py_kernel = py_ExpSq.get_kernel(coords)
    data_fit_term, complexity_penality, const = \
        py_ExpSq.ln_likelihood(coords, psi, separate_terms=True)

    py_ln_likelihood_val = np.sum([data_fit_term, complexity_penality, const])

    assert gp._const == complexity_penality + const
    assert py_ln_likelihood_val == lnlikelihood
    return


def test_ln_likelihood2(truth=[1., .3, .3, 1e-2]):
    coords = fit.make_grid((0, 1), 5, regular=True)
    py_ExpSq = ExpSq(*truth)
    psi = py_ExpSq.draw_sample(coords)

    py_kernel = py_ExpSq.get_kernel(coords)
    py_ln_likelihood_val = py_ExpSq.ln_likelihood(coords, psi)

    kernels = [ExpSquaredKernel, WhiteKernel]
    gp = \
        fit.construct_gp_for_ExpSqlike_and_white_kernels(
            kernels, truth)
    # lnlike_gp actually changes the GP object to new param values
    lnlikelihood = fit.lnlike_gp(np.log((truth[0], truth[1], truth[2],
                                         truth[3] ** 2)), gp, coords, psi)

    assert gp._const == complexity_penality + const
    assert py_ln_likelihood_val == lnlikelihood
    return
#
# def test_ln_likelihood3(truth=[1., .3, 1e-2]):
#     coords = fit.make_grid((0, 1), 5, regular=True)
#     py_ExpSq = ExpSq(*truth)
#     psi = py_ExpSq.draw_sample(coords)
#
#     py_kernel = py_ExpSq.get_kernel(coords)
#     py_ln_likelihood_val = py_ExpSq.ln_likelihood(coords, psi)
#
#     kernels = [ExpSquaredKernel, WhiteKernel]
#     lnlikelihood = fit.lnlike_gp(np.log(truth), kernels, coords, psi)
#
#     assert py_ln_likelihood_val == lnlikelihood
#     return
#
# def test_beta_ln_likelihood():
#     return
#
#
# def test_grad_ln_likelihood():
#     return
#
#
# def test_grad_ln_transformed_likelihood():
#     return
#
#
# def test_ln_transformed_likelihood():
#     return
