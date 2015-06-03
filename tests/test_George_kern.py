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
def TwoD_test_data_w_WhiteKernel(truth=[1., 0.3, 1e-2],
                                 data_pts_no_per_side=5):
    kernels = (ExpSquaredKernel, WhiteKernel)

    return fit.generate_2D_data(truth, data_pts_no_per_side, kernels,
                                noise_amp=truth[2])

def test_kernel_values(params=[1, .3, 1e-2], coords=None):
    if coords is None:
        coords = fit.make_grid((0, 1), 5)

    gp = george.GP(params[0] *
                   ExpSquaredKernel(np.ones(2) * params[1], ndim=2) +
                   WhiteKernel(params[2] ** 2, ndim=2),
                   mean=0)

    gp.compute(coords, yerr=0.)
    george_kernel_matrix = gp.get_matrix(coords)

    py_ExpSq = ExpSq(*params)
    py_kernel = py_ExpSq.get_kernel(coords)

    assert np.allclose(py_kernel, george_kernel_matrix)

    return

def test_ln_likelihood1(truth=[1., .3, 1e-2]):
    coords, psi = TwoD_test_data_w_WhiteKernel(truth)

    kernels = [ExpSquaredKernel, WhiteKernel]
    lnlikelihood = fit.lnlike_gp(np.log(truth), kernels, coords, psi)

    py_ExpSq = ExpSq(*truth)
    py_kernel = py_ExpSq.get_kernel(coords)
    py_ln_likelihood_val = py_ExpSq.ln_likelihood(coords, psi)

    assert py_ln_likelihood_val == lnlikelihood
    return

def test_ln_likelihood2(truth=[1., .3, 1e-2]):
    coords = fit.make_grid((0, 1), 5)
    py_ExpSq = ExpSq(*truth)
    psi = py_ExpSq.draw_sample(coords)

    py_kernel = py_ExpSq.get_kernel(coords)
    py_ln_likelihood_val = py_ExpSq.ln_likelihood(coords, psi)

    kernels = [ExpSquaredKernel, WhiteKernel]
    lnlikelihood = fit.lnlike_gp(np.log(truth), kernels, coords, psi)

    assert py_ln_likelihood_val == lnlikelihood
    return

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
