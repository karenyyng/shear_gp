"""Tests for understanding George 's computation of the likelihood function."""
from __future__ import (division, print_function)
import pytest
import sys
sys.path.append("../")
import sample_and_fit_gp as fit
import numpy as np
from george.kernels import ExpSquaredKernel, WhiteKernel


@pytest.fixture
def TwoD_test_data():
    truth = [1., 0.7]
    data_pts_no_per_side = 2
    kernels = (ExpSquaredKernel, WhiteKernel)

    return fit.generate_2D_data(truth, data_pts_no_per_side, kernels,
                                noise_amp=1e-12)


def py_ExpSquaredKernel(params, coords):
    r_sq = np.matrix([[np.dot(coords[i] - coords[j], coords[i] - coords[j])
                     for j in range(len(coords))]
                     for i in range(len(coords))])
    kernel = params[0] * np.exp(-r_sq / (2. * params[1])) + \
        params[2] ** 2 * np.eye(len(coords))  # nugget term
    return kernel


def py_ln_likelihood(coord, psi, params):
    """
    """

    return


def py_ln_transformed_ln_likelihood():
    return


def test_ln_likelihood():
    return


def test_beta_ln_likelihood():
    return


def test_grad_ln_likelihood():
    return


def test_grad_ln_transformed_likelihood():
    return


def test_ln_transformed_likelihood():
    return
