""" Script to run MCMC on data with known input param. values.

Author: Karen Ng <karenyng@ucdavis.edu>
LICENSE: BSD
"""
from __future__ import (division, absolute_import,
                        print_function, unicode_literals)

import george
assert george.__version__ == '0.2.1 experimental', \
    "Wrong version of George is being used. \n" + \
    "Please use GitHub version from https://github.com/karenyyng/george"

import pandas as pd
from george.kernels import (KappaKappaExpSquaredKernel,
                            Gamma1Gamma1ExpSquaredKernel,
                            KappaGamma1ExpSquaredKernel,
                            ExpSquaredKernel,
                            WhiteKernel)
import sys
sys.path.append("../")
import sample_and_fit_gp as fit
import diag_plots as plot

# --------- initialization of parameters ------------------------------
# want to be able to parse arguments from a file later on
if len(sys.argv) == 4:
    inv_lambda, l_sq, noise_amp = sys.argv[:1]
else:
    print ("No param values supplied, using default values of \n" +
           "inv_lambda, l_sq, noise_amp = 1., .5, 1e-2")
    inv_lambda, l_sq, noise_amp = 1., .5, 1e-2

truth = (inv_lambda, l_sq)
data_pt_nos_per_side = 5
kernels = (KappaKappaExpSquaredKernel, WhiteKernel)
rng = (0., 1.)
regular_grid = True

# -----------generate data --------------------------------------------
coords, psi = \
    fit.generate_2D_data(truth, data_pt_nos_per_side, kernels=kernels,
                         rng=rng, noise_amp=noise_amp,
                         white_kernel_as_nugget=True,
                         regular_grid=regular_grid)


# -------------------------------------------------------------
