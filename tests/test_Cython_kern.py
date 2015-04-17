from __future__ import (print_function, division, unicode_literals)
from george.kernels import (KappaKappaExpSquaredKernel,
                            KappaGamma1ExpSquaredKernel,
                            KappaGamma2ExpSquaredKernel,
                            Gamma1Gamma1ExpSquaredKernel,
                            Gamma2Gamma2ExpSquaredKernel,
                            Gamma1Gamma2ExpSquaredKernel)
import george
import numpy as np
import sys
sys.path.append("../")
from kern_deriv import KappaKappaExpSquaredKernel as KKker
from kern_deriv import KappaGamma1ExpSquaredKernel as KG1ker
from kern_deriv import KappaGamma2ExpSquaredKernel as KG2ker
from kern_deriv import Gamma1Gamma1ExpSquaredKernel as G1G1ker
from kern_deriv import Gamma2Gamma2ExpSquaredKernel as G2G2ker
from kern_deriv import Gamma1Gamma2ExpSquaredKernel as G1G2ker
from plot_kern_deriv import plotDerivCov

# --------------------- starting to test Cython Kernel values --------


def test_Cython_kappakappa_2_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1., 2.], [4., 7.]])
    beta = 1.

    cythonGPs = [george.GP(KappaKappaExpSquaredKernel(beta, ndim=2L))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotDerivCov(KKker, coords, beta)
                 for beta in betas]

    for i in range(len(betas)):
        assert np.array_equal(cythonCov[i], pythonCov[i])

    return


def test_Cython_kappakappa_10_coords_vary_beta():
    betas = np.arange(0.1, 1.0, 0.1)
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])

    cythonGPs = [george.GP(KappaKappaExpSquaredKernel(beta, ndim=2L))
                 for beta in betas]
    cythonCov = [cythonGP.get_matrix(coords)
                 for cythonGP in cythonGPs]

    pythonCov = [plotDerivCov(KKker, coords, beta)
                 for beta in betas]

    for i in range(len(betas)):
        assert np.array_equal(cythonCov[i], pythonCov[i])


if __name__ == "__main__":
    cythonCov = test_Cython_kappakappa_2_coords()
