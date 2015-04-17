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


def test_Cython_kappakappa_2_coords():
    coords = np.array([[1., 2.], [4., 7.]])
    beta = 1.

    cythonGP = george.GP(KappaKappaExpSquaredKernel(beta, ndim=2L))
    cythonCov = cythonGP.get_matrix(coords)

    pythonCov = plotDerivCov(KKker, coords, beta)
    assert np.array_equal(cythonCov, pythonCov)
    return cythonCov


def test_Cython_kappakapp_10_coords():
    beta = 1.
    coords = np.array([[1, i] for i in np.arange(0.1, 1.1, 0.1)])
    cythonGP = george.GP(KappaKappaExpSquaredKernel(beta, ndim=2L))
    cythonCov = cythonGP.get_matrix(coords)

    pythonCov = plotDerivCov(KKker, coords, beta)
    assert np.array_equal(cythonCov, pythonCov)

if __name__ == "__main__":
    cythonCov = test_Cython_kappakappa_2_coords()
