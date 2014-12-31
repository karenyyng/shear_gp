from sample_and_fit_gp import invgamma_pdf
from scipy.stats import invgamma  # scipy library for checking function
import numpy as np


def test_invgamma_pdf():
    alphas = np.arange(1., 5., .5)
    x = np.arange(0.01, 5.0, 0.1)
    for alpha in alphas:
        assert np.allclose(invgamma_pdf(x, alpha=alpha, beta=1.),
                           invgamma.pdf(x, a=alpha)), \
            "something went wrong with the invgamma_pdf function"
    return


if __name__ == "__main__":
    test_invgamma_pdf()
