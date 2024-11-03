# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, M_PI
from scipy.integrate import quad
from scipy.stats import norm

ctypedef np.float64_t DTYPE_t


cdef double norm_pdf(double x, double mu, double sigma) nogil:
    """Fast normal PDF computation."""
    cdef double z = (x - mu) / sigma
    return (1.0 / (sigma * sqrt(2.0 * M_PI))) * exp(-0.5 * z * z)


def compute_portfolio_moments(
    np.ndarray[DTYPE_t, ndim=1] weights,
    np.ndarray[DTYPE_t, ndim=1] returns,
    np.ndarray[DTYPE_t, ndim=2] variance,
) -> tuple[float, float]:
    cdef double port_ret = np.dot(weights, returns)
    cdef double port_vol = sqrt(np.dot(weights.T, np.dot(variance, weights)))
    return port_ret, port_vol


def compute_lpm_integrand(
    double x,
    double tau,
    int power,
    double port_ret,
    double port_vol,
) -> float:
    return (tau - x)**power * norm_pdf(x, port_ret, port_vol)


def compute_cvar_integrand(
    double x,
    double port_ret,
    double port_vol,
) -> float:
    return x * norm_pdf(x, port_ret, port_vol)
