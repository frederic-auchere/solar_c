#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:33:30 2018

@author: fauchere
"""


import numpy as _np
from scipy.optimize import curve_fit

__all__ = ["polyval", "polyfit2d", "sfit"]

#def polyfit2d(x, y, z, order):
#    """
#    From
#    https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
#    """
#    ncols = (order + 1)**2
#    G = _np.zeros((x.size, ncols))
#    ij = itertools.product(range(order+1), range(order+1))
#    for k, (i, j) in enumerate(ij):
#        G[:, k] = x**i * y**j
#    m, _, _, _ = _np.linalg.lstsq(G, z, rcond=-1)
#    return m
#
#
#def polyval2d(x, y, m):
#    """
#    From
#    https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
#    """
#    order = int(_np.sqrt(len(m))) - 1
#    ij = itertools.product(range(order+1), range(order+1))
#    z = _np.zeros_like(x)
#    for a, (i, j) in zip(m, ij):
#        z += a * x**i * y**j
#    return z

def polyval(x, y, coefficients):
    """
    Evaluates bivariate polynomial at points (x, y)
    For some reason faster than numpy.polynomial.polyval2d
    """
    degree = coefficients.shape[0]-1
    poly = 0
    for j in range(degree, -1, -1):
        dum = 0
        for i in range(degree, -1, -1):
            dum *= x
            dum += coefficients[i, j]
        poly *= y
        poly += dum
    return poly


def polyfit2d(x, y, f, deg, maxdegree=False):
    """
    From
    https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    """
    from numpy.polynomial import polynomial
    vander = polynomial.polyvander2d(x, y, [int(deg), int(deg)])
    vander = vander.reshape((-1, vander.shape[-1]))
    if maxdegree is True:
        # the summ of indices gives the combined degree for each coefficient,
        # which is then compared to maxdegree if not None
        dy, dx = _np.indices((deg+1, deg+1))
        vander[:, (dx.reshape(-1) + dy.reshape(-1)) > deg] = 0
    c, _, _, _ = _np.linalg.lstsq(vander,
                                  f.reshape((vander.shape[0],)),
                                  rcond=-1)
    return c.reshape((deg+1, deg+1))


def sfit(z, degree, missing=None, maxdegree=False):
    """
    Mimics the behavior of the IDL sfit function
    z: array of data points
    degree: degree (on one axis, see maxdegree) of the fitted polynomial
    maxdegree: if True, degree represents the maximum degree of the fitting
               polynomial of all dimensions combined, rather than the maximum
               degree of the polynomial in a single variable.
    """
#    from numpy.polynomial import polynomial
    y, x = _np.indices(z.shape, dtype=float)
    y /= z.shape[0]-1
    y -= 0.5
    x /= z.shape[1]-1
    x -= 0.5

    good = z != missing if missing is not None else Ellipsis

    m = polyfit2d(x[good], y[good], z[good], degree, maxdegree=maxdegree)
#    return polynomial.polyval2d(x, y, m), m
    return polyval(x, y, m), m


def gauss2d(X, a, u0, v0, sx, sy, theta, offset):

    u, v = X
    u = u.astype(float)
    v = v.astype(float)
    u -= float(u0)
    v -= float(v0)
    xp = u*_np.cos(theta) - v*_np.sin(theta)
    yp = u*_np.sin(theta) + v*_np.cos(theta)

    return offset + a*_np.exp(-((xp/(_np.sqrt(2)*sx))**2 +
                                (yp/(_np.sqrt(2)*sy))**2))


def gauss2d_2(X,
              a, u0, v0, sx, sy, theta, offset,
              a2, u02, v02, sx2, sy2, theta2):

    u, v = X
    u = u.astype(float)
    v = v.astype(float)
    u -= float(u0)
    v -= float(v0)
    xp = u*_np.cos(theta) - v*_np.sin(theta)
    yp = u*_np.sin(theta) + v*_np.cos(theta)

    g1 = a*_np.exp(-((xp/(_np.sqrt(2)*sx))**2 +
                              (yp/(_np.sqrt(2)*sy))**2))

    u2, v2 = X
    u2 = u2.astype(float)
    v2 = v2.astype(float)
    u2 -= float(u02)
    v2 -= float(v02)
    xp2 = u2*_np.cos(theta2) - v2*_np.sin(theta2)
    yp2 = u2*_np.sin(theta2) + v2*_np.cos(theta2)

    g2 = a2*_np.exp(-((xp2/(_np.sqrt(2)*sx2))**2 +
                                (yp2/(_np.sqrt(2)*sy2))**2))

    return offset + g1 + g2


def gauss_2dfit(data, guess):
    u, v = _np.indices(data.shape)
    p, cov = curve_fit(gauss2d,
                       (u.flatten(), v.flatten()),
                       data.flatten(),
                       guess)
    return p, gauss2d((u, v), *p)


def gauss_2dfit2(data, guess):
    u, v = _np.indices(data.shape)
    p, cov = curve_fit(gauss2d_2,
                       (u.flatten(), v.flatten()),
                       data.flatten(),
                       guess)
    return p, gauss2d_2((u, v), *p)
