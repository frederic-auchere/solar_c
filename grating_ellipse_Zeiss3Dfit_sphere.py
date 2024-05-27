#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:22:37 2019

@author: fauchere
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from mpl_toolkits.axes_grid1 import make_axes_locatable

def rotationmatrix(angle, axis):
    """
    Returns a rotation matrix about the specified axis (z=0, y=1, x=2) for the
    specififed angle (in radians).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)

    if axis == 0:  # Rz
        matrix = np.array([[ cos, sin,    0],
                           [-sin, cos,    0],
                           [   0,   0,    1]])
    elif axis == 1:  # Ry
        matrix = np.array([[ cos,   0,  sin],
                           [   0,   1,    0],
                           [-sin,   0,  cos]])
    elif axis == 2:  # Rx
        matrix = np.array([[   1,   0,    0],
                           [   0, cos, -sin],
                           [   0, sin,  cos]])

    return matrix

def transform(x, y, z, matrix):
    xyz = np.stack((x.ravel(), y.ravel(), z.ravel()))
    nx, ny, nz = np.matmul(matrix, xyz)
    return nx.reshape(x.shape), ny.reshape(x.shape), nz.reshape(x.shape)

def ellipse_min(params, x, y, surface):
    R1, R2, R3, dx, dy, dz, e1, e2, e3 = params
    if R1 <= 0 or R2 <=0 or R3 <= 0:
        return 1e10
    delta = surface - ellipse_sag((x, y), R1, R2, R3, dx, dy, dz, e1, e2, e3)
    return np.mean(delta**2)

def sphere_min(params, x, y, surface):
    rc, dx, dy, dz = params
    if rc <= 0:
        return 1e10
    delta = surface - sphere_sag((x, y), rc, dx, dy, dz)
    return np.sqrt(np.mean(delta**2))

def sphere_sag(XY, rc, dx, dy, dz):
    x, y = XY

    r = np.sqrt((x-dx)**2 + (y-dy)**2)

    c = 1/rc
    return dz + (c*r**2)/(1 + np.sqrt(1 - (c*r)**2))

def ellipse_sag(XY, R1, R2, R3, dx, dy, dz, e1, e2, e3):
    x, y = XY

    u2 = ((x-dx)/R1)**2 + ((y-dy)/R2)**2
    if u2.max() > 1: return 1e10
    z = dz + R3*u2/(1 + np.sqrt(1 - u2))

    Rx = rotationmatrix(e1, 2)
    Ry = rotationmatrix(e2, 1)
    Rz = rotationmatrix(e3, 0)
    R = Rx @ Ry @ Rz

    x, y, z = transform(x, y, z, R)

    return z + dz

outpath = "C:\\Users\\fauchere\\Documents\\01-Projects\\02-Space\\Solar C\\EPSILON\\Optics\\"

channel = "LW"

if channel == "SW":
    R1 = 730.5044
    R2 = 730.9414
    R3 = 1035.82
    R = R3# mm
    decx = 24.366
    decy = 15.031
    ymin = -decy-20
    ymax = -decy+20
    #SW along -Y
    xmin = -decx
    xmax = -decx + 20
else:
    R1 = 845.139
    R2 = 846.299
    R3 = 1356.579
    decx = 0
    decy = 2.0
    ymin = -decy-20
    ymax = -decy+20
    #SW along -Y
    xmin = decx - 20
    xmax = decx

dx = 1.0
dy = 1.0
nx = np.round((xmax - xmin)/dx)
ny = np.round((ymax - ymin)/dy)

yy, xx = np.indices((int(ny), int(nx)))
xx = xx*dx
xx += xmin
yy = yy*dy
yy += ymin

xy_uncertainty = 1e-2  # [mm]
z_uncertainty = 2e-3  # [mm]

ntrials = 1000
nominal_parameters = (R1, R2, R3, 0, 0, 0, 0, 0, 0)
sphere_nominal_parameters = (528, 0, 0, 0)
names = ("$R$", "dx", "dy", "dz")
labels = ("relative error",
         "error", "error", "error")
units = ("[%]",
         "[mm]", "[mm]", "[mm]",
         "[arcsec]", "[arcsec]", "[degrees]")
limits = [(-5, 5), (-0.2, 0.2), (-0.2, 0.2), (-0.01, 0.01)]
n_parameters = len(nominal_parameters)
errors = []
for p in sphere_nominal_parameters:
    errors.append(np.zeros(ntrials, dtype=float))

nominal_ellipse_z = ellipse_sag((xx, yy), *nominal_parameters)

bounds = Bounds([0.9*R, -0.2, -0.2, -0.2],
                [1.1*R, 0.2, 0.2, 0.2])

for i in range(ntrials):
    measured_x = xx + np.random.normal(0, xy_uncertainty, xx.shape)
    measured_y = yy + np.random.normal(0, xy_uncertainty, yy.shape)
    measured_z = nominal_ellipse_z + np.random.normal(0, z_uncertainty, nominal_ellipse_z.shape)
    mini = minimize(sphere_min,
                    sphere_nominal_parameters,
                    args=(measured_x, measured_y, measured_z),
                    method='Powell',
                    # bounds=bounds
                    )
    for error, m, p in zip(errors, mini.x, sphere_nominal_parameters):
        error[i] = m #- p

# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# for ax, error, p, name, label, unit, limit in zip(axes.flatten(), errors, sphere_nominal_parameters, names, labels, units, limits):
#     if "R" == name:
#         error = 100*error/p
#
#     ax.hist(error, label="$\sigma$={:.2f} {}".format(np.std(error), unit), bins=10)
#     ax.set_title(name)
#     ax.set_xlabel(label + " " + unit)
#     ax.set_xlim(*limit)
#     ax.legend(frameon=False)
#
# plt.tight_layout()

#fig.savefig(outpath + 'substrate_3Dfit_1x1mm_2micronsZ_' + channel + '.png', format='png')
