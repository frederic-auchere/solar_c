#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:22:37 2019

@author: fauchere
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def erosion_min(params, x, y, surface):
    dx1, dy1, *coeffs = params
    delta = surface - poly_sag(x, y, dx1, dy1, coeffs)
    return np.sqrt(np.mean(delta**2))

def erosion_min_2passes(params, x, y, surface):
    a, dx1, dy1, dx2, dy2, *coeffs = params
    delta = surface - poly_2passes(x, y, a, dx1, dy1, dx2, dy2, coeffs)
    return np.sqrt(np.mean(delta**2))

# def erosion_min_2passesfree(coeffs, surface):
#     dx1, dy1, dx2, dy2, *coeffs1 = coeffs
#     delta = surface - poly_2passesfree((x, y), dx1, dy1, dx2, dy2, coeffs1)
#     return np.sqrt(np.mean(delta**2))

def sphere_min(params, x, y, surface):
    rc, d, dx, dy = params
    if rc <= 0:
        return 1e10
    delta = surface - sphere_sag((x, y), rc, d, dx, dy)
    return np.sqrt(np.mean(delta**2))

# def ellipse_min(coeffs, surface):
#     a2, c2, d = coeffs
#     if a2 <= 0 or c2 <= 0:
#         return 1e10
#     delta = surface - ellipse_sag((x, y), a2, a2, c2, d, 0, 0)
#     return np.mean(delta**2)

def poly_2passes(x, y, a, dx1, dy1, dx2, dy2, coeffs):
    return poly_sag(x, y, dx1, dy1, coeffs)\
         + poly_sag(x, y, dx2, dy2, coeffs)*a

# def poly_2passesfree(XY, dx1, dy1, dx2, dy2, coeffs1):
#     n = len(coeffs1)//2
#     return poly_sag((x, y), dx1, dy1, coeffs1[0:n])\
#            + poly_sag((x, y), dx2, dy2, coeffs1[n:])

def poly_sag(x, y, dx, dy, coeffs):
    r = np.sqrt((x - dx)**2 + (y - dy)**2)
    poly = np.zeros_like(r)
    for c in coeffs:
        poly *= r
        poly += c
    return poly

def sphere_sag(XY, rc, d, dx, dy):
    x, y = XY
    
    r = np.sqrt((x-dx)**2 + (y-dy)**2)
        
    c = 1/rc
    return d + (c*r**2)/(1 + np.sqrt(1 - (c*r)**2))

def ellipse_sag(XY, a, b, c, d, dx, dy):
    x, y = XY

    u2 = (a*(x-dx))**2 + (b*(y-dy))**2
    if u2.max() > 1: return 1e10
    return d + c*u2/(1 + np.sqrt(1 - u2))

outpath = "C:\\Users\\fauchere\\Documents\\01-Projects\\02-Space\\Solar C\\EPSILON\\Optics\\"

channel = "SW"

if channel == "SW":
    a = 1/732.769
    b = 1/733.217
    c = 1041.96
    decx = 24.366
    decy = 15.031
    ymin = -decy-20
    ymax = -decy+20
    #SW along -Y
    xmin = -decx
    xmax = -decx + 20
else:
    a = 1/845.14
    b = 1/846.29
    c = 1356.58
    decx = 0
    decy = 2.0
    ymin = -decy-20
    ymax = -decy+20
    #SW along -Y
    xmin = decx - 20
    xmax = decx

rc = c
d = 0.0

dx = 0.2
dy = 0.2
nx = np.round((xmax - xmin)/dx)
ny = np.round((ymax - ymin)/dy)

yy, xx = np.indices((int(ny), int(nx)))
xx = xx*dx
xx += xmin
yy = yy*dy
yy += ymin

ellipse_z = ellipse_sag((xx, yy), a, b, c, 0, 0, 0)  # reference ellipsoid
ellipse_z = ma.masked_array(ellipse_z)
ellipse_z.mask = np.sqrt((xx + decx)**2 + (yy + decy)**2) > 20
mini = minimize(sphere_min, [rc, d, 0, 0], args=(xx, yy, ellipse_z), method='Powell')

fit = sphere_sag((xx, yy), *mini.x)

delta_sphere = ellipse_z - fit

mini_erosion = minimize(erosion_min_2passes, [0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0],
                        args=(xx, yy, delta_sphere),
                        method='Powell', options={"ftol":1e-6})
fit_erosion = poly_2passes(xx, yy, *mini_erosion.x[0:5], mini_erosion.x[5:])
delta = delta_sphere - fit_erosion

delta *= 1e6

delta.mask = np.sqrt((xx + decx)**2 + (yy + decy)**2) > 20
print(delta.max() - delta.min(), delta.std(), delta.min(), delta.max(), mini_erosion.x)

wvl = 632.0

dl = (1 + np.cos(2*np.pi*2*delta/wvl))/2

title = channel + " - z-sag from nominal ellipse"
clabel = "sag difference (nm)"

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(delta, extent=[xmin, xmax, ymin, ymax], origin='lower', )    
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(im, cax=cax, label=clabel)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title(title)
circ = plt.Circle((-decx, -decy), 20, color="red", fill=False)
ax.add_patch(circ)

fig.savefig(outpath + 'torus-ellipse_sag_' + channel + '.png', format='png', transparent=False)

