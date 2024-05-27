#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:22:37 2019

@author: fauchere
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def erosion_min(coeffs, surface):
    dx1, dy1, *coeffs1 = coeffs
    delta = surface - poly_sag((x, y), dx1, dy1, coeffs1)
    return np.mean(delta**2)

def sphere_min(coeffs, *args):
    x, y, r = args
    rc, d = coeffs
    if rc <= 0:
        return 1e10
    delta = conic_sag((x, y), r, 0, 0, 0, -1) - conic_sag((x, y), rc, d, 0, 0, 0)
    return np.mean(delta**2)

def conic_sag(XY, rc, d, dx, dy, k):
    x, y = XY
    
    r2 = (x-dx)**2 + (y-dy)**2
        
    c = 1/rc
    return d + (c*r2)/(1 + np.sqrt(1 - (1+k)*r2*c*c))

def poly_sag(XY, dx1, dy1, coeffs1):
    x, y = XY
    r1 = np.sqrt((x - dx1)**2 + (y - dy1)**2)
    p = np.zeros_like(x)
    for d, c in enumerate(coeffs1):
        p += c*r1**d
    return p

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

c = 1518.067
decx = 80
decy = 0
ymin = -27
ymax = 27
#SW along -Y
xmin = 80 - 27
xmax = 80 + 27

rc = c
d = 0.0

dx = 0.5
dy = 0.5
nx = 1 + np.round((xmax - xmin)/dx)
ny = 1 + np.round((ymax - ymin)/dy)

y, x = np.indices((int(ny), int(nx)))
x = x*dx
x += xmin
y = y*dy
y += ymin

#torus_z = torus_sag((x, y), rh, rv, d, decx, decy)
hri_z = conic_sag((x, y), c, 0, 0, 0, -1)  # reference ellipsoid
mini = minimize(sphere_min, [rc, d], args=(x, y, c), method='Powell')

fit = conic_sag((x, y), *mini.x, 0, 0, 0)

delta_sphere = (hri_z - fit)*1e6

mini_erosion = minimize(erosion_min, [0, 0, 0, 0, 0, 0, 0, 0, 0], args=delta_sphere, method='Powell')
fit_erosion = poly_sag((x, y), *mini_erosion.x[0:2], mini_erosion.x[2:])
delta = (delta_sphere - fit_erosion)
print(delta.max() - delta.min(), delta.std(), delta.min(), delta.max(), mini_erosion.x)


# delta2 = (ellipse_z - fit2)*1e6
# print(delta2.max() - delta2.min(), delta2.std(), delta2.min(), delta2.max(), mini2.x)
# delta = delta2

wvl = 632.0

dl = (1 + np.cos(2*np.pi*2*delta/wvl))/2

titles = ["HRI z-sag from best sphere", "Simulated interferogram"]
clabels = ["sag difference (nm)", "intensity"]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, data, title, clabel in zip(axes, (delta, dl), titles, clabels):
    im = ax.imshow(data, extent=[xmin, xmax, ymin, ymax], origin='lower', )    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, label=clabel)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(title)
    circ = plt.Circle((-decx, -decy), 20, color="red", fill=False)
    ax.add_patch(circ)

fig.savefig(outpath + 'hri_sag.png', format='png', transparent=False)
