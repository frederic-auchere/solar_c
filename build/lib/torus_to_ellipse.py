import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from scipy.optimize import minimize


def ellipse_min(coeffs, *args):
    x, y, rh, rv, decx, decy = args
    a, b, c, d, dx, dy = coeffs
    delta = torus_sag((x, y), rh, rv, decx, decy) - ellipse_sag((x, y), a, b, c, d, dx, dy)
    return np.mean(delta**2)


def ellipse_sag(XY, a, b, c, d, dx, dy):
    x, y = XY

    u2 = (a*(x-dx))**2 + (b*(y-dy))**2
    if u2.max() > 1:
        return 1e20
    else:
        return d + c*u2/(1 + np.sqrt(1 - u2))


def torus_sag(XY, rh, rv, dx, dy):
    x, y = XY

    if rh <= 0 or rv <= 0:
        return 1e10

    c = 1/rh
    if ((c*(y - dy)**2) > 1).any():
        return 1e20
    zy = c*(y - dy)**2/(1 + np.sqrt(1 - (c*(y - dy))**2))
    return rv - np.sqrt((rv - zy)**2 - (x - dx)**2)


outpath = r"C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics"

channel = "LW"

if channel == "SW":
    rh = 526.516
    rv = 524.506
    decx = 0.0
    decy = 0.0
    xmin = 0
    xmax = 17.4
    # #SW along -Y
    ymin = -17.4
    ymax = 17.4
else:
    rh = 526.516
    rv = 524.506
    decx = 0.0
    decy = 0.0
    xmin = -17.4
    xmax = 0
    #LW along +Y
    ymin = -17.4
    ymax = 17.4

a = 1 / rh
b = 1 / rh
c = rh

d = 0.0
rmax = 17.4

dx = 0.1
dy = 0.1
nx = int(1 + np.round((xmax - xmin)/dx))
ny = int(1 + np.round((ymax - ymin)/dy))

x, y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
r = np.sqrt(x**2 + y**2)
good = r < rmax
valid_x = x[good]
valid_y = y[good]

#torus_z = torus_sag((x, y), rh, rv, d, decx, decy)
torus_z = torus_sag((x, y), rh, rv, decx, decy)  # reference ellipsoid


mini = minimize(ellipse_min, [a, b, c, 0, decx, decy], args=(valid_x, valid_y, rh, rv, decx, decy), method='Powell')
fit = ellipse_sag((x, y), *mini.x)

delta = (torus_z - fit)*1e6
fig = plt.figure(figsize=(5, 8))
plt.imshow(delta, extent=[xmin, xmax, ymin, ymax], origin='lower', vmin=delta[good].min(), vmax=delta[good].max())
plt.colorbar(label='sag difference (nm)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title("Solar-C - " + channel)
circle = plt.Circle((0, 0), rmax, color='red', fill=False)
fig.axes[0].add_patch(circle)
fig.savefig(os.path.join(outpath, 'torus-ellipse_sag_' + channel + '.png'), format='png', transparent=False)

print(delta[good].max() - delta[good].min(), delta[good].std(), delta.min(), delta.max(), mini.x)
