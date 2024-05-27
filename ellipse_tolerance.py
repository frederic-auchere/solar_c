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


outpath = r"C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics"

channel = "SW"


if channel == "SW":
    a = 1 / 730.504
    b = 1 / 730.941
    c = 1035.820
    decx = 0.0
    decy = 0.0
    xmin = 0
    xmax = 17.4
    # #SW along -Y
    ymin = -17.4
    ymax = 17.4
else:
    a = 1 / 845.139
    b = 1 / 846.299
    c = 1356.579
    decx = 0.0
    decy = 0.0
    xmin = -17.4
    xmax = 0
    #LW along +Y
    ymin = -17.4
    ymax = 17.4


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

uncertainty = 1 + 0.1/100.0  # 0.1%

ellipse1_z = ellipse_sag((x, y), a, b, c, 0, decx, decy)  # reference ellipsoid
ellipse2_z = ellipse_sag((x, y), a*uncertainty, b*uncertainty, c, 0, decx, decy)  # perturbed ellipsoid


delta = (ellipse1_z - ellipse2_z)*1e6
fig = plt.figure(figsize=(5, 8))
plt.imshow(delta, extent=[xmin, xmax, ymin, ymax], origin='lower', vmin=delta[good].min(), vmax=delta[good].max())
plt.colorbar(label='sag difference (nm)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title("Solar-C - " + channel)
circle = plt.Circle((0, 0), rmax, color='red', fill=False)
fig.axes[0].add_patch(circle)
fig.savefig(os.path.join(outpath, 'ellipse_uncertainty_sag_' + channel + '.png'), format='png', transparent=False)
