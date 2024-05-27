import os
import numpy as np
from optics.surfaces import Standard
from optical import rectangular_sw_substrate, rectangular_lw_substrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
 
outpath = r"C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics"

reference_surface = Standard(np.inf, 0)
margin = -2, 2, -2, 2

fig, axes = plt.subplots(2, 3, figsize=(13, 12))

for name, substrate, ax_row in zip(('SW', 'LW'), (rectangular_sw_substrate, rectangular_lw_substrate), axes):

    xc, yc = substrate.best_sphere.dx, substrate.best_sphere.dy
    limits = xc + margin[0], xc + margin[1], yc + margin[2], yc + margin[3]
    grid = substrate.grid(200, 200, limits=limits)
    substrate_z = substrate.sag(grid)
    delta = substrate_z - reference_surface.sag(grid)
    dl = substrate.interferogram(reference_surface=reference_surface, grid=grid)

    titles = [f"{name} z sag", f"from best sphere (R={substrate.best_sphere.r:3.2f} mm)", "Simulated interferogram"]
    clabels = ["sag [mm]", "sag difference [nm]", "intensity"]

    for ax, data, title, clabel in zip(ax_row, (substrate_z, delta, dl), titles, clabels):
        v_min, v_max = np.nanpercentile(data[~data.mask], [1, 99])
        im = ax.imshow(data, extent=limits, origin='lower', vmin=v_min, vmax=v_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, label=clabel)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(title)

plt.tight_layout()

fig.savefig(os.path.join(outpath, 'best_sphere.png'), transparent=False)
