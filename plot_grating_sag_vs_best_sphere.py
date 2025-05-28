import os
import numpy as np
from optical import sw_substrate, lw_substrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

outpath = r"C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics"

fig, axes = plt.subplots(2, 3, figsize=(13, 12))

for name, substrate, ax_row in zip(('SW', 'LW'), (sw_substrate, lw_substrate), axes):

    substrate_z = substrate.sag()
    delta = substrate.sag_from_sphere() * 1e6
    dl = substrate.interferogram()

    titles = [f"{name} z sag", f"from best sphere (R={substrate.best_sphere.r:3.2f} mm)", "Simulated interferogram"]
    clabels = ["sag [mm]", "sag difference [nm]", "intensity"]

    for ax, data, title, clabel in zip(ax_row, (substrate_z, delta, dl), titles, clabels):
        v_min, v_max = np.nanpercentile(data[~data.mask], [1, 99])
        im = ax.imshow(data, extent=substrate.aperture.limits, origin='lower', vmin=v_min, vmax=v_max, cmap='hsv')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, label=clabel)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(title)

plt.tight_layout()

fig.savefig(os.path.join(outpath, 'rectangular_best_sphere.png'), transparent=False, dpi=300)
