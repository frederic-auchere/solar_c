import os
from optics import sw_substrate, lw_substrate
from optics.surfaces import Toroidal, Standard
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

outpath = r"C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics"

fig, axes = plt.subplots(2, 2, figsize=(9, 12))

for name, substrate, ax_row in zip(('SW', 'LW'), (sw_substrate, lw_substrate), axes):

    x, y, = substrate.meshgrid()
    ellipse_z = substrate.sag(x, y)

    surface_names = ['toroidal', 'conic']
    for ax, surface_type, surface_name in zip(ax_row, (Toroidal, Standard), surface_names):
        best_surface = substrate.find_best_surface(surface_type, tilt=True)
        diff = (substrate.sag(x, y) - best_surface.sag(x, y))*1e6
        im = ax.imshow(diff, extent=[substrate.aperture.dx - substrate.aperture.x_width / 2,
                                     substrate.aperture.dx + substrate.aperture.x_width / 2,
                                     substrate.aperture.dy - substrate.aperture.y_width / 2,
                                     substrate.aperture.dy + substrate.aperture.y_width / 2], origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, label='sag [nm]')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        if surface_type is Toroidal:
            parameters = f'Rc={best_surface.rc:6.3f} Rr={best_surface.rr:6.3f}'
        else:
            parameters = f'R={best_surface.r:6.3f} k={best_surface.k:4.3f}'
        ax.set_title(f'{name} vs. {surface_name} - {np.std(diff):4.2f} rms\n{parameters}')

plt.tight_layout()

fig.savefig(os.path.join(outpath, 'ellipse_vs_conic_vs_toroidal.png'), transparent=False)