from optics.surfaces import Flat
from optical import rectangular_lw_substrate, spherical_lw_substrate,\
                    rectangular_sw_substrate, spherical_sw_substrate
import matplotlib.pyplot as plt

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

half_width = 2

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for substrate, spherical_substrate, ax, title in zip((rectangular_lw_substrate, rectangular_sw_substrate),
                                                     (spherical_lw_substrate, spherical_sw_substrate),
                                                      axes,
                                                      ('LW', 'SW')):

    dx, dy = spherical_substrate.surface.dx, spherical_substrate.surface.dy

    lw_limits = [dx - half_width, dx + half_width, dy - half_width, dy + half_width]

    interferogram = substrate.interferogram(grid=substrate.grid(nx=400, ny=400, limits=lw_limits),
                                            reference_surface=Flat())
    ax[0].imshow(rebin(interferogram, (40, 40)), origin='lower', extent=lw_limits)
    ax[0].set_title(f'{title} substrate')
    interferogram = spherical_substrate.interferogram(grid=spherical_substrate.grid(nx=400, ny=400, limits=lw_limits),
                                                      reference_surface=Flat())
    ax[1].imshow(rebin(interferogram, (40,40)), origin='lower', extent=lw_limits)
    ax[1].set_title(f'{title} spherical substrate')
