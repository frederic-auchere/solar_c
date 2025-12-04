from optics.surfaces import Flat
from optical import rectangular_lw_substrate, spherical_lw_substrate,\
                    rectangular_sw_substrate, spherical_sw_substrate
import matplotlib.pyplot as plt
import numpy as np
from optical.utils import rebin, parabolic_interpolation


half_width = 1
nx, ny = 200, 200
bnx, bny = 20, 20

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for substrate, spherical_substrate, ax, title in zip((rectangular_lw_substrate, rectangular_sw_substrate),
                                                     (spherical_lw_substrate, spherical_sw_substrate),
                                                      axes,
                                                      ('LW', 'SW')):

    dx, dy = spherical_substrate.surface.dx, spherical_substrate.surface.dy

    limits = [dx - half_width, dx + half_width, dy - half_width, dy + half_width]

    grid = substrate.grid(nx=nx, ny=ny, limits=limits)
    interferogram = substrate.interferogram(grid=grid, reference_surface=Flat())
    sag = substrate.sag(grid)
    x, y = parabolic_interpolation(sag, extremum=np.nanargmin)
    x = limits[0] + (grid[0][0, 1] - grid[0][0, 0]) * x
    y = limits[2] + (grid[1][1, 0] - grid[1][0, 0]) * y
    print(f"{title} measured dx={x:.3f} [mm] dy={y:.3f} [mm]")

    ax[0].imshow(rebin(interferogram, (bnx, bny)), origin='lower', extent=limits)
    ax[0].set_title(f'{title} substrate')
    interferogram = spherical_substrate.interferogram(grid=grid, reference_surface=Flat())
    sag = spherical_substrate.sag(grid)
    x, y = parabolic_interpolation(sag, extremum=np.nanargmin)
    x = limits[0] + (grid[0][0, 1] - grid[0][0, 0] ) * x
    y = limits[2] + (grid[1][1, 0] - grid[1][0, 0]) * y
    print(f"{title} spherical measured dx={x:.3f} [mm] dy={y:.3f} [mm]")
    print(f"{title} spherical nominal dx={spherical_substrate.surface.dx:.3f} [mm] dy={spherical_substrate.surface.dy:.3f} [mm]")
    ax[1].imshow(rebin(interferogram, (bnx, bny)), origin='lower', extent=limits)
    ax[1].set_title(f'{title} spherical substrate')
