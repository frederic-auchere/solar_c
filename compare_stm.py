from astropy.io import fits
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from optical import sw_substrate, lw_substrate
from mpl_toolkits.axes_grid1 import make_axes_locatable

base_path = r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements"

sw1_after = r"\SW1_STM_BONDED\Zygo\20250625\20250625_sw1_stm_bonded_report.fits"
sw1_before = r"\SW1_STM\Zygo\20250619\20250619_sw1_stm_report.fits"

lw1_after = r"\LW1_STM_BONDED\Zygo\20250625\20250625_lw1_stm_bonded_report.fits"
lw1_before = r"\LW1_STM\Zygo\20250619\20250619_lw1_stm_report.fits"

for substrate, (file_after, file_before) in zip([sw_substrate, lw_substrate], [[sw1_after, sw1_before], [lw1_after, lw1_before]]):

    after = fits.getdata(base_path + file_after)
    before = fits.getdata(base_path + file_before)

    x = np.linspace(0, after.shape[1]-1, before.shape[1])
    y = np.linspace(0, after.shape[0]-1, before.shape[0])
    x, y = np.meshgrid(x, y)

    coords = np.stack((y, x), axis=0)
    after = map_coordinates(after,  # input array
                            coords,  # array of coordinates
                            order=2,  # spline order
                            mode='constant',  # fills in with constant value
                            cval=np.nan,  # constant value
                            prefilter=False)

    fig, axes = plt.subplots(1, 3, tight_layout=True)
    residuals = before - after

    for data, ax in zip([before, after, residuals], axes):

        minimum, maximum = np.nanmin(data), np.nanmax(data)
        pv = maximum - minimum
        rms = np.nanstd(data)
        vmax = max(abs(minimum), abs(maximum))

        im = ax.imshow(data,
                       origin='lower', extent=substrate.aperture.limits, vmin=-vmax, vmax=vmax)
        ax.set_xlim(substrate.limits[0], substrate.limits[1])
        ax.set_ylim(substrate.limits[2], substrate.limits[3])
        ax.set_xlabel('[mm]')
        ax.set_ylabel('[mm]')
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.05)
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_xlabel(f"[nm] | {rms:.2f} RMS | {pv:.1f} PV")

        fig.savefig(substrate.name + '_bonding.png', dpi=300)