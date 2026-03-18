
from astropy.io import fits
import matplotlib.pyplot as plt
from optical import rectangular_sw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np

from rectify.rectify import PolarTransform, Rectifier

path_nanomefos = r'Y:/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/'
files_nanomefos = '25264_Subsrat_SW_SN5 1_innercirclecropped_woTilt.datx'
files_zygo = r'Y:/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/report_polished_area3.fits'
dx_nanomefos = 61
dy_nanomefos = 139

nanomefos_data = SagData(
    os.path.join(path_nanomefos, files_nanomefos),
    dx=dx_nanomefos, dy=dy_nanomefos, theta=0, binning=1, auto_crop=True
)

measured_surface_nanomefos = surfaces.MeasuredSurface(nanomefos_data, alpha=0, beta=0, gamma=0)

measured_substrate_nanomefos = surfaces.Substrate(
    measured_surface_nanomefos,
    rectangular_sw_substrate.aperture,
    rectangular_sw_substrate.useful_area,
    x_grid_step=0.1,
    y_grid_step=0.1,
)

map1 = measured_substrate_nanomefos.sag() * 1e6

zygo_data = SagData(files_zygo)
print(zygo_data.theta)
measured_surface_zygo = surfaces.MeasuredSurface(zygo_data, alpha=0, beta=0, gamma=0)

measured_substrate_zygo = surfaces.Substrate(
    measured_surface_zygo,
    rectangular_sw_substrate.aperture,
    rectangular_sw_substrate.useful_area,
    x_grid_step=0.1,
    y_grid_step=0.1
)
map2 = measured_substrate_zygo.sag()

fig, ax = plt.subplots(1, 3, figsize=(13, 5), tight_layout=True)

im1 = ax[0].imshow(map1, origin='lower',extent=measured_substrate_zygo.aperture.limits, vmin=-30, vmax=30)
ax[0].set_title("Nanomefos")
fig.colorbar(im1, ax=ax[0], label="sag [nm]")

im2 = ax[1].imshow(map2, origin='lower',extent=measured_substrate_zygo.aperture.limits, vmin=-30, vmax=30)
ax[1].set_title("Zygo")
fig.colorbar(im2, ax=ax[1], label="sag [nm]")

diff=map2-map1
im3 = ax[2].imshow(diff, origin='lower', vmin=-30, vmax=30,extent=measured_substrate_zygo.aperture.limits)
ax[2].set_title("Difference")
fig.colorbar(im3, ax=ax[2], label="sag [nm]")

plt.show()

fig, ax = plt.subplots(2, 1)
polar = PolarTransform(diff.shape[1] / 2, diff.shape[0] / 2)
polarizer = Rectifier(polar)
polar = polarizer(diff, (220, 360), (0, 360), (0, 220))
ax[0].imshow(polar, origin='lower', vmin=-100, vmax=100)
ax[0].axis('auto')
ax[1].plot(polar[100, :])