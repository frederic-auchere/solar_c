
from astropy.io import fits
import matplotlib.pyplot as plt
from optical import rectangular_sw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np

path_nanomefos = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/'
files_nanomefos = '25264_Subsrat_SW_SN5 1_full_woTilt.datx'
files_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/report_full_substrate.fits'
dx_nanomefos = 60
dy_nanomefos=138

nanomefos_data = SagData(
    os.path.join(path_nanomefos, files_nanomefos),
    dx=dx_nanomefos, dy=dy_nanomefos, theta=0, binning=1, auto_crop=True
)
# plt.imshow(SagData(os.path.join(path_nanomefos, files_nanomefos)).sag, aspect="equal", origin="lower")
# plt.show()
measured_surface_nanomefos = surfaces.MeasuredSurface(nanomefos_data, alpha=0, beta=0, gamma=0)

measured_substrate_nanomefos = surfaces.Substrate(
    measured_surface_nanomefos,
    rectangular_sw_substrate.aperture,
    rectangular_sw_substrate.useful_area
)
measured_substrate_nanomefos.x_grid_step=0.1
measured_substrate_nanomefos.y_grid_step=0.1

map1 = np.asarray(measured_substrate_nanomefos.sag().data)*1e6
# plt.imshow(map1, origin='lower')
# plt.show()


with fits.open(files_zygo, mode="update") as hdul:
    zygo_data = hdul[0].data
    header = hdul[0].header
    header["WAVELNTH"] =  633     # Add or change a keyword
    header["GX"] = 0.0989
    header["GY"] = 0.0989

    hdul.flush()
## Plot
# plt.imshow(zygo_data, origin="lower", cmap="gray")
# plt.colorbar(label="Intensity")
# plt.title("FITS Image")
# plt.show()
# print(header)
#
dx_zygo = 200
dy_zygo=432
zygo_data = SagData(
    files_zygo,
    dx=dx_zygo, dy=dy_zygo, theta=0, binning=1, auto_crop=True
)

measured_surface_zygo = surfaces.MeasuredSurface(zygo_data, alpha=0, beta=0, gamma=0)

measured_substrate_zygo = surfaces.Substrate(
    measured_surface_zygo,
    rectangular_sw_substrate.aperture,
    rectangular_sw_substrate.useful_area,
)
measured_substrate_zygo.x_grid_step=0.1
measured_substrate_zygo.y_grid_step=0.1
map2 = np.asarray(measured_substrate_zygo.sag().data)
plt.imshow(map2, origin='lower')
plt.show()
#
fig, ax = plt.subplots(1, 3, figsize=(13, 5))


im1 = ax[0].imshow(map1, origin='lower',extent=measured_substrate_zygo.aperture.limits)
ax[0].set_title("Nanomefos")
fig.colorbar(im1, ax=ax[0], label="Intensity")


im2 = ax[1].imshow(map2, origin='lower',extent=measured_substrate_zygo.aperture.limits)
ax[1].set_title("Zygo")


fig.colorbar(im2, ax=ax[1], label="Intensity")

diff=map2-map1
im3 = ax[2].imshow(diff, origin='lower', vmin=-100, vmax=100,extent=measured_substrate_zygo.aperture.limits)
ax[2].set_title("Difference")
fig.colorbar(im3, ax=ax[2], label="Intensity")

plt.tight_layout()
plt.show()
