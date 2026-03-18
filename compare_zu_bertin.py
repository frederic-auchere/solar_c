from astropy.io import fits
import matplotlib.pyplot as plt
from optical import sw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
from rectify.rectify import PolarTransform, Rectifier

path_nanomefos = r'Y:/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/'
files_nanomefos = '25264_Subsrat_SW_SN5 2_ZU_woTilt.datx'
files_zygo = r'Y:/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/report.fits'
dx_nanomefos = 0
dy_nanomefos = 189

nanomefos_data = SagData(
    os.path.join(path_nanomefos, files_nanomefos),
    dx=dx_nanomefos, dy=dy_nanomefos, theta=0, binning=1, auto_crop=True
)

measured_surface_nanomefos = surfaces.MeasuredSurface(nanomefos_data, alpha=0, beta=0, gamma=0)

measured_substrate_nanomefos = surfaces.Substrate(
    measured_surface_nanomefos,
    sw_substrate.aperture,
    sw_substrate.useful_area,
    x_grid_step = 0.1,
    y_grid_step = 0.1,
)
nanomefos_sag = measured_substrate_nanomefos.sag() * 1e6

with fits.open(files_zygo, mode="update") as hdul:
    zygo_data = hdul[0].data
    header = hdul[0].header
    header["WAVELNTH"] =  633     # Add or change a keyword
    header["GX"] = 0.0989
    header["GY"] = 0.0989
    header["DX"] = 15
    header["DY"] = 251
    header["THETA"] = 0
    hdul.flush()

print(header)

# dx_zygo = 15
# dy_zygo= 251
zygo_data = SagData(files_zygo, binning=1)

measured_surface_zygo = surfaces.MeasuredSurface(zygo_data, alpha=0, beta=0, gamma=0)

measured_substrate_zygo = surfaces.Substrate(
    measured_surface_zygo,
    sw_substrate.aperture,
    sw_substrate.useful_area,
    x_grid_step = 0.1,
    y_grid_step = 0.1,
)

zygo_sag = measured_substrate_zygo.sag()

fig, ax = plt.subplots(1, 3, figsize=(13, 5))

im1 = ax[0].imshow(nanomefos_sag, origin='lower', extent=measured_substrate_nanomefos.aperture.limits)
ax[0].set_title("Nanomefos")
fig.colorbar(im1, ax=ax[0], label="Sag [nm]")

im2 = ax[1].imshow(zygo_sag, origin='lower', extent=measured_substrate_zygo.aperture.limits)
ax[1].set_title("Zygo")

fig.colorbar(im2, ax=ax[1], label="Sag [nm]")

diff = zygo_sag - nanomefos_sag
im3 = ax[2].imshow(diff, origin='lower', extent=measured_substrate_nanomefos.aperture.limits)
ax[2].set_title("Difference")
fig.colorbar(im3, ax=ax[2], label="Sag [nm]")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1)
polar = PolarTransform(diff.shape[1] / 2, diff.shape[0] / 2)
polarizer = Rectifier(polar)
polar = polarizer(diff, (100, 360), (0, 360), (0, 100))
ax[0].imshow(polar, origin='lower', vmin=-100, vmax=100)
ax[0].axis('auto')
ax[1].plot(polar[30, :])

