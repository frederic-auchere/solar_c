import numpy as np
import os
from astropy.io import fits
from optical import rectangular_sw_substrate, lw_substrate
from optics.zygo import SagData
from optical.zygo import EGAFit
from optics import surfaces
import matplotlib.pyplot as plt
from fitting import sfit
from optical.utils import write_zygo_dat



# matplotlib.rcParams['figure.autolayout'] = True

files_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_FM_form_casquette_Nina_full_sphere.fits'


# =============ZYGO DATA==============

# with fits.open(files_zygo, mode="update") as hdul:
#     zygo_data = hdul[0].data
#     header = hdul[0].header
#     header["WAVELNTH"] =  633
#     header["GX"] = 0.0989
#     header["GY"] = 0.0989
#
#     hdul.flush()
# ## Plot
# plt.imshow(zygo_data, origin="lower", cmap="gray")
# plt.colorbar(label="Intensity")
# plt.title("FITS Image")
# plt.show()
# print(header)

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

sz2 = measured_substrate_zygo.sag().shape
ny2, nx2 = sz2

xpix2=measured_surface_zygo.sag_data.gx
ypix2=measured_surface_zygo.sag_data.gy
print('aa')
write_zygo_dat('/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_FM_form_casquette_Nina_full_sphere5.dat',measured_substrate_zygo.sag().data,ypix2)