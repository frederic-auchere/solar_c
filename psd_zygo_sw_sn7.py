import numpy as np
import os
from astropy.io import fits
from optical import sw_substrate,rectangular_sw_substrate, lw_substrate
from optics.zygo import SagData
from optical.zygo import EGAFit
from optics import surfaces
import matplotlib.pyplot as plt
from fitting import sfit



# matplotlib.rcParams['figure.autolayout'] = True

files_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN7/Zygo/Form/07012026/substrate_FM_form_SW_SN7_Nina.fits'
files_roughness='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN7/25-11-07_SW-SN7_rugo.datx'
file_one_measurement='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/STM/LW_RECTANGLE/Zygo/Form/20260427/substrates_template_FM_form_casquette_rectangular.fits'
# =============ZYGO DATA==============

# with fits.open(files_zygo, mode="update") as hdul:
#     zygo_data = hdul[0].data
#     header = hdul[0].header
#     print(header)
#     header["WAVELNTH"] =  633
#     header["GX"] = 0.0989
#     header["GY"] = 0.0989
#
#     hdul.flush()
# ## Plot



zygo_data = SagData(
    files_zygo,theta=0, binning=1, auto_crop=True)
#     dx=dx_zygo, dy=dy_zygo, theta=0, binning=1, auto_crop=True
# )
# plt.imshow(zygo_data, origin="lower", cmap="gray")
# plt.colorbar(label="Intensity")
# plt.title("FITS Image")
# plt.show()

measured_surface_zygo = surfaces.MeasuredSurface(zygo_data, alpha=0, beta=0, gamma=0)

measured_substrate_zygo = surfaces.Substrate(
    measured_surface_zygo,
    sw_substrate.aperture,
    sw_substrate.useful_area,
)


measured_substrate_zygo.x_grid_step=0.1
measured_substrate_zygo.y_grid_step=0.1



map2 = np.asarray(measured_substrate_zygo.sag().data)
# plt.imshow(map2, aspect="equal", origin="lower")
# plt.colorbar()
# plt.show()
sz2 = measured_substrate_zygo.sag().shape
ny2, nx2 = sz2

xpix2=measured_surface_zygo.sag_data.gx
ypix2=measured_surface_zygo.sag_data.gy


win_x2 = np.hanning(nx2)
win_y2 = np.hanning(ny2)
window2 = np.outer(win_y2, win_x2)
mask = np.isnan(map2)
median_val = np.nanmedian(map2)
map2[mask] = median_val
dataimg2 = map2 - np.nanmean(map2)

# =============ROUGHNESS DATA==============

gx = 0.00109
gy = gx
dx_roughness = 61.925 /2 / gx
dy_roughness = 86.188 / 2 / gy


roughness_data = SagData(files_roughness,dx=dx_roughness, dy=dy_roughness, theta=0, binning=1, auto_crop=True)

map1 = np.asarray(roughness_data.sag.data)*1e6 #convert to mm to nm
# map1 -=sfit(map1,degree=2)[0]
sz1 = roughness_data.sag.shape
ny1, nx1 = sz1
# plt.imshow(map1, aspect="equal", origin="lower")
# plt.colorbar()

xw1, yw1 = 0.702, 0.526 #in mm
xpix1 = xw1 / nx1   # mm/pixel
ypix1 = yw1 / ny1   # mm/pixel


win_x1 = np.hanning(nx1)
win_y1 = np.hanning(ny1)
window1 = np.outer(win_y1, win_x1)

dataimg1 = map1 - np.nanmean(map1)

zygo_data_one_measurement = SagData(
    file_one_measurement,theta=0, binning=1, auto_crop=True)
#     dx=dx_zygo, dy=dy_zygo, theta=0, binning=1, auto_crop=True
# )
#
measured_surface_zygo_one_measurement = surfaces.MeasuredSurface(zygo_data_one_measurement , alpha=0, beta=0, gamma=0)


measured_substrate_zygo_one_measurement = surfaces.Substrate(
    measured_surface_zygo_one_measurement,
    sw_substrate.aperture,
    sw_substrate.useful_area,
)


measured_substrate_zygo_one_measurement.x_grid_step=0.1
measured_substrate_zygo_one_measurement.y_grid_step=0.1

map3 = np.asarray(measured_substrate_zygo_one_measurement.sag().data)
# plt.imshow(map3, origin="lower")
# plt.colorbar()
# plt.show()
sz3 = measured_substrate_zygo_one_measurement.sag().shape
ny3, nx3 = sz3

xpix3=measured_surface_zygo_one_measurement.sag_data.gx
ypix3=measured_surface_zygo_one_measurement.sag_data.gy
#
#
win_x3 = np.hanning(nx3)
win_y3 = np.hanning(ny3)
window3 = np.outer(win_y3, win_x3)
mask = np.isnan(map3)
median_val = np.nanmedian(map3)
map3[mask] = median_val
dataimg3 = map3 - np.nanmean(map3)

# Variance
var1 = np.var(dataimg1, ddof=1)
var2 = np.var(dataimg2, ddof=1)
var3 = np.var(dataimg3, ddof=1)
# Appliquer la fenêtre
dataimg1 *= window1
dataimg2 *= window2
dataimg3 *= window3



# ---------------------
# CALCUL DU PSD
# ---------------------
# Compute the 2D power spectral density (PSD) and normalize by pixel size and variance
 #pas besoin de multiplier par N car different de IDL
psd1 = (np.abs(np.fft.fft2(dataimg1))**2) * (xpix1 * 1e6 * ypix1 * 1e6)/ (var1*nx1*ny1 )
psd2 = (np.abs(np.fft.fft2(dataimg2))**2) * (xpix2 * 1e6 * ypix2 * 1e6) / (var2*nx2*ny2)
psd3 = (np.abs(np.fft.fft2(dataimg3))**2) * (xpix3 * 1e6 * ypix3 * 1e6) / (var3*nx3*ny3)

# psd1 = (np.abs(np.fft.fft2(dataimg1))**2) * (xpix1 * 1e6 * ypix1 * 1e6)/ (nx1*ny1 )
# psd2 = (np.abs(np.fft.fft2(dataimg2))**2) * (xpix2 * 1e6 * ypix2 * 1e6) / (nx2*ny2)
# psd3 = (np.abs(np.fft.fft2(dataimg3))**2) * (xpix3 * 1e6 * ypix3 * 1e6) / (nx3*ny3)
psd1 = psd1[:ny1 // 2, :nx1 // 2]
psd2 = psd2[:ny2 // 2, :nx2 // 2]
psd3 = psd3[:ny3 // 2, :nx3 // 2]
nu_x1 = np.fft.fftfreq(nx1, d=xpix1*1e6)[:nx1//2]  # nm^-1
nu_y1 = np.fft.fftfreq(ny1, d=ypix1*1e6)[:ny1//2]  # nm^-1

nu_x2 = np.fft.fftfreq(nx2, d=xpix2*1e6)[:nx2//2]  # nm^-1
nu_y2 = np.fft.fftfreq(ny2, d=ypix2*1e6)[:ny2//2]  # nm^-1

nu_x3 = np.fft.fftfreq(nx3, d=xpix3*1e6)[:nx3//2]  # nm^-1
nu_y3 = np.fft.fftfreq(ny3, d=ypix3*1e6)[:ny3//2]  # nm^-1

# Extract 1D cuts along x and y
psd_y = psd1[:, 0]
psd_x = psd1[0, :]

psd_y2 = psd2[:, 0]
psd_x2 = psd2[0, :]


psd_y3 = psd3[:, 0]
psd_x3 = psd3[0, :]
# ===================FIT========================================
log_x = np.log10(nu_x1[1:])
log_psd_x = np.log10(psd_x[1:])
A_x = np.vstack([log_x, np.ones(len(log_x))]).T
m_x, c_x = np.linalg.lstsq(A_x, log_psd_x, rcond=None)[0]
# print(f"Pente PSD X = {m_x:.2f}")

psd_fit_x = 10 ** (m_x * np.log10(nu_x1[1:]) + c_x)

log_x2 = np.log10(nu_x2[1:])
log_psd_x2 = np.log10(psd_x2[1:])
A_x2 = np.vstack([log_x2, np.ones(len(log_x2))]).T
m_x2, c_x2 = np.linalg.lstsq(A_x2, log_psd_x2, rcond=None)[0]
# print(f"Pente PSD X rugo = {m_x2:.2f}")

psd_fit_x2 = 10 ** (m_x2 * np.log10(nu_x2[1:]) + c_x2)

log_y = np.log10(nu_y1[1:])
log_psd_y = np.log10(psd_y[1:])
A_y = np.vstack([log_y, np.ones(len(log_y))]).T
m_y, c_y = np.linalg.lstsq(A_y, log_psd_y, rcond=None)[0]
# print(f"Pente PSD Y = {m_y:.2f}")

psd_fit_y = 10 ** (m_y * np.log10(nu_y1[1:]) + c_y)

log_y2 = np.log10(nu_y2[1:])
log_psd_y2 = np.log10(psd_y2[1:])
A_y2 = np.vstack([log_y2, np.ones(len(log_y2))]).T
m_y2, c_y2 = np.linalg.lstsq(A_y2, log_psd_y2, rcond=None)[0]
# print(f"Pente PSD Y = {m_y2:.2f}")

psd_fit_y2 = 10 ** (m_y2 * np.log10(nu_y2[1:]) + c_y2)

# nu measured in nm^-1
nu_nm = nu_x2[1:] # x-axis in nm^-1
nu_nm=np.append(nu_nm,np.linspace(1.78e-6,1e-3))

# convert to mm^-1 for spec formula
nu_mm = nu_nm * 1e6  # nm^-1 → mm^-1

# compute spec PSD
psd_spec = 2e10 * nu_mm**(-3)  # PSD in nm^4

# nu_all = nu_x2
# psd_all =  psd_x2[1:]
#
# log_nu_all = np.log10(nu_all)
# log_psd_all = np.log10(psd_all)

# A_all = np.vstack([log_nu_all, np.ones(len(log_nu_all))]).T
# m_all, c_all = np.linalg.lstsq(A_all, log_psd_all, rcond=None)[0]
# print(f"Pente globale PSD (X1+X2) = {m_all:.2f}")
#
# psd_fit_x1x2 = 10 ** (m_all * log_nu_all + c_all)



# ---------------------
# PLOTS
# ---------------------
plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 14})
plt.loglog(nu_x2[1:], psd_x2[1:], label="Cut X zygo")
plt.loglog(nu_x1[1:], psd_x[1:], label="Cut X rugo")
# plt.loglog(nu_x3[1:], psd_x3[1:], label="Cut X one")
# plt.loglog(nu_x2[1:], psd_fit_x2, '--', label=f"Fit X zygo (slope={m_x2:.2f})")
# plt.loglog(nu_x1[1:], psd_fit_x, '--', label=f"Fit X rugo (slope={m_x:.2f})")
plt.loglog(nu_y2[1:], psd_y2[1:], label="Cut Y zygo")
# plt.loglog(nu_y3[1:], psd_y3[1:], label="Cut Y one")
plt.loglog(nu_y1[1:], psd_y[1:], label="Cut Y rugo")
# plt.loglog(nu_y2[1:], psd_fit_y2, '--', label=f"Fit Y zygo (slope={m_y2:.2f})")
# plt.loglog(nu_y1[1:], psd_fit_y, '--', label=f"Fit Y rugo (slope={m_y:.2f})")
plt.loglog(nu_nm, psd_spec, '--', label="Spec 2e10 ν⁻³ (ν in mm⁻¹)",color='k')


plt.xlabel(r" $\nu$ (nm$^{-1}$)")
plt.ylabel(r"PSD [nm$^4$]")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.title(" Anisotropic PSD : cuts X et Y")
plt.suptitle('SW SN7')
plt.show()

