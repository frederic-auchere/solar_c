import numpy as np
import os
from optical import sw_substrate, lw_substrate
from optics.zygo import SagData
from optical.zygo import EGAFit
from optics import surfaces
import matplotlib.pyplot as plt
from fitting import sfit

path_nanomefos = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/LW_SN1/'
files_nanomefos = [
    'LW_SN1_2_ZU_wotiltnina3.datx',
    'LW_SN1_2_ZU_wotilt.datx',
    'LW-SN2_2_ZU_woTilt.datx',
    'LW-SN2_2_ZU_raw.datx']
files_roughness=['LW-SN1_Rugo_mesure_allFitted.datx','LW-SN1_Rugo_mesure.datx']
#mesurement of the dx and dy in the images
dx_nanomefos = [60,60]
dy_nanomefos = [61,61]



file_idx = 1
print(SagData(os.path.join(path_nanomefos, files_nanomefos[file_idx])).gx)

# plt.imshow(SagData(os.path.join(path_nanomefos, files_nanomefos[file_idx])).sag, aspect="equal", origin="lower")

#
nanomefos_data = SagData(
    os.path.join(path_nanomefos, files_nanomefos[file_idx]),
    dx=dx_nanomefos[file_idx], dy=dy_nanomefos[file_idx], theta=0, binning=1, auto_crop=True
)

gx = 0.00109
gy = gx
dx_roughness = 61.925 /2 / gx
dy_roughness = 86.188 / 2 / gy


roughness_data = SagData(
    os.path.join(path_nanomefos, files_roughness[file_idx]),
    dx=dx_roughness, dy=dy_roughness, theta=0, binning=1, auto_crop=True
)

    # dx=136.25, dy=93.5,
# print(SagData(os.path.join(path_nanomefos, files_roughness[file_idx])).gx)


print(roughness_data.sag.shape)
# dx and dy: sampling steps in x and y
# theta: rotation angle
# binning: pixel binning factor
# auto_crop: automatically crop the measurement area


measured_surface = surfaces.MeasuredSurface(nanomefos_data, alpha=0, beta=0, gamma=0)

measured_substrate = surfaces.Substrate(
    measured_surface,
    lw_substrate.aperture,
    lw_substrate.useful_area
)

# Get sag map as a 2D array
map1 = np.asarray(measured_substrate.sag().data)*1e6
# map1-=sfit(map1,2, missing=np.nan)[0]
sz = measured_substrate.sag().shape
ny1, nx1 = sz
map2 = np.asarray(roughness_data.sag.data)*1e6
map2 -=sfit(map2,4)[0]
sz2 = roughness_data.sag.shape
ny2, nx2 = sz2
#
# plt.imshow(map1, aspect="equal", origin="lower")
# plt.colorbar()
# plt.show()


# Physical size of the substrate in mm
xw, yw = 18.3, 36.9
xpix1 = xw / nx1   # mm/pixel
ypix1 = yw / ny1   # mm/pixel

# Physical size of the substrate in mm
xw2, yw2 = 0.702, 0.526
xpix2 = xw2 / nx2   # mm/pixel
ypix2 = yw2 / ny2   # mm/pixel

# plt.imshow(measured_substrate.sag(), aspect="auto", origin="lower")
# plt.colorbar()
# plt.show()
print("Shape:", sz)
print("Shape:", sz2)

win_x1 = np.hanning(nx1)
win_y1 = np.hanning(ny1)
window1 = np.outer(win_y1, win_x1)

dataimg1 = map1 - np.nanmean(map1)

win_x2 = np.hanning(nx2)
win_y2 = np.hanning(ny2)
window2 = np.outer(win_y2, win_x2)

dataimg2 = map2 - np.nanmean(map2)

# Replace NaN values with the mean

dataimg1 =  np.nan_to_num(dataimg1,nan=0.0)
dataimg2 = np.nan_to_num(dataimg2, nan=0.0)

# Variance
var1 = np.var(dataimg1, ddof=1)
if var1 <= 0:
    raise ValueError("Variance is zero or negative, PSD cannot be calculated")
var2 = np.var(dataimg2, ddof=1)
if var2 <= 0:
    raise ValueError("Variance is zero or negative, PSD cannot be calculated")

# Appliquer la fenêtre
dataimg1 *= window1
dataimg2 *= window2
# ---------------------
# CALCUL DU PSD
# ---------------------
# Compute the 2D power spectral density (PSD) and normalize by pixel size and variance
psd1 = (np.abs(np.fft.fft2(dataimg1))**2) * (xpix1 * 1e6 * ypix1 * 1e6) / (var1*nx1*ny1) #pas besoin de multiplier par N car different de IDL
psd2 = (np.abs(np.fft.fft2(dataimg2))**2) * (xpix2 * 1e6 * ypix2 * 1e6) / (var2*nx2*ny2)
# plt.imshow(np.log10(psd1), aspect="auto")
# plt.colorbar()
# plt.show()
# plt.imshow(np.log10(psd2), aspect="auto")s
# plt.colorbar()
# plt.show()
psd1 = psd1[:ny1 // 2, :nx1 // 2] # Keep only the first quadrant (positive frequencies)
psd2 = psd2[:ny2 // 2, :nx2 // 2]


# nu_x = np.fft.fftfreq(nx1, d=xpix1* 1e6)#[:nx1//2]  # nm^-1
nu_x=(np.linspace(0, nx1-1, nx1)/(nx1*xpix1*1e6))[:nx1//2]
nu_y=(np.linspace(0, ny1-1, ny1)/(ny1*ypix1*1e6))[:ny1//2]  # nm^-1
print(nu_x.shape, nu_y.shape)
nu_x2 = np.fft.fftfreq(nx2, d=xpix2* 1e6)[:nx2//2]  # nm^-1
nu_y2 = np.fft.fftfreq(ny2, d=ypix2* 1e6)[:ny2//2]  # nm^-1
# print(nu_x2.shape, nu_y2.shape)
# Extract 1D cuts along x and y
psd_y = psd1[:, 0]
psd_x = psd1[0, :]
print(psd_y.shape,psd_x.shape)

psd_y2 = psd2[:, 0]
psd_x2 = psd2[0, :]


log_x = np.log10(nu_x[1:])
log_psd_x = np.log10(psd_x[1:])
A_x = np.vstack([log_x, np.ones(len(log_x))]).T
m_x, c_x = np.linalg.lstsq(A_x, log_psd_x, rcond=None)[0]
print(f"Pente PSD X = {m_x:.2f}")

psd_fit_x = 10 ** (m_x * np.log10(nu_x[1:]) + c_x)

log_x2 = np.log10(nu_x2[1:])
log_psd_x2 = np.log10(psd_x2[1:])
A_x2 = np.vstack([log_x2, np.ones(len(log_x2))]).T
m_x2, c_x2 = np.linalg.lstsq(A_x2, log_psd_x2, rcond=None)[0]
print(f"Pente PSD X rugo = {m_x2:.2f}")

psd_fit_x2 = 10 ** (m_x2 * np.log10(nu_x2[1:]) + c_x2)

log_y = np.log10(nu_y[1:])
log_psd_y = np.log10(psd_y[1:])
A_y = np.vstack([log_y, np.ones(len(log_y))]).T
m_y, c_y = np.linalg.lstsq(A_y, log_psd_y, rcond=None)[0]
print(f"Pente PSD Y = {m_y:.2f}")

psd_fit_y = 10 ** (m_y * np.log10(nu_y[1:]) + c_y)

log_y2 = np.log10(nu_y2[1:])
log_psd_y2 = np.log10(psd_y2[1:])
A_y2 = np.vstack([log_y2, np.ones(len(log_y2))]).T
m_y2, c_y2 = np.linalg.lstsq(A_y2, log_psd_y2, rcond=None)[0]
print(f"Pente PSD Y = {m_y2:.2f}")

psd_fit_y2 = 10 ** (m_y2 * np.log10(nu_y2[1:]) + c_y2)

# nu measured in nm^-1
nu_nm = nu_x[1:]  # your x-axis in nm^-1

# convert to mm^-1 for spec formula
nu_mm = nu_nm * 1e6  # nm^-1 → mm^-1

# compute spec PSD
psd_spec = 2e10 * nu_mm**(-3)  # PSD in nm^4

nu_all = np.concatenate([nu_x[1:], nu_x2[1:]])
psd_all = np.concatenate([psd_x[1:], psd_x2[1:]])

log_nu_all = np.log10(nu_all)
log_psd_all = np.log10(psd_all)

A_all = np.vstack([log_nu_all, np.ones(len(log_nu_all))]).T
m_all, c_all = np.linalg.lstsq(A_all, log_psd_all, rcond=None)[0]
print(f"Pente globale PSD (X1+X2) = {m_all:.2f}")

psd_fit_x1x2 = 10 ** (m_all * log_nu_all + c_all)



# ---------------------
# PLOTS
# ---------------------
plt.figure(figsize=(7,5))
plt.loglog(nu_x[1:], psd_x[1:], label="Cut X")
plt.loglog(nu_x2[1:], psd_x2[1:], label="Cut X rugo")
plt.loglog(nu_x[1:], psd_fit_x, '--', label=f"Fit X (slope={m_x:.2f})")
plt.loglog(nu_x2[1:], psd_fit_x2, '--', label=f"Fit X rugo (slope={m_x2:.2f})")
plt.loglog(nu_y[1:], psd_y[1:], label="Cut Y")
plt.loglog(nu_y2[1:], psd_y2[1:], label="Cut Y rugo")
plt.loglog(nu_y[1:], psd_fit_y, '--', label=f"Fit Y (slope={m_y:.2f})")
plt.loglog(nu_y2[1:], psd_fit_y2, '--', label=f"Fit Y rugo (slope={m_y2:.2f})")
plt.loglog(nu_nm, psd_spec, '--', label="Spec 2e10 ν⁻³ (ν in mm⁻¹)")
order = np.argsort(nu_all)
plt.loglog(nu_all[order], psd_fit_x1x2[order], '--', label=f"Fit global X (slope={m_all:.2f})")

plt.xlabel(r" $\nu$ (nm$^{-1}$)")
plt.ylabel(r"PSD [nm$^4$]")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.title("Anisotropic PSD : cuts X et Y")
plt.show()

