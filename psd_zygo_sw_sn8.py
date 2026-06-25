import numpy as np
import os
from astropy.io import fits
from optical import sw_substrate,rectangular_sw_substrate, lw_substrate
from optics.zygo import SagData
from optical.zygo import EGAFit
from optics import surfaces

import matplotlib
matplotlib.use('TkAgg')  # ou 'TkAgg'
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from fitting import sfit

from scipy.stats import binned_statistic

# matplotlib.rcParams['figure.autolayout'] = True

files_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN8/Zygo/Form/19012026/report_binning_1.fits'
files_zygo_zoom='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN8/Zygo/Form/30032026/substrates_template_FM_form_casquette.fits'
files_roughness='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/25-11-13_SW-SN8_rugo.datx'
# file_one_measurement='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/report_test_1_mesure.fits'
# =============ZYGO DATA==============





zygo_data = SagData(
    files_zygo,theta=0, binning=1, auto_crop=True)

measured_surface_zygo = surfaces.MeasuredSurface(zygo_data, alpha=0, beta=0, gamma=0)

measured_substrate_zygo = surfaces.Substrate(
    measured_surface_zygo,
    sw_substrate.aperture,
    sw_substrate.useful_area,
)


# measured_substrate_zygo.x_grid_step=0.1
# measured_substrate_zygo.y_grid_step=0.1



map2 = np.asarray(measured_substrate_zygo.sag().data)
# plt.imshow(map2, aspect="equal", origin="lower")
# plt.colorbar()
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



zygo_data_zoom = SagData(
    files_zygo_zoom,theta=0, binning=1, auto_crop=True)

measured_surface_zygo_zoom = surfaces.MeasuredSurface(zygo_data_zoom, alpha=0, beta=0, gamma=0)

measured_substrate_zygo_zoom = surfaces.Substrate(
    measured_surface_zygo_zoom,
    sw_substrate.aperture,
    sw_substrate.useful_area,
)


# measured_substrate_zygo.x_grid_step=0.1
# measured_substrate_zygo.y_grid_step=0.1



map3 = np.asarray(measured_substrate_zygo_zoom.sag().data)
# plt.imshow(map3, aspect="equal", origin="lower")
# plt.colorbar()
sz3 = measured_substrate_zygo_zoom.sag().shape
ny3, nx3 = sz3

xpix3=measured_surface_zygo_zoom.sag_data.gx
ypix3=measured_surface_zygo_zoom.sag_data.gy


win_x3 = np.hanning(nx3)
win_y3 = np.hanning(ny3)
window3 = np.outer(win_y3, win_x3)
mask = np.isnan(map3)
median_val = np.nanmedian(map3)
map3[mask] = median_val
dataimg3 = map3 - np.nanmean(map3)

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


# var3 = np.var(dataimg3, ddof=1)
# Appliquer la fenêtre
dataimg1 *= window1
dataimg2 *= window2
dataimg3 *= window3

# Variance
var1 = np.var(dataimg1, ddof=1)
var2 = np.var(dataimg2, ddof=1)
var3 = np.var(dataimg3, ddof=1)

# ---------------------
# CALCUL DU PSD
# ---------------------
# Compute the 2D power spectral density (PSD) and normalize by pixel size and variance
 #pas besoin de multiplier par N car different de IDL
 #sans variance
# psd1 = (np.abs(np.fft.fft2(dataimg1))**2) * (xpix1 * 1e6 * ypix1 * 1e6)/(nx1*ny1 )
# psd2 = (np.abs(np.fft.fft2(dataimg2))**2) * (xpix2 * 1e6 * ypix2 * 1e6) /(nx2*ny2)
# psd3 = (np.abs(np.fft.fft2(dataimg3))**2) * (xpix3 * 1e6 * ypix3 * 1e6) /(nx3*ny3)

#avec variance
psd1 = (np.abs(np.fft.fft2(dataimg1))**2) * (xpix1 * 1e6 * ypix1 * 1e6)/(var1*nx1*ny1 )
psd2 = (np.abs(np.fft.fft2(dataimg2))**2) * (xpix2 * 1e6 * ypix2 * 1e6) / (var2*nx2*ny2)
psd3 = (np.abs(np.fft.fft2(dataimg3))**2) * (xpix3 * 1e6 * ypix3 * 1e6) /(var3*nx3*ny3)
psd1 = psd1[:ny1 // 2, :nx1 // 2]
psd2 = psd2[:ny2 // 2, :nx2 // 2]
psd3 = psd3[:ny3 // 2, :nx3 // 2]

# plt.imshow(np.log10(psd3), origin='lower')


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

# print(f"Zygo full  : nx={nx2}, xpix={xpix2*1e6:.4f} nm/pix, champ={nx2*xpix2*1e6:.1f} nm")
# print(f"Zygo zoom  : nx={nx3}, xpix={xpix3*1e6:.4f} nm/pix, champ={nx3*xpix3*1e6:.1f} nm")
# print(f"ν_min zygo full : {nu_x2[1]:.3e} nm⁻¹")
# print(f"ν_min zygo zoom : {nu_x3[1]:.3e} nm⁻¹")
# print(f"ν_max zygo full : {nu_x2[-1]:.3e} nm⁻¹")
# print(f"ν_max zygo zoom : {nu_x3[-1]:.3e} nm⁻¹")
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
fig, axes = plt.subplots(1, 1, figsize=(7, 5))
plt.rcParams.update({'font.size': 14})

# ============================================================
# PANEL 1 : Toutes les courbes
# ============================================================
ax = axes
ax.loglog(nu_x2[1:], psd_x2[1:], label="Cut X zygo")
ax.loglog(nu_x1[1:], psd_x[1:],  label="Cut X rugo")
# ax.loglog(nu_x3[1:], psd_x3[1:], label="Cut X zoom")
ax.loglog(nu_y2[1:], psd_y2[1:], label="Cut Y zygo", linestyle='--')
ax.loglog(nu_y1[1:], psd_y[1:],  label="Cut Y rugo", linestyle='--')
# ax.loglog(nu_y3[1:], psd_y3[1:], label="Cut Y zoom", linestyle='--')
ax.loglog(nu_nm, psd_spec, 'k--', label="Spec 2e10 ν⁻³")
ax.set_xlabel(r"$\nu$ (nm$^{-1}$)")
ax.set_ylabel(r"PSD [nm$^4$]")
ax.grid(True, which="both", ls="--")
ax.legend(fontsize=10)
# ax.set_title("X et Y")
#
# ============================================================
# # PANEL 2 : Coupes X seulement
# # ============================================================
# ax = axes[1]
# ax.loglog(nu_x2[1:], psd_x2[1:], label="Cut X zygo")
# ax.loglog(nu_x1[1:], psd_x[1:],  label="Cut X rugo")
# ax.loglog(nu_x3[1:], psd_x3[1:], label="Cut X zoom")
# ax.loglog(nu_nm, psd_spec, 'k--', label="Spec 2e10 ν⁻³")
# ax.set_xlabel(r"$\nu$ (nm$^{-1}$)")
# ax.set_ylabel(r"PSD [nm$^4$]")
# ax.grid(True, which="both", ls="--")
# ax.legend(fontsize=8)
# ax.set_title("Cut X")
# #
# # ============================================================
# # PANEL 3 : Coupes Y seulement
# # ============================================================
# ax = axes[2]
# ax.loglog(nu_y2[1:], psd_y2[1:], label="Cut Y zygo")
# ax.loglog(nu_y1[1:], psd_y[1:],  label="Cut Y rugo")
# ax.loglog(nu_y3[1:], psd_y3[1:], label="Cut Y zoom")
# ax.loglog(nu_nm, psd_spec, 'k--', label="Spec 2e10 ν⁻³")
# ax.set_xlabel(r"$\nu$ (nm$^{-1}$)")
# ax.set_ylabel(r"PSD [nm$^4$]")
# ax.grid(True, which="both", ls="--")
# ax.legend(fontsize=8)
# ax.set_title("Cut Y")

# plt.suptitle("SW SN8 - Anisotropic PSD", fontsize=14)
plt.tight_layout()
plt.show()
#
# # vmin = min(np.nanmin(map2), np.nanmin(map3))
# # vmax = max(np.nanmax(map2), np.nanmax(map3))
# #
# # fig, axes = plt.subplots(1, 2, figsize=(12, 7))
# #
# # im2 = axes[0].imshow(map2, aspect="equal", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
# # axes[0].set_title("Map 2 - Full")
# # plt.colorbar(im2, ax=axes[0], label="nm")
# #
# # im3 = axes[1].imshow(map3, aspect="equal", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
# # axes[1].set_title("Map 3 - Zoom")
# # plt.colorbar(im3, ax=axes[1], label="nm")
# #
# # plt.suptitle("SW SN8 - Comparaison Full vs Zoom")
# # plt.tight_layout()
# # plt.show()
#
#
#
#
#
#
#
# #
# #
# nu_x1_mm = nu_x1 * 1e6
# nu_y1_mm = nu_y1 * 1e6
# nu_x2_mm = nu_x2 * 1e6
# nu_y2_mm = nu_y2 * 1e6
# nu_x3_mm = nu_x3 * 1e6
# nu_y3_mm = nu_y3 * 1e6
# #
# #
# #
# fig, axes = plt.subplots(2, 3, figsize=(18, 11))
#
# nu_spec_mm = np.logspace(np.log10(min(nu_x2_mm[1], nu_x1_mm[1])),
#                           np.log10(nu_x1_mm[-1]), 500)
# psd_spec_mm = 2e10 * nu_spec_mm**(-3)
#
# # ============================================================
# # PLOT 1 : PSD 1D rugo vs spec avec zones annotées
# # ============================================================
# ax = axes[0, 0]
# ax.loglog(nu_x1_mm[1:], psd_x[1:], label="Cut X rugo")
# ax.loglog(nu_y1_mm[1:], psd_y[1:], label="Cut Y rugo", alpha=0.7)
# ax.loglog(nu_spec_mm, psd_spec_mm, '--', color='k', label="Spec 2e10 ν⁻³")
# ax.axvspan(1.4,   5,   alpha=0.1, color='red',    label="Forme résiduelle")
# ax.axvspan(5,   100,   alpha=0.1, color='green',  label="Gamme fiable")
# ax.axvspan(100, 500,   alpha=0.1, color='orange', label="Plancher bruit")
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("PSD [nm⁴]")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD rugo 1D vs spec")
#
# # ============================================================
# # PLOT 2 : Ratio 1D rugo/spec avec zones
# # ============================================================
# ax = axes[0, 1]
# spec_x = 2e10 * nu_x1_mm[1:]**(-3)
# spec_y = 2e10 * nu_y1_mm[1:]**(-3)
# ax.semilogx(nu_x1_mm[1:], np.log10(psd_x[1:] / spec_x), label="Cut X rugo")
# ax.semilogx(nu_y1_mm[1:], np.log10(psd_y[1:] / spec_y), label="Cut Y rugo", alpha=0.7)
# ax.axhline(0, color='k', linewidth=1.5, linestyle='--', label="Limite spec")
# ax.axvspan(1.4,   5,   alpha=0.1, color='red')
# ax.axvspan(5,   100,   alpha=0.1, color='green')
# ax.axvspan(100, 500,   alpha=0.1, color='orange')
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("log10(PSD/spec)")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("Ratio rugo/spec 1D")
# ax.set_ylim(-3, 3)
# ax.set_xlim(nu_x1_mm[1], 500)
#
# # ============================================================
# # PLOT 3 : Carte 2D ratio rugo avec masque Nyquist
# # ============================================================
# ax = axes[0, 2]
# Nu_x_r, Nu_y_r = np.meshgrid(nu_x1_mm, nu_y1_mm)
# Nu_r_rugo = np.sqrt(Nu_x_r**2 + Nu_y_r**2)
# Nu_r_rugo[Nu_r_rugo == 0] = np.nan
# ratio_rugo_2d = psd1[:ny1//2, :nx1//2] / (2e10 * Nu_r_rugo**(-3))
# ratio_rugo_masked = np.where((Nu_r_rugo >= 1.4) & (Nu_r_rugo <= 200), ratio_rugo_2d, np.nan)
# im = ax.pcolormesh(Nu_x_r, Nu_y_r, np.log10(ratio_rugo_masked), cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# ax.set_xlabel("νx (mm⁻¹)")
# ax.set_ylabel("νy (mm⁻¹)")
# ax.set_title("Carte ratio rugo 2D [1.4-200 mm⁻¹]")
#
# # ============================================================
# # PLOT 4 : PSD 1D Cut X - toutes sources
# # ============================================================
# ax = axes[1, 0]
# ax.loglog(nu_x2_mm[1:], psd_x2[1:], label="Cut X zygo full")
# ax.loglog(nu_x3_mm[1:], psd_x3[1:], label="Cut X zygo zoom")
# ax.loglog(nu_x1_mm[1:], psd_x[1:],  label="Cut X rugo")
# ax.loglog(nu_spec_mm, psd_spec_mm, '--', color='k', label="Spec")
# ax.axvspan(nu_x2_mm[-1], nu_x3_mm[-1], alpha=0.05, color='blue', label="Zoom only")
# ax.axvspan(nu_x3_mm[-1], nu_x1_mm[-1], alpha=0.05, color='gray', label="Plancher bruit rugo")
# # Bandes contractuelles
# ax.axvspan(0.02, 4,   alpha=0.08, color='purple', label="Bande spec 1 : 0.02-4 mm⁻¹")
# ax.axvspan(2,  500,   alpha=0.05, color='cyan',   label="Bande spec 2 : 2-500 mm⁻¹")
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("PSD [nm⁴]")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D Cut X - toutes sources")
# # ============================================================
# # PLOT 5 : PSD 1D Cut Y toutes sources vs spec
# # ============================================================
# ax = axes[1, 1]
# ax.loglog(nu_y2_mm[1:], psd_y2[1:], label="Cut Y zygo full")
# ax.loglog(nu_y3_mm[1:], psd_y3[1:], label="Cut Y zygo zoom")
# ax.loglog(nu_y1_mm[1:], psd_y[1:],  label="Cut Y rugo")
# ax.loglog(nu_spec_mm, psd_spec_mm, '--', color='k', label="Spec")
# ax.axvspan(nu_x2_mm[-1], nu_x3_mm[-1], alpha=0.05, color='blue', label="Zoom only")
# ax.axvspan(nu_x3_mm[-1], nu_x1_mm[-1], alpha=0.05, color='gray', label="Plancher bruit rugo")
# ax.axvspan(0.02, 4,   alpha=0.08, color='purple', label="Bande spec 1 : 0.02-4 mm⁻¹")
# ax.axvspan(2,  500,   alpha=0.05, color='cyan',   label="Bande spec 2 : 2-500 mm⁻¹")
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("PSD [nm⁴]")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D Cut Y - toutes sources")
#
#
# # ============================================================
# # PLOT 6 : Carte 2D ratio zygo zoom
# # ============================================================
# ax = axes[1, 2]
# Nu_x_z, Nu_y_z = np.meshgrid(nu_x3_mm, nu_y3_mm)
# Nu_r_zoom = np.sqrt(Nu_x_z**2 + Nu_y_z**2)
# Nu_r_zoom[Nu_r_zoom == 0] = np.nan
# ratio_zoom_2d = psd3[:ny3//2, :nx3//2] / (2e10 * Nu_r_zoom**(-3))
# im = ax.pcolormesh(Nu_x_z, Nu_y_z, np.log10(ratio_zoom_2d), cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# ax.set_xlabel("νx (mm⁻¹)")
# ax.set_ylabel("νy (mm⁻¹)")
# ax.set_title("Carte ratio zygo zoom 2D")
#
# # Stats finales
# print("\n=== BILAN CONFORMITÉ SW SN8 ===")
# for label, psd_2d, Nu_r_2d, nu_min, nu_max in [
#     ("zygo full", psd2[:ny2//2, :nx2//2],
#      np.sqrt(np.meshgrid(nu_x2_mm, nu_y2_mm)[0]**2 + np.meshgrid(nu_x2_mm, nu_y2_mm)[1]**2),
#      nu_x2_mm[1], nu_x2_mm[-1]),
#     ("zygo zoom", psd3[:ny3//2, :nx3//2],
#      np.sqrt(np.meshgrid(nu_x3_mm, nu_y3_mm)[0]**2 + np.meshgrid(nu_x3_mm, nu_y3_mm)[1]**2),
#      nu_x3_mm[1], nu_x3_mm[-1]),
#     ("rugo [5-100]", psd1[:ny1//2, :nx1//2], Nu_r_rugo, 5, 100),
# ]:
#     Nu_r_2d[Nu_r_2d == 0] = np.nan
#     ratio = psd_2d / (2e10 * Nu_r_2d**(-3))
#     mask = (Nu_r_2d >= nu_min) & (Nu_r_2d <= nu_max)
#     n_h = np.nansum((ratio > 1) & mask)
#     n_t = np.nansum(~np.isnan(ratio) & mask)
#     print(f"  {label:15s} : {n_h:5d}/{n_t} ({n_h/n_t*100:.3f}%), ratio max = {np.nanmax(ratio[mask]):.2f}")
#
# plt.suptitle("SW SN8 - Analyse conformité PSD complète", fontsize=13)
# plt.tight_layout()
# plt.show()
#
# from numpy import trapezoid
#
# # Fréquences en mm⁻¹
# nu_x1_mm = nu_x1 * 1e6
# nu_y1_mm = nu_y1 * 1e6
# nu_x2_mm = nu_x2 * 1e6
# nu_y2_mm = nu_y2 * 1e6
# nu_x3_mm = nu_x3 * 1e6
# nu_y3_mm = nu_y3 * 1e6
#
# SPEC_CONST = 2e10  # nm⁴, ν en mm⁻¹, comparaison directe avec PSD
#
# # Vérification RMS
# print("=== Vérification RMS spec ===")
# for nu_lo, nu_hi, rms_exp in [(0.02, 4, 2.5), (2, 500, 0.25)]:
#     nu_int = np.logspace(np.log10(nu_lo), np.log10(nu_hi), 1000)
#     psd_int = SPEC_CONST * nu_int**(-3)
#     rms = np.sqrt(2 * np.pi * trapezoid(nu_int * psd_int, nu_int)) / 1e6
#     print(f"  [{nu_lo}-{nu_hi} mm⁻¹] : RMS = {rms:.4f} nm, attendu = {rms_exp} nm")
#
# # Grille spec pour les plots
# nu_spec_mm = np.logspace(np.log10(min(nu_x2_mm[1], nu_x1_mm[1])),
#                           np.log10(nu_x1_mm[-1]), 500)
# psd_spec_plot = SPEC_CONST * nu_spec_mm**(-3)
#
# # Coupes 1D
# psd_x  = psd1[0, :]
# psd_y  = psd1[:, 0]
# psd_x2 = psd2[0, :]
# psd_y2 = psd2[:, 0]
# psd_x3 = psd3[0, :]
# psd_y3 = psd3[:, 0]
#
# # Grilles 2D pour les cartes
# Nu_x_r, Nu_y_r = np.meshgrid(nu_x1_mm, nu_y1_mm)
# Nu_r_rugo = np.sqrt(Nu_x_r**2 + Nu_y_r**2)
# Nu_r_rugo[Nu_r_rugo == 0] = np.nan
#
# Nu_x_z, Nu_y_z = np.meshgrid(nu_x3_mm, nu_y3_mm)
# Nu_r_zoom = np.sqrt(Nu_x_z**2 + Nu_y_z**2)
# Nu_r_zoom[Nu_r_zoom == 0] = np.nan
#
# fig, axes = plt.subplots(2, 3, figsize=(18, 11))
#
# # ============================================================
# # PLOT 1 : PSD 1D rugo vs spec
# # ============================================================
# ax = axes[0, 0]
# ax.loglog(nu_x1_mm[1:], psd_x[1:], label="Cut X rugo")
# ax.loglog(nu_y1_mm[1:], psd_y[1:], label="Cut Y rugo", alpha=0.7)
# ax.loglog(nu_spec_mm, psd_spec_plot, '--', color='k', label="Spec 2e10 ν⁻³")
# ax.axvspan(1.4,   5,   alpha=0.1, color='red',    label="Forme résiduelle")
# ax.axvspan(5,   100,   alpha=0.1, color='green',  label="Gamme fiable")
# ax.axvspan(100, 500,   alpha=0.1, color='orange', label="Plancher bruit")
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("PSD [nm⁴]")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD rugo 1D vs spec")
#
# # ============================================================
# # PLOT 2 : Ratio 1D rugo/spec
# # ============================================================
# ax = axes[0, 1]
# spec_x = SPEC_CONST * nu_x1_mm[1:]**(-3)
# spec_y = SPEC_CONST * nu_y1_mm[1:]**(-3)
# ax.semilogx(nu_x1_mm[1:], np.log10(psd_x[1:] / spec_x), label="Cut X rugo")
# ax.semilogx(nu_y1_mm[1:], np.log10(psd_y[1:] / spec_y), label="Cut Y rugo", alpha=0.7)
# ax.axhline(0, color='k', linewidth=1.5, linestyle='--', label="Limite spec")
# ax.axvspan(1.4,   5,   alpha=0.1, color='red')
# ax.axvspan(5,   100,   alpha=0.1, color='green')
# ax.axvspan(100, 500,   alpha=0.1, color='orange')
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("log10(PSD/spec)")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("Ratio rugo/spec 1D")
# ax.set_ylim(-3, 3)
# ax.set_xlim(nu_x1_mm[1], 500)
#
# # ============================================================
# # PLOT 3 : Carte 2D ratio rugo
# # ============================================================
# ax = axes[0, 2]
# ratio_rugo_2d = psd1[:ny1//2, :nx1//2] / (SPEC_CONST * Nu_r_rugo**(-3))
# ratio_rugo_masked = np.where((Nu_r_rugo >= 1.4) & (Nu_r_rugo <= 200),
#                               ratio_rugo_2d, np.nan)
# im = ax.pcolormesh(Nu_x_r, Nu_y_r, np.log10(ratio_rugo_masked),
#                    cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# ax.set_xlabel("νx (mm⁻¹)")
# ax.set_ylabel("νy (mm⁻¹)")
# ax.set_title("Carte ratio rugo 2D [1.4-200 mm⁻¹]")
#
# # ============================================================
# # PLOT 4 : PSD 1D Cut Y toutes sources
# # ============================================================
# ax = axes[1, 0]
# ax.loglog(nu_y2_mm[1:], psd_y2[1:], label="Cut Y zygo full")
# ax.loglog(nu_y3_mm[1:], psd_y3[1:], label="Cut Y zygo zoom")
# ax.loglog(nu_y1_mm[1:], psd_y[1:],  label="Cut Y rugo")
# ax.loglog(nu_spec_mm, psd_spec_plot, '--', color='k', label="Spec")
# ax.axvspan(0.02, 4,   alpha=0.08, color='purple', label="Bande spec 1 : 0.02-4 mm⁻¹")
# ax.axvspan(2,  500,   alpha=0.05, color='cyan',   label="Bande spec 2 : 2-500 mm⁻¹")
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("PSD [nm⁴]")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D Cut Y - toutes sources")
#
# # ============================================================
# # PLOT 5 : PSD 1D Cut X toutes sources
# # ============================================================
# ax = axes[1, 1]
# ax.loglog(nu_x2_mm[1:], psd_x2[1:], label="Cut X zygo full")
# ax.loglog(nu_x3_mm[1:], psd_x3[1:], label="Cut X zygo zoom")
# ax.loglog(nu_x1_mm[1:], psd_x[1:],  label="Cut X rugo")
# ax.loglog(nu_spec_mm, psd_spec_plot, '--', color='k', label="Spec")
# ax.axvspan(nu_x2_mm[-1], nu_x3_mm[-1], alpha=0.05, color='blue',
#            label="Zoom only")
# ax.axvspan(nu_x3_mm[-1], nu_x1_mm[-1], alpha=0.05, color='gray',
#            label="Plancher bruit rugo")
# ax.axvspan(0.02, 4,   alpha=0.08, color='purple', label="Bande spec 1 : 0.02-4 mm⁻¹")
# ax.axvspan(2,  500,   alpha=0.05, color='cyan',   label="Bande spec 2 : 2-500 mm⁻¹")
# ax.set_xlabel("ν (mm⁻¹)")
# ax.set_ylabel("PSD [nm⁴]")
# ax.legend(fontsize=7)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D Cut X - toutes sources")
#
# # ============================================================
# # PLOT 6 : Carte 2D ratio zygo zoom
# # ============================================================
# ax = axes[1, 2]
# ratio_zoom_2d = psd3[:ny3//2, :nx3//2] / (SPEC_CONST * Nu_r_zoom**(-3))
# im = ax.pcolormesh(Nu_x_z, Nu_y_z, np.log10(ratio_zoom_2d),
#                    cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# ax.set_xlabel("νx (mm⁻¹)")
# ax.set_ylabel("νy (mm⁻¹)")
# ax.set_title("Carte ratio zygo zoom 2D")
#
# # Stats finales
# print("\n=== BILAN CONFORMITÉ SW SN8 ===")
# for label, psd_2d, Nu_r_2d, nu_min, nu_max in [
#     ("zygo full",    psd2[:ny2//2, :nx2//2],
#      np.sqrt(np.meshgrid(nu_x2_mm, nu_y2_mm)[0]**2 +
#              np.meshgrid(nu_x2_mm, nu_y2_mm)[1]**2),
#      nu_x2_mm[1], nu_x2_mm[-1]),
#     ("zygo zoom",    psd3[:ny3//2, :nx3//2],
#      np.sqrt(np.meshgrid(nu_x3_mm, nu_y3_mm)[0]**2 +
#              np.meshgrid(nu_x3_mm, nu_y3_mm)[1]**2),
#      nu_x3_mm[1], nu_x3_mm[-1]),
#     ("rugo [5-100]", psd1[:ny1//2, :nx1//2], Nu_r_rugo, 5, 100),
# ]:
#     Nu_r_2d[Nu_r_2d == 0] = np.nan
#     ratio = psd_2d / (SPEC_CONST * Nu_r_2d**(-3))
#     mask = (Nu_r_2d >= nu_min) & (Nu_r_2d <= nu_max)
#     n_h = np.nansum((ratio > 1) & mask)
#     n_t = np.nansum(~np.isnan(ratio) & mask)
#     print(f"  {label:15s} : {n_h:5d}/{n_t} ({n_h/n_t*100:.3f}%), "
#           f"ratio max = {np.nanmax(ratio[mask]):.2f}")
#
# plt.suptitle("SW SN8 - Analyse conformité PSD complète", fontsize=13)
# plt.tight_layout()
# plt.show()
#
#
# from numpy import trapezoid
#
# # Spec en nm⁻¹
# SPEC_CONST_NM = 2e10 * 1e-18  # nm⁴, ν en nm⁻¹
#
# # Vérification RMS
# print("=== Vérification RMS spec en nm⁻¹ ===")
# for nu_lo_mm, nu_hi_mm, rms_exp in [(0.02, 4, 2.5), (2, 500, 0.25)]:
#     nu_lo_nm = nu_lo_mm * 1e-6
#     nu_hi_nm = nu_hi_mm * 1e-6
#     nu_int = np.logspace(np.log10(nu_lo_nm), np.log10(nu_hi_nm), 1000)
#     psd_int = SPEC_CONST_NM * nu_int**(-3)
#     rms = np.sqrt(2 * np.pi * trapezoid(nu_int * psd_int, nu_int))
#     print(f"  [{nu_lo_mm}-{nu_hi_mm} mm⁻¹] : RMS = {rms:.4f} nm, attendu = {rms_exp} nm")
#
# # Grille spec en nm⁻¹
# nu_spec_nm = np.logspace(np.log10(min(nu_x2[1], nu_x1[1])),
#                           np.log10(nu_x1[-1]), 500)
# psd_spec_plot = SPEC_CONST_NM * nu_spec_nm**(-3)
#
# # Coupes 1D
# psd_x  = psd1[0, :]
# psd_y  = psd1[:, 0]
# psd_x2 = psd2[0, :]
# psd_y2 = psd2[:, 0]
# psd_x3 = psd3[0, :]
# psd_y3 = psd3[:, 0]
#
# # Grilles 2D pour les cartes ratio
# Nu_x_r, Nu_y_r = np.meshgrid(nu_x1, nu_y1)
# Nu_r_rugo = np.sqrt(Nu_x_r**2 + Nu_y_r**2)
# Nu_r_rugo[Nu_r_rugo == 0] = np.nan
#
# Nu_x_2, Nu_y_2 = np.meshgrid(nu_x2, nu_y2)
# Nu_r_full = np.sqrt(Nu_x_2**2 + Nu_y_2**2)
# Nu_r_full[Nu_r_full == 0] = np.nan
#
# Nu_x_z, Nu_y_z = np.meshgrid(nu_x3, nu_y3)
# Nu_r_zoom = np.sqrt(Nu_x_z**2 + Nu_y_z**2)
# Nu_r_zoom[Nu_r_zoom == 0] = np.nan
#
# fig, axes = plt.subplots(2, 3, figsize=(18, 11))
#
# # ============================================================
# # LIGNE 1 : Coupes 1D
# # ============================================================
#
# # PLOT 1 : Cut X toutes sources
# ax = axes[0, 0]
# ax.loglog(nu_x2[1:], psd_x2[1:], label="Cut X zygo full")
# ax.loglog(nu_x3[1:], psd_x3[1:], label="Cut X zygo zoom")
# ax.loglog(nu_y1[1:], psd_y[1:],  label="Cut X rugo")
# ax.loglog(nu_spec_nm, psd_spec_plot, '--', color='k', label="Spec")
# ax.set_xlabel(r"$\nu$ (nm$^{-1}$)")
# ax.set_ylabel(r"PSD [nm$^4$]")
# ax.legend(fontsize=8)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D - Cut X")
#
# # PLOT 2 : Cut Y toutes sources
# ax = axes[0, 1]
# ax.loglog(nu_y2[1:], psd_y2[1:], label="Cut Y zygo full")
# ax.loglog(nu_y3[1:], psd_y3[1:], label="Cut Y zygo zoom")
# ax.loglog(nu_x1[1:], psd_x[1:],  label="Cut Y rugo")
# ax.loglog(nu_spec_nm, psd_spec_plot, '--', color='k', label="Spec")
# ax.set_xlabel(r"$\nu$ (nm$^{-1}$)")
# ax.set_ylabel(r"PSD [nm$^4$]")
# ax.legend(fontsize=8)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D - Cut Y")
#
# # PLOT 3 : Cut X et Y rugo
# ax = axes[0, 2]
# ax.loglog(nu_x1[1:], psd_x[1:], label="Cut X rugo")
# ax.loglog(nu_y1[1:], psd_y[1:], label="Cut Y rugo", alpha=0.7)
# ax.loglog(nu_spec_nm, psd_spec_plot, '--', color='k', label="Spec")
# ax.set_xlabel(r"$\nu$ (nm$^{-1}$)")
# ax.set_ylabel(r"PSD [nm$^4$]")
# ax.legend(fontsize=8)
# ax.grid(True, which="both", ls="--")
# ax.set_title("PSD 1D - Rugo X vs Y")
#
# # ============================================================
# # LIGNE 2 : Cartes ratio 2D
# # ============================================================
#
# # PLOT 4 : Carte ratio zygo full
# ax = axes[1, 0]
# ratio_full_2d = psd2[:ny2//2, :nx2//2] / (SPEC_CONST_NM * Nu_r_full**(-3))
# im = ax.pcolormesh(Nu_x_2, Nu_y_2, np.log10(ratio_full_2d),
#                    cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# ax.set_xlabel(r"$\nu_x$ (nm$^{-1}$)")
# ax.set_ylabel(r"$\nu_y$ (nm$^{-1}$)")
# ax.set_title("Carte ratio zygo full 2D")
#
# # PLOT 5 : Carte ratio zygo zoom
# ax = axes[1, 1]
# ratio_zoom_2d = psd3[:ny3//2, :nx3//2] / (SPEC_CONST_NM * Nu_r_zoom**(-3))
# im = ax.pcolormesh(Nu_x_z, Nu_y_z, np.log10(ratio_zoom_2d),
#                    cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# ax.set_xlabel(r"$\nu_x$ (nm$^{-1}$)")
# ax.set_ylabel(r"$\nu_y$ (nm$^{-1}$)")
# ax.set_title("Carte ratio zygo zoom 2D")
#
# # PLOT 6 : Carte ratio rugo
# ax = axes[1, 2]
# ratio_rugo_2d = psd1[:ny1//2, :nx1//2] / (SPEC_CONST_NM * Nu_r_rugo**(-3))
# im = ax.pcolormesh(Nu_x_r, Nu_y_r, np.log10(ratio_rugo_2d),
#                    cmap='RdBu_r', vmin=-2, vmax=2)
# plt.colorbar(im, ax=ax, label='log10(PSD/spec)')
# nu_nyq = min(1/(2*xpix1*1e6), 1/(2*ypix1*1e6))  # nm⁻¹
# ax.set_xlim(0, nu_nyq)
# ax.set_ylim(0, nu_nyq)
# ax.set_xlabel(r"$\nu_x$ (nm$^{-1}$)")
# ax.set_ylabel(r"$\nu_y$ (nm$^{-1}$)")
# ax.set_title("Carte ratio rugo 2D")
#
# # Stats finales
# print("\n=== BILAN CONFORMITÉ SW SN8 ===")
# for label, psd_2d, Nu_r_2d, nu_min, nu_max in [
#     ("zygo full",         psd2[:ny2//2, :nx2//2], Nu_r_full, nu_x2[1], nu_x2[-1]),
#     ("zygo zoom",         psd3[:ny3//2, :nx3//2], Nu_r_zoom, nu_x3[1], nu_x3[-1]),
#     ("rugo [5-100 mm⁻¹]", psd1[:ny1//2, :nx1//2], Nu_r_rugo, 5e-6,    100e-6),
# ]:
#     Nu_r_2d[Nu_r_2d == 0] = np.nan
#     ratio = psd_2d / (SPEC_CONST_NM * Nu_r_2d**(-3))
#     mask = (Nu_r_2d >= nu_min) & (Nu_r_2d <= nu_max)
#     n_h = np.nansum((ratio > 1) & mask)
#     n_t = np.nansum(~np.isnan(ratio) & mask)
#     print(f"  {label:20s} : {n_h:5d}/{n_t} ({n_h/n_t*100:.3f}%), "
#           f"ratio max = {np.nanmax(ratio[mask]):.2f}")
#
# plt.suptitle("SW SN8 - Analyse conformité PSD complète", fontsize=13)
# plt.tight_layout()
# plt.show()
#
# # Bande 1 : [0.02-4 mm⁻¹] → zygo full (meilleure couverture basse fréquence)
# # Bande 2 : [2-500 mm⁻¹] → zygo zoom + rugo combinés
#
# print("=== RMS sur bandes contractuelles complètes ===")
#
# # Bande 1 : zygo full sur [0.046-4 mm⁻¹]
# Nu_x_2, Nu_y_2 = np.meshgrid(nu_x2, nu_y2)
# Nu_r_2 = np.sqrt(Nu_x_2**2 + Nu_y_2**2)
# Nu_r_2[Nu_r_2 == 0] = np.nan
# Nu_r_flat2 = Nu_r_2.flatten()
# psd_flat2 = psd2[:ny2//2, :nx2//2].flatten()
# valid2 = ~np.isnan(Nu_r_flat2) & ~np.isnan(psd_flat2)
#
# nu_bins2 = np.logspace(np.log10(nu_x2[1]), np.log10(nu_x2[-1]), 200)
# psd_rad2, nu_edges2, _ = binned_statistic(
#     Nu_r_flat2[valid2], psd_flat2[valid2], statistic='mean', bins=nu_bins2)
# nu_cent2 = 0.5 * (nu_edges2[1:] + nu_edges2[:-1])
# valid_rad2 = ~np.isnan(psd_rad2)
#
# nu_lo, nu_hi = 0.02e-6, 4e-6
# mask = valid_rad2 & (nu_cent2 >= nu_lo) & (nu_cent2 <= nu_hi)
# rms_b1 = np.sqrt(2 * np.pi * trapezoid(nu_cent2[mask] * psd_rad2[mask], nu_cent2[mask]))
# rms_spec_b1 = np.sqrt(2 * np.pi * trapezoid(
#     nu_cent2[mask] * SPEC_CONST_NM * nu_cent2[mask]**(-3), nu_cent2[mask]))
# print(f"Bande [0.02-4 mm⁻¹] via zygo full [{nu_cent2[mask][0]*1e6:.3f}-{nu_cent2[mask][-1]*1e6:.2f} mm⁻¹] :")
# print(f"  RMS mesuré = {rms_b1:.4f} nm")
# print(f"  RMS spec   = {rms_spec_b1:.4f} nm  (spec complète = 2.5 nm)")
# print(f"  Ratio      = {rms_b1/rms_spec_b1:.2f}")
#
# # Bande 2 : zygo zoom [2-9.6 mm⁻¹] + rugo [9.6-454 mm⁻¹] combinés
# # Zygo zoom radial
# Nu_x_3, Nu_y_3 = np.meshgrid(nu_x3, nu_y3)
# Nu_r_3 = np.sqrt(Nu_x_3**2 + Nu_y_3**2)
# Nu_r_3[Nu_r_3 == 0] = np.nan
# Nu_r_flat3 = Nu_r_3.flatten()
# psd_flat3 = psd3[:ny3//2, :nx3//2].flatten()
# valid3 = ~np.isnan(Nu_r_flat3) & ~np.isnan(psd_flat3)
#
# nu_bins3 = np.logspace(np.log10(nu_x3[1]), np.log10(nu_x3[-1]), 200)
# psd_rad3, nu_edges3, _ = binned_statistic(
#     Nu_r_flat3[valid3], psd_flat3[valid3], statistic='mean', bins=nu_bins3)
# nu_cent3 = 0.5 * (nu_edges3[1:] + nu_edges3[:-1])
# valid_rad3 = ~np.isnan(psd_rad3)
#
# # Rugo radial déjà calculé : nu_centers, psd_radial, valid
#
# # Combiner zoom [2-9.6 mm⁻¹] + rugo [9.6-454 mm⁻¹]
# nu_split = nu_x3[-1]  # fréquence de coupure entre zoom et rugo
#
# mask_zoom = valid_rad3 & (nu_cent3 >= 2e-6) & (nu_cent3 <= nu_split)
# mask_rugo = valid & (nu_centers >= nu_split) & (nu_centers <= 500e-6)
#
# nu_combined  = np.concatenate([nu_cent3[mask_zoom],  nu_centers[mask_rugo]])
# psd_combined = np.concatenate([psd_rad3[mask_zoom],  psd_radial[mask_rugo]])
#
# rms_b2 = np.sqrt(2 * np.pi * trapezoid(nu_combined * psd_combined, nu_combined))
#
# nu_lo2, nu_hi2 = 2e-6, 500e-6
# nu_spec_int = np.logspace(np.log10(nu_lo2), np.log10(nu_hi2), 1000)
# rms_spec_b2 = np.sqrt(2 * np.pi * trapezoid(
#     nu_spec_int * SPEC_CONST_NM * nu_spec_int**(-3), nu_spec_int))
#
# print(f"\nBande [2-500 mm⁻¹] via zygo zoom + rugo :")
# print(f"  Zoom  : {nu_cent3[mask_zoom][0]*1e6:.2f} - {nu_cent3[mask_zoom][-1]*1e6:.2f} mm⁻¹")
# print(f"  Rugo  : {nu_centers[mask_rugo][0]*1e6:.2f} - {nu_centers[mask_rugo][-1]*1e6:.2f} mm⁻¹")
# print(f"  RMS mesuré = {rms_b2:.4f} nm")
# print(f"  RMS spec   = {rms_spec_b2:.4f} nm  (spec complète = 0.25 nm)")
# print(f"  Ratio      = {rms_b2/rms_spec_b2:.2f}")
#
