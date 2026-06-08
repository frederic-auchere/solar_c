import numpy as np
import matplotlib.pyplot as plt
import os
from optical import sw_substrate, lw_substrate, rectangular_sw_substrate, rectangular_lw_substrate
from optics.zygo import SagData
from optics import surfaces
import matplotlib as mpl
from optical.utils import write_zygo_dat



mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
# =====================================================================
# CASES
# =====================================================================
cases = [
    # dict(
    #     name="LW_SN1",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/LW_SN1/",
    #     file_nanomefos="new_jan_2026/25264-301-PL002_SubstratLW-SN1 ASPH 3_ZU_woTilt.datx",
    #     dx_nanomefos=-30, dy_nanomefos=189,
    #     file_zygo="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/LW/FM_LW_SN1/Zygo/Form/29012026/report_substrate_FM_form_LW_SN1_binning1.fits"
    # ),
    dict(
        name="SW_SN5",
        path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/",
        # file_nanomefos="25264_Subsrat_SW_SN5 2_ZU_woTilt.datx",
        #  dx_nanomefos=1, dy_nanomefos=189,
        file_nanomefos='25264_Subsrat_SW_SN5 1_full_woTilt.datx',
        dx_nanomefos=60, dy_nanomefos=140,
        file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/substrate_FM_form_SW_SN5_bertin.fits',
        # file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/substrate_FM_form_SW_SN5.fits',
    ),
    dict(
        name="SW_SN7",
        path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN7/",
        # file_nanomefos="25264_Subsrat_SW_SN7 3_ZU_woTilt.datx",
        # dx_nanomefos=10, dy_nanomefos=189,
        file_nanomefos='25264_Subsrat_SW_SN7 1_Full_woTilt.datx',
        dx_nanomefos=60, dy_nanomefos=140,
        file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN7/Zygo/Form/07012026/substrate_FM_form_SW_SN7_bertin.fits',
    ),
    dict(
        name="SW_SN8",
        path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
        # file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
        # dx_nanomefos=0, dy_nanomefos=189,
        file_nanomefos='25264_Subsrat_SW_SN8 1_Full_woTilt.datx',
        dx_nanomefos=60, dy_nanomefos=140,
        file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN8/Zygo/Form/20012026/substrate_FM_form_SW_SN8_bertin.fits',
    ),
    # dict(
    #     name="SW_SN1",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN1/new_fev_2026/",
    #     file_nanomefos="26-02-16_SW-SN1_mesZygo.datx",
    #     dx_nanomefos=0, dy_nanomefos=189,
    #     file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_vertex.fits',
    # ),
    # dict(
    #     name="SW_SN1_reprise",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
    #     file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
    #     dx_nanomefos=2, dy_nanomefos=189,
    #     file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Reprise_Mai/Zygo/Form/20260513/substrates_template_FM_form_casquette_sathyne_finale_2.fits',
    # ),
    # dict(
    #     name="SW_SN1_tilt",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
    #     file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
    #     dx_nanomefos=2, dy_nanomefos=189,
    #     file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Reprise_Mai/Zygo/Form/20260513/substrates_template_FM_form_casquette_sathyne_finale_2.fits',
    # ),
]

# =====================================================================
# FONCTIONS
# =====================================================================
def load_map_nano(file, dx, dy, scale=1.0):
    data = SagData(file, dx=dx, dy=dy, theta=0, binning=1, auto_crop=False)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(surf, rectangular_sw_substrate.aperture, rectangular_sw_substrate.useful_area)
    return subs

def load_map_zygo_sw(file, scale=1.0):
    data = SagData(file, theta=0, binning=1, auto_crop=True)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(surf, rectangular_sw_substrate.aperture, rectangular_sw_substrate.useful_area)
    return subs

def load_map_zygo_lw(file, scale=1.0):
    data = SagData(file, theta=0, binning=1, auto_crop=True)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(surf, rectangular_lw_substrate.aperture, rectangular_lw_substrate.useful_area)
    return subs

def stats(arr, label):
    valid = arr[np.isfinite(arr)]
    rms = np.sqrt(np.mean(valid**2))
    pv = np.percentile(valid, 99.9) - np.percentile(valid, 0.1)
    print(f"  {label:40s} | RMS = {rms:.2f} nm | PV = {pv:.2f} nm")

# =====================================================================
# PASSE 1 : chargement et calcul des biais
# =====================================================================
print("=== Chargement des cartes et calcul des biais ===\n")

maps_zygo  = {}
maps_nano  = {}
maps_biais = {}

for case in cases:
    name = case["name"]
    print(f"Chargement {name}...")

    subs_nano = load_map_nano(
        os.path.join(case["path_nanomefos"], case["file_nanomefos"]),
        case["dx_nanomefos"], case["dy_nanomefos"]
    )
    if name.startswith("LW"):
        subs_zygo = load_map_zygo_lw(case["file_zygo"])
    else:
        subs_zygo = load_map_zygo_sw(case["file_zygo"])

    grid_nano=subs_nano.grid()
    map_nano=subs_nano.sag().data *1e6

    map_zygo=subs_zygo.sag(grid_nano).data
    # plt.imshow(map_nano, origin='lower', extent=rectangular_sw_substrate.limits)

    # biais = zygo - nano, valide seulement là où les deux sont définis
    biais = map_zygo - map_nano
    # biais[~np.isfinite(biais)] = np.nan
    # plt.imshow(biais, origin='lower', extent=rectangular_sw_substrate.limits)
    maps_zygo[name]  = map_zygo
    maps_nano[name]  = map_nano
    maps_biais[name] = biais

    stats(map_zygo, f"{name} zygo")
    stats(map_nano,  f"{name} nano")
    stats(biais,     f"{name} biais")
    print()

#
# =====================================================================
# PASSE 2 : carte de correction moyenne
# =====================================================================
print("=== Calcul de la correction moyenne ===\n")

stack_biais = np.array([maps_biais[c["name"]] for c in cases])
correction_moyenne = np.nanmean(stack_biais, axis=0)   # biais moyen
correction_std     = np.nanstd(stack_biais,  axis=0)   # variabilité
carte_correction   = correction_moyenne                # à appliquer au zygo

stats(correction_moyenne, "Biais moyen")
stats(correction_std,     "Ecart-type inter-substrats")
print()
# plt.imshow(correction_moyenne, origin='lower')
# # =====================================================================
# # PASSE 3 : application de la correction et stats finales
# # =====================================================================
print("=== Application de la correction moyenne ===\n")

maps_zygo_corr = {}
for case in cases:
    name = case["name"]
    map_zygo = maps_zygo[name]
    mask_nan = ~np.isfinite(map_zygo)

    map_corr = map_zygo + carte_correction
    map_corr[mask_nan] = np.nan
    maps_zygo_corr[name] = map_corr

    stats(map_zygo, f"{name} original")
    stats(map_corr, f"{name} corrigé")
    print()
#
# =====================================================================
# FIGURES
# =====================================================================
#
# --- Figure 1 : biais par substrat ---
# n = len(cases)
# fig, axes = plt.subplots(2, n, figsize=(3*n, 7))
# vabs_biais = np.nanpercentile(np.abs(stack_biais), 99)
#
# for i, case in enumerate(cases):
#     name = case["name"]
#     im0 = axes[0, i].imshow(maps_biais[name], origin="lower",
#                              cmap="RdBu_r", vmin=-vabs_biais, vmax=vabs_biais)
#     axes[0, i].set_title(f"{name}\nbiais")
#     plt.colorbar(im0, ax=axes[0, i], label="nm")
#
#     im1 = axes[1, i].imshow(maps_zygo_corr[name], origin="lower", cmap="viridis")
#     axes[1, i].set_title(f"{name}\nzygo corrigé")
#     plt.colorbar(im1, ax=axes[1, i], label="nm")
#
# plt.suptitle("Biais (zygo - nano) par substrat et zygo corrigé", fontsize=13)
# plt.tight_layout()
# plt.show()
#
# --- Figure 2 : correction moyenne et variabilité ---
# fig, axes = plt.subplots(1, 3, figsize=(14, 5))
#
# vabs = np.nanpercentile(np.abs(correction_moyenne), 99)
#
# im0 = axes[0].imshow(correction_moyenne, origin="lower",
#                      cmap="RdBu_r", vmin=-vabs, vmax=vabs)
# axes[0].set_title("Biais moyen (zygo - nano)")
# plt.colorbar(im0, ax=axes[0], label="nm")
#
# im1 = axes[1].imshow(correction_std, origin="lower", cmap="hot_r")
# axes[1].set_title("Ecart-type inter-substrats")
# plt.colorbar(im1, ax=axes[1], label="nm")
#
# ratio = correction_std / (np.abs(correction_moyenne) + 1e-6)
# im2 = axes[2].imshow(ratio, origin="lower", cmap="hot_r", vmin=0, vmax=2)
# axes[2].set_title("Variabilité relative (std / |biais moyen|)")
# plt.colorbar(im2, ax=axes[2])
#
# plt.suptitle("Carte de correction moyenne", fontsize=13)
# plt.tight_layout()
# plt.show()

# # --- Figure 3 : avant / après par substrat ---
# for case in cases:
#     name = case["name"]
#     map_zygo = maps_zygo[name]
#     map_corr = maps_zygo_corr[name]
#
#     vmin = np.nanpercentile(map_zygo, 0.5)
#     vmax = np.nanpercentile(map_zygo, 99.5)
#
#     fig, axes = plt.subplots(1, 3, figsize=(13, 4))
#
#     im0 = axes[0].imshow(map_zygo, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
#     axes[0].set_title("Zygo original")
#     plt.colorbar(im0, ax=axes[0], label="nm")
#
#     im1 = axes[1].imshow(map_corr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
#     axes[1].set_title("Zygo corrigé")
#     plt.colorbar(im1, ax=axes[1], label="nm")
#
#     diff = map_zygo - map_corr
#     vabs_d = np.nanpercentile(np.abs(diff), 99)
#     im2 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vabs_d, vmax=vabs_d)
#     axes[2].set_title("Correction appliquée")
#     plt.colorbar(im2, ax=axes[2], label="nm")
#
#     plt.suptitle(f"{name} — correction par biais moyen", fontsize=12)
#     plt.tight_layout()
#     plt.show()


# --- Figure : nano / zygo original / zygo corrigé / correction — tous substrats ---
n = len(cases)
fig, axes = plt.subplots(n, 4, figsize=(18, 5 * n))

vabs_biais = np.nanpercentile(np.abs(stack_biais), 99)

for i, case in enumerate(cases):
    name     = case["name"]
    map_nano = maps_nano[name]
    map_zygo = maps_zygo[name]
    map_corr = maps_zygo_corr[name]

    all_valid = np.concatenate([
        map_nano[np.isfinite(map_nano)],
        map_zygo[np.isfinite(map_zygo)],
        map_corr[np.isfinite(map_corr)],
    ])
    # vmin = np.percentile(all_valid, 0.5)
    # vmax = np.percentile(all_valid, 99.5)
    vmin=-100
    vmax=100
    rms_orig = np.sqrt(np.nanmean(map_zygo**2))
    rms_corr = np.sqrt(np.nanmean(map_corr**2))
    pv_orig  = np.nanpercentile(map_zygo, 99.9) - np.nanpercentile(map_zygo, 0.1)
    pv_corr  = np.nanpercentile(map_corr, 99.9) - np.nanpercentile(map_corr, 0.1)

    im0 = axes[i, 0].imshow(map_nano, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f"{name} — Nanomefos")
    plt.colorbar(im0, ax=axes[i, 0], label="nm")

    im1 = axes[i, 1].imshow(map_zygo, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f"{name} — Zygo original\nRMS={rms_orig:.1f} nm  PV={pv_orig:.1f} nm")
    plt.colorbar(im1, ax=axes[i, 1], label="nm")

    im2 = axes[i, 2].imshow(map_corr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[i, 2].set_title(f"{name} — Zygo corrigé\nRMS={rms_corr:.1f} nm  PV={pv_corr:.1f} nm")
    plt.colorbar(im2, ax=axes[i, 2], label="nm")

    diff = map_zygo - map_corr
    vabs_d = np.nanpercentile(np.abs(diff), 99)
    im3 = axes[i, 3].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-vabs_d, vmax=vabs_d)
    axes[i, 3].set_title(f"{name} — Correction appliquée")
    plt.colorbar(im3, ax=axes[i, 3], label="nm")

# plt.suptitle("Avant / après correction par biais moyen", fontsize=13)
plt.tight_layout()
plt.show()

for case in cases:
    if case["name"]=='SW_SN5':
        # On prend un fichier SW quelconque pour initialiser
        ref_case = next(c for c in cases if not c["name"].startswith("LW"))
        biais_sagdata = SagData(ref_case["file_zygo"], theta=0, binning=1, auto_crop=True)

        # Force l'init lazy pour remplir gx, gy, dx, dy
        # _ = biais_sagdata.gx

        # On écrase juste le sag avec le biais moyen
        biais_sagdata._sag = correction_moyenne.copy()

        measured_surface   = surfaces.MeasuredSurface(biais_sagdata, alpha=0, beta=0, gamma=0)
        measured_substrate = surfaces.Substrate( measured_surface, rectangular_sw_substrate.aperture, rectangular_sw_substrate.useful_area)


        sag_biais_rect = np.asarray(measured_substrate.sag(grid_nano).data)
        # write_zygo_dat('/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/carte_correction_sw/carte_correction_SW_shape.dat',sag_biais_rect, subs_nano.x_grid_step)

        # Figure
        fig, ax = plt.subplots(figsize=(10, 8))
        vabs = np.nanpercentile(np.abs(sag_biais_rect), 99)
        im = ax.imshow(sag_biais_rect, origin="lower", vmin=-vabs, vmax=vabs)
        ax.set_title("Biais moyen — substrat rectangulaire",fontsize=14)
        plt.colorbar(im, ax=ax, label="nm")
        plt.tight_layout()
        plt.show()