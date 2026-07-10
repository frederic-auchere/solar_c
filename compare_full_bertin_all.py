from astropy.io import fits
import matplotlib.pyplot as plt
from optical import sw_substrate,lw_substrate, rectangular_sw_substrate, rectangular_lw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np
from fitting import sfit
path_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/'
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
    data = SagData(file, dx=dx, dy=dy, theta=0, binning=1, auto_crop=True)
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




print("=== Chargement des cartes et calcul des biais ===\n")

maps_zygo  = {}
maps_nano  = {}
maps_biais = {}

fig, axes = plt.subplots(1, len(cases), figsize=(15, 8))
axes = axes.flatten()


for ax, case in zip(axes, cases):
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

    im = ax.imshow(map_nano, origin="lower")
    ax.set_title(case["name"])
    # ax.set_xticks([])
    # ax.set_yticks([])
all_data = np.concatenate([
    maps_zygo[c["name"]][np.isfinite(maps_zygo[c["name"]])].ravel()
    for c in cases
])
vmin_global = float(np.percentile(all_data, 1))
vmax_global = float(np.percentile(all_data, 99))
for ax in axes:
    for img in ax.get_images():
        img.set_clim(vmin_global, vmax_global)

cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Zygo [nm]")

def on_ylim_change(ax_cb):
    vmin, vmax = ax_cb.get_ylim()
    for a in axes:
        for img in a.get_images():
            img.set_clim(vmin, vmax)
    fig.canvas.draw_idle()

cax.callbacks.connect("ylim_changed", on_ylim_change)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # laisse de la place à droite
# plt.savefig(path_zygo + "nanomefos_ALL.png", dpi=300)
plt.show()
