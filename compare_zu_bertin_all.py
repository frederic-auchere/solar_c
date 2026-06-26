from astropy.io import fits
import matplotlib.pyplot as plt
from optical import sw_substrate,lw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np
from fitting import sfit
path_individuel = "Y:/02- Engineering"
cases = [
    dict(
        name="LW_SN1",
        path_nanomefos= path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/LW_SN1/",
        file_nanomefos="new_jan_2026/25264-301-PL002_SubstratLW-SN1 ASPH 3_ZU_woTilt.datx",
        dx_nanomefos=-30, dy_nanomefos=189,
        file_zygo= path_individuel + "/08 - Metrology/01 - Optics/07 - Measurements/FM/LW/FM_LW_SN1/Zygo/Form/29012026/report_substrate_FM_form_LW_SN1_binning1.fits"
    ),
    dict(
        name="SW_SN5",
        path_nanomefos=path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/",
        file_nanomefos="25264_Subsrat_SW_SN5 2_ZU_woTilt.datx",
        dx_nanomefos=1, dy_nanomefos=189,
        file_zygo= path_individuel + "/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/substrate_FM_form_SW_SN5.fits",
        # dx_zygo=15, dy_zygo=251,
    ),
    dict(
        name="SW_SN7",
        path_nanomefos= path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN7/",
        file_nanomefos="25264_Subsrat_SW_SN7 3_ZU_woTilt.datx",
        dx_nanomefos=10, dy_nanomefos=189,
        file_zygo= path_individuel + '/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN7/Zygo/Form/07012026/substrate_FM_form_SW_SN7_vertex_tol20.fits',
        # dx_zygo=15, dy_zygo=251,
    ),
    dict(
        name="SW_SN8",
        path_nanomefos= path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
        file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo= path_individuel + "/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN8/Zygo/Form/19012026/report_binning_1.fits",
        # dx_zygo=15, dy_zygo=251,
    ),
    dict(
        name="SW_SN1",
        path_nanomefos= path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN1/new_fev_2026/",
        file_nanomefos="26-02-16_SW-SN1_mesZygo.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo= path_individuel + '/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_vertex.fits',
        # dx_zygo=15, dy_zygo=251,
    ),
    dict(
        name="SW_SN6",
        path_nanomefos= path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN1/new_fev_2026/",
        file_nanomefos="26-02-16_SW-SN1_mesZygo.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo= path_individuel + '/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN6/Zygo/Form/20260623/substrates_FM_form_SW_SN6_20260625_Nina_small_aperture.fits',
        # dx_zygo=15, dy_zygo=251,
    ),
    # dict(
    #     name="SW_SN1_reprise",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
    #     file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
    #     dx_nanomefos=2, dy_nanomefos=189,
    #     file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Reprise_Mai/Zygo/Form/20260513/substrates_template_FM_form_casquette_sathyne_finale_2.fits',
    #     # dx_zygo=15, dy_zygo=251,
    # ),
    # dict(
    #     name="SW_SN1_tilt",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
    #     file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
    #     dx_nanomefos=2, dy_nanomefos=189,
    #     file_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Reprise_Mai/Zygo/Form/20260513/substrates_template_FM_form_casquette_sathyne_finale_2.fits',
    #     # dx_zygo=15, dy_zygo=251,
    # ),
]

def load_map_nano(file, dx, dy, scale=1.0):
    data = SagData(file, dx=dx, dy=dy, theta=0, binning=1, auto_crop=True)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(
        surf,
        sw_substrate.aperture,
        sw_substrate.useful_area,
    )
    subs.x_grid_step = 0.1
    subs.y_grid_step = 0.1
    return np.asarray(subs.sag().data) * scale
def load_map_zygo_sw(file, scale=1.0):
    data = SagData(file, theta=0, binning=1, auto_crop=True)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(
        surf,
        sw_substrate.aperture,
        sw_substrate.useful_area,
    )
    subs.x_grid_step = 0.1
    subs.y_grid_step = 0.1
    return np.asarray(subs.sag().data) * scale

def load_map_zygo_lw(file, scale=1.0):
    data = SagData(file, theta=0, binning=1, auto_crop=True)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(
        surf,
        lw_substrate.aperture,
        lw_substrate.useful_area,
    )
    subs.x_grid_step = 0.1
    subs.y_grid_step = 0.1
    return np.asarray(subs.sag().data) * scale

fig, axes = plt.subplots(1, len(cases), figsize=(15, 8))
axes = axes.flatten()

vmin, vmax = -25, 25

for ax, case in zip(axes, cases):
    map_nano = load_map_nano(
        os.path.join(case["path_nanomefos"], case["file_nanomefos"]),
        case["dx_nanomefos"], case["dy_nanomefos"],
        scale=1e6
    )
    if case["name"][0] == "L":
        map_zygo = load_map_zygo_lw(
            case["file_zygo"],
            scale=1.0
        )
    elif case["name"] == "SW_SN6":
        map_zygo = load_map_zygo_sw(
            case["file_zygo"],
            scale=1.0
        )
        map_fill = map_zygo.copy()
        mask_nan = ~np.isfinite(map_fill)
        map_fill[mask_nan] = np.nanmean(map_fill)
        # biais_fit, _ = sfit(biais_fill, degree=1)
        map_zygo -= sfit(map_fill, degree=1)[0]

    else:
        map_zygo = load_map_zygo_sw(
            case["file_zygo"],
            scale=1.0
        )
        print('b')

#afficher zygo ou carte de biais

    # diff = map_zygo - map_nano
    biais = map_zygo

    # biais_fill = biais.copy()
    # mask_nan = ~np.isfinite(biais_fill)
    # biais_fill[mask_nan] = np.nanmean(biais_fill)

    # biais_fit, _ = sfit(biais_fill, degree=1)

    diff =  biais
    # diff[mask_nan] = np.nan
    dmin, dmax = np.nanpercentile(diff, [0.1, 99.9])
    pv = dmax - dmin
    rms = np.sqrt(np.nanmean(diff ** 2))
    # print(f"{case['name']:>7s} | min = {dmin:7.2f} µm | max = {dmax:7.2f} µm")
    print(
        f"{case['name']:>7s} | "
        f"RMS = {rms:6.2f} µm | "
        f"PV = {pv:6.2f} µm "
        f"(min={dmin:6.2f}, max={dmax:6.2f})"
    )

    im = ax.imshow(diff, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(case["name"])
    # ax.set_xticks([])
    # ax.set_yticks([])

# 👉 axe dédié pour la colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(" Zygo [nm]")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # laisse de la place à droite
# plt.savefig(path_zygo + "nanomefos_ALL.png", dpi=300)
plt.show()
