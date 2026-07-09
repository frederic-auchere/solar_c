from astropy.io import fits
import matplotlib.pyplot as plt
from optical import sw_substrate,lw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np
from fitting import sfit
path_zygo='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/'

path_individuel = "Y:/02- Engineering"
cases = [
    dict(
        name="SW_SN2",
        substrate=sw_substrate,
        path_nanomefos=path_individuel + r"/08 - Metrology/01 - Optics/08 - Bertin/SW_SN2/",
        file_nanomefos="25264_Subsrat_SW_SN2 3_ZU_woTilt.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo=path_individuel + r"\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN2\Zygo\Form\20260625\substrates_template_FM_form_casquette_SN2.fits",
    ),
    dict(
        name="SW_SN3",
        substrate=sw_substrate,
        path_nanomefos=path_individuel +r"/08 - Metrology/01 - Optics/08 - Bertin/SW_SN3/",
        file_nanomefos="25264_Subsrat_SW_SN3 3_ZU_woTilt.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo=path_individuel + r"\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN3\Zygo\Form\20260629\substrates_template_FM_form_casquette.fits",
    ),
    dict(
        name="SW_SN5",
        substrate=sw_substrate,
        path_nanomefos=path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/",
        file_nanomefos="25264_Subsrat_SW_SN5 2_ZU_woTilt.datx",
        dx_nanomefos=1, dy_nanomefos=189,
        file_zygo=path_individuel + "/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/substrate_FM_form_SW_SN5.fits",
    ),
    dict(
        name="SW_SN6",
        substrate=sw_substrate,
        path_nanomefos=path_individuel + r"/08 - Metrology/01 - Optics/08 - Bertin/SW_SN6/",
        file_nanomefos="25264_Subsrat_SW_SN6 3_ZU_woTilt.datx",
        dx_nanomefos=1, dy_nanomefos=189,
        file_zygo=path_individuel + r"\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN6\Zygo\Form\20260623\substrates_FM_form_SW_SN6_20260626_Nina.fits",
    ),
    dict(
        name="SW_SN7",
        substrate=sw_substrate,
        path_nanomefos=path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN7/",
        file_nanomefos="25264_Subsrat_SW_SN7 3_ZU_woTilt.datx",
        dx_nanomefos=10, dy_nanomefos=189,
        file_zygo=path_individuel + r"\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN7\Zygo\Form\07012026\substrate_FM_form_SW_SN7.fits",
    ),
    dict(
        name="SW_SN8",
        substrate=sw_substrate,
        path_nanomefos=path_individuel + "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
        file_nanomefos="25264_Subsrat_SW_SN8 3_ZU_woTilt.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo=path_individuel + r"\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN8\Zygo\Form\20012026\substrate_FM_form_SW_SN8.fits",
    ),
    dict(
        name="SW_SN9",
        substrate=sw_substrate,
        path_nanomefos=path_individuel + r"/08 - Metrology/01 - Optics/08 - Bertin/SW_SN9/",
        file_nanomefos="25264_Subsrat_SW_SN9 5_ZU_woTilt.datx",
        dx_nanomefos=0, dy_nanomefos=189,
        file_zygo=path_individuel + r"\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN9\Zygo\Form\20260622\substrates_template_FM_form_casquette_29062029_Nina.fits",
    ),
    dict(
        name="SW_SN10",
        substrate=sw_substrate,
        path_nanomefos=path_individuel+ "/08 - Metrology/01 - Optics/08 - Bertin/SW_SN10/",
        file_nanomefos="25264_Subsrat_SW_SN10 3_ZU_woTilt.datx",
        dx_nanomefos=1, dy_nanomefos=189,
        file_zygo=r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN10\Zygo\Form\20260630\substrates_template_FM_form_casquette.fits",
    ),
    # dict(
    #     name="LW_SN1",
    #     substrate=lw_substrate,
    #     path_nanomefos= path_individuel +r"/08 - Metrology/01 - Optics/07 - Measurements/FM/LW/FM_LW_SN1/Zygo/Form/29012026/",
    #     file_nanomefos= "substrate_FM_form_LW_SN1_vertex.fits",
    #     dx_nanomefos=-30, dy_nanomefos=189,
    #     file_zygo=r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements\FM\LW\FM_LW_SN1\Zygo\Form\29012026\substrate_FM_form_LW_SN1_vertex.fits",
    # ),
    # dict(
    #     name="LW_SN2",
    #     substrate=lw_substrate,
    #     path_nanomefos=path_individuel+ "/08 - Metrology/01 - Optics/08 - Bertin/LW_SN2/",
    #     file_nanomefos=r"LW-SN2_2_ZU_woTilt.datx",
    #     dx_nanomefos=-30, dy_nanomefos=189,
    #     file_zygo=r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements\FM\LW\FM_LW_SN2\Zygo\Form\06042026\substrates_template_FM_form_casquette.fits",
    # ),
    # dict(
    #     name="LW_SN3",
    #     substrate=lw_substrate,
    #     path_nanomefos=path_individuel+ "/08 - Metrology/01 - Optics/08 - Bertin/LW_SN3/",
    #     file_nanomefos="25264-301-PL002_SubstratLW-SN3 ASPH 3_ZU_woTilt.datx",
    #     dx_nanomefos=-30, dy_nanomefos=189,
    #     file_zygo=r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements\FM\LW\FM_LW_SN3\Zygo\Form\20260612\substrates_template_FM_form_casquette_sathyne.fits",
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

fig, axes = plt.subplots(3, len(cases), figsize=(15, 8))
axes = axes.reshape(3, len(cases))

vmin, vmax = -80, 80

for i, case in enumerate(cases):
    ax = axes[0, i]
    ax_nano = axes[1, i]
    ax_diff = axes[2, i]

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
    elif case["name"][0] == "S":
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
    ax.set_title(f"{case['name']}\nRMS = {rms:.2f}")
    # ax.set_xticks([])
    # ax.set_yticks([])

    # --- ligne du milieu : carte nanomefos (ZU) ---
    dmin_n, dmax_n = np.nanpercentile(map_nano, [0.1, 99.9])
    pv_n = dmax_n - dmin_n
    rms_n = np.sqrt(np.nanmean(map_nano ** 2))
    print(
        f"{case['name']:>7s} (nano) | "
        f"RMS = {rms_n:6.2f} µm | "
        f"PV = {pv_n:6.2f} µm "
        f"(min={dmin_n:6.2f}, max={dmax_n:6.2f})"
    )

    im_nano = ax_nano.imshow(map_nano, origin="lower", vmin=vmin, vmax=vmax)
    ax_nano.set_title(f"Nano {case['name']}\nRMS = {rms_n:.2f}")

    # --- ligne du bas : carte de différence zygo - nanomefos (ZU) ---
    diff_bias = map_zygo - map_nano
    dmin_b, dmax_b = np.nanpercentile(diff_bias, [0.1, 99.9])
    pv_b = dmax_b - dmin_b
    rms_b = np.sqrt(np.nanmean(diff_bias ** 2))
    print(
        f"{case['name']:>7s} (diff) | "
        f"RMS = {rms_b:6.2f} | "
        f"PV = {pv_b:6.2f} "
        f"(min={dmin_b:6.2f}, max={dmax_b:6.2f})"
    )

    im_diff = ax_diff.imshow(diff_bias, origin="lower", vmin=vmin, vmax=vmax)
    ax_diff.set_title(f"Diff {case['name']}\nRMS = {rms_b:.2f}")

# 👉 axe dédié pour la colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(" Zygo [nm]")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # laisse de la place à droite
# plt.savefig(path_zygo + "nanomefos_ALL.png", dpi=300)
plt.show()