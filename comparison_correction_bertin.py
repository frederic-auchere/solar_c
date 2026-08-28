from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from optical import sw_substrate, lw_substrate,rectangular_sw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np
from fitting import sfit

cases = [
    dict(
        name="SW_SN2",
        substrate_type="SW",
        file_zygo_a='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN2/Zygo/Form/20260625/substrates_template_FM_form_casquette_SN2.fits',
        label_a="2026-06-25",
        file_zygo_b='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN2/Zygo/Form/20260813 Correction/substrates_template_FM_form_casquette_SN2_16orientations.fits',
        label_b="2026-08-13",
        file_zygo_c='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW_livraison_finale/FM_SW_SN2/Zygo/Form/20260828/substrates_template_FM_form_casquette_SN2.fits',
        label_c="2026-08-28",
        # file_zygo_d='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20211202/substrate_FM_form_SW_SN5_full.fits',
        # label_d="sw5"
    ),
    # dict(
    #     name="LW_SN1",
    #     substrate_type="LW",
    #     file_zygo_a='/Volumes/solarc/.../FM_LW_SN1/Zygo/Form/29012026/report_substrate_FM_form_LW_SN1_binning1.fits',
    #     label_a="2026-01-29",
    #     file_zygo_b='/Volumes/solarc/.../FM_LW_SN1/Zygo/Form/XXXXXXXX/report_v2.fits',
    #     label_b="XXXX-XX-XX",
    # ),
    # ajouter d'autres substrats ici
]


def load_map_zygo(file, substrate_type, scale=1.0):
    data = SagData(file, theta=0, binning=1, auto_crop=True)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    aperture = sw_substrate.aperture if substrate_type == "SW" else lw_substrate.aperture
    useful_area = sw_substrate.useful_area if substrate_type == "SW" else lw_substrate.useful_area
    subs = surfaces.Substrate(surf, aperture, useful_area)
    subs.x_grid_step = 0.1
    subs.y_grid_step = 0.1
    return np.asarray(subs.sag().data) * scale


vmin, vmax = -20, 20

for case in cases:
    map_a = load_map_zygo(case["file_zygo_a"], case["substrate_type"])
    map_b = load_map_zygo(case["file_zygo_b"], case["substrate_type"])
    map_c= load_map_zygo(case["file_zygo_c"], case["substrate_type"])
    # map_d = load_map_zygo(case["file_zygo_d"], case["substrate_type"])
    diff = map_b - map_c
    # map_c=map_c*1e3
    # map_c_corr = map_c.copy()
    # map_c_fill = map_c.copy()
    # mask_nan = ~np.isfinite(map_c_fill)
    # map_c_fill[mask_nan] = np.nanmean(map_c_fill)  # ou 0.0
    #
    # poly, _ = sfit(map_c_fill, degree=1)
    # map_c_corr = map_c_corr - poly
    # map_c_corr[mask_nan] = np.nan  # remet le masque
    # Stats
    for label, m in [(case["label_a"], map_a), (case["label_b"], map_b),(case["label_c"], map_c) ]:
        dmin, dmax = np.nanpercentile(m, [0.1, 99.9])
        rms = np.sqrt(np.nanmean(m ** 2))
        print(f"{case['name']:>7s} | {label:>12s} | RMS = {rms:6.2f} µm | PV = {dmax - dmin:6.2f} µm")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(case["name"], fontsize=13)

    panels = [
        (map_a, f"Zygo 1\n{case['label_a']}", None, None),
        (map_b, f"Zygo correction 1\n{case['label_b']}", None, None),
        (map_c,f"Zygo correction 2\n{case['label_c']}", None,None),# échelle libre pour la diff
        (diff, f"Zygo diff \nCorrection 1 moins correction 2", None, None),

    ]

    for ax, (data, title, lo, hi) in zip(axes, panels):
        if lo is None:
            lo, hi = np.nanpercentile(data, [0.5, 99.5])
        im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="nm", shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"zygo_comparison_{case['name']}.png", dpi=150)
    plt.show()