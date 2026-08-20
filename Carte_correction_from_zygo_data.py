import numpy as np
import matplotlib.pyplot as plt
import os
from optical import sw_substrate, lw_substrate, rectangular_sw_substrate, rectangular_lw_substrate
from optics.zygo import SagData
from optics import surfaces
import matplotlib as mpl
from optical.utils import write_zygo_dat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optical.zygo import SagData, EGAFit


mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

# =====================================================================
# LOAD zygo map full with fit on ZU
# =====================================================================


excel_file='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN5/Zygo/Form/20260706 apres correction/substrates_FM_SW_SN5_form.xlsm'

fitter=EGAFit.from_xlsx(excel_file)
outfile=excel_file[:-5]+'_correction_map.pdf'
print(fitter.sag_data)
print(fitter.substrate.surface)

print('XXXXXXXX')
fit = fitter.fit()


fitter.make_report(outfile=outfile, save_stack=True,mask_output=False)



#need to laod nano maps to interpolate at same resolution
cases = [

    # dict(
    #     name="LW_SN1",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/LW_SN1/",
    #     file_nanomefos="new_jan_2026/25264-301-PL002_SubstratLW-SN1 ASPH 1_Full_woTilt.datx",
    #     dx_nanomefos=133, dy_nanomefos=138,
    #     file_zygo=outfile[:-3]+'fits'
    # ),
    # dict(
    #     name="SW_SN5",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN5/",
    #     file_nanomefos='25264_Subsrat_SW_SN5 1_full_woTilt.datx',
    #     dx_nanomefos=60, dy_nanomefos=140,
    #     file_zygo=outfile[:-3]+'fits'
    # ),
    dict(
        name="SW_SN2",
        path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN2/",
        file_nanomefos='25264_Subsrat_SW_SN2 1_Full_woTilt.datx',
        dx_nanomefos=60, dy_nanomefos=140,
        file_zygo=outfile[:-3]+'fits',
    ),
    # dict(
    #     name="SW_SN3",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN3/",
    #     file_nanomefos='25264_Subsrat_SW_SN3 1_Full_woTilt.datx',
    #     dx_nanomefos=60, dy_nanomefos=140,
    #     file_zygo=outfile[:-3]+'fits',
    # ),
    # dict(
    #     name="SW_SN6",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN6/",
    #     file_nanomefos='25264_Subsrat_SW_SN6 1_Full_woTilt.datx',
    #     dx_nanomefos=60, dy_nanomefos=140,
    #     file_zygo=outfile[:-3]+'fits',
    # ),
    # dict(
    #     name="SW_SN7",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN7/",
    #     file_nanomefos='25264_Subsrat_SW_SN7 1_Full_woTilt.datx',
    #     dx_nanomefos=60, dy_nanomefos=140,
    #     file_zygo=outfile[:-3]+'fits',
    # ),
    # dict(
    #     name="SW_SN8",
    #     path_nanomefos="/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/08 - Bertin/SW_SN8/",
    #     file_nanomefos='25264_Subsrat_SW_SN8 1_Full_woTilt.datx',
    #     dx_nanomefos=60, dy_nanomefos=140,
    #     file_zygo=outfile[:-3]+'fits',
    # ),

]







# =====================================================================
# FONCTIONS
# =====================================================================
def load_map_nano_sw(file, dx, dy, scale=1.0):
    data = SagData(file, dx=dx, dy=dy, theta=0, binning=1, auto_crop=False)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(surf, rectangular_sw_substrate.aperture, rectangular_sw_substrate.useful_area)
    return subs

def load_map_nano_lw(file, dx, dy, scale=1.0):
    data = SagData(file, dx=dx, dy=dy, theta=0, binning=1, auto_crop=False)
    surf = surfaces.MeasuredSurface(data, alpha=0, beta=0, gamma=0)
    subs = surfaces.Substrate(surf, rectangular_lw_substrate.aperture, rectangular_lw_substrate.useful_area)
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

for case in cases:
    name = case["name"]
    print(f"Chargement {name}...")


    if name.startswith("LW"):
        subs_zygo = load_map_zygo_lw(case["file_zygo"])
        subs_nano = load_map_nano_lw(
            os.path.join(case["path_nanomefos"], case["file_nanomefos"]),
            case["dx_nanomefos"], case["dy_nanomefos"]
        )
    else:
        subs_zygo = load_map_zygo_sw(case["file_zygo"])
        subs_nano = load_map_nano_sw(
            os.path.join(case["path_nanomefos"], case["file_nanomefos"]),
            case["dx_nanomefos"], case["dy_nanomefos"]
        )

    grid_nano=subs_nano.grid()
    map_nano=subs_nano.sag().data *1e6

    map_zygo=subs_zygo.sag(grid_nano).data
    correction_map = map_zygo

    # Figure
    fig, ax = plt.subplots(1,1, figsize=(10, 8))
    vabs = np.nanpercentile(np.abs(correction_map), 99)
    im=ax.imshow(map_zygo, origin='lower', extent=rectangular_lw_substrate.limits, vmin=-vabs, vmax=vabs)


    plt.colorbar(im, label="nm")
    plt.tight_layout()
    plt.show()

    write_zygo_dat(f'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/carte_correction_sw/carte_correction_{name}.dat',
        correction_map, subs_nano.x_grid_step)