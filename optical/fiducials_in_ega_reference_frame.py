import os
import glob
from optical.zygo import SagData, EGAFit
from optics.surfaces import Substrate, CircularAperture, Standard, Sphere
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
def read_table():
    fields = next(rows)
    table_keys = []
    column = 0
    while fields[column].value is not None:
        table_keys.append(fields[column].value.split(' ')[0].lower())
        column += 1
    table = []
    while True:
        try:
            row = next(rows)
        except StopIteration:
            break
        if row[0].value is None:
            break
        table.append({key: cell.value for key, cell in zip(table_keys, row)})

    return table
excel_file='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN9/Zygo/Form/20260622/substrates_template_FM_form_casquette_29062026.xlsm' #sw9 kapton
    #'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW_livraison_finale/FM_SW_SN9/Zygo/Form/20261109_new_fiducials/substrates_template_FM_form_casquette.xlsm'
    #'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW_livraison_finale/FM_SW_SN3/Zygo/Tilt/20261023/substrates_template_FM_form_casquette.xlsm' #sw3 tilt
    # '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN10/Zygo/Form/20260630/substrates_template_FM_form_casquette_user_1.xlsm'
    # '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN9/Zygo/Form/20260622/substrates_template_FM_form_casquette_29062026.xlsm'
    # '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN3/Zygo/Form/20260814_correction/substrates_template_FM_form_casquette_SWSN3_without_crown.xlsm'
# '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN3/Zygo/Form/20260814_correction/substrates_template_FM_form_casquette_SWSN3.xlsm'
fitter=EGAFit.from_xlsx(excel_file)
print(fitter.sag_data)
print(fitter.substrate.surface)

fiducials_in_substrate=[]
wb = load_workbook(excel_file, keep_vba=True)
rows = wb.active.rows
while True:
    try:
        row = next(rows)
    except StopIteration:
        break
    if row[0].value is None:
        continue
    value = row[0].value
    if not isinstance(value, str):
        continue          # None, float, date... : on ignore
    elif "4.2 Data" in row[0].value:
        table = read_table()
        fiducials = []
        crown_indices = []
        for i,row in enumerate(table):
            crown_index = row.pop('crown')
            crown_indices.append(crown_index if crown_index > 0 else None)
            fiducials_in_substrate.append([
                fitter.sag_data[i].to_substrate(row.pop('x1'), row.pop('y1')),
                fitter.sag_data[i].to_substrate(row.pop('x2'), row.pop('y2')),
                fitter.sag_data[i].to_substrate(row.pop('x3'), row.pop('y3')),
            ])

fid = np.array(fiducials_in_substrate, dtype=float)   # forme (n_substrats, 3, 2)
from scipy.optimize import linear_sum_assignment

def reorder_fiducials(fid):
    """Réordonne les 3 points de chaque substrat pour qu'ils correspondent
    à ceux des substrats précédents (moyenne courante comme référence)."""
    out = fid.copy()
    perms = [np.arange(3)]
    for i in range(1, len(fid)):
        ref = out[:i].mean(axis=0)                       # (3, 2)
        cost = np.linalg.norm(fid[i][:, None, :] - ref[None, :, :], axis=2)  # (3, 3)
        cand, slot = linear_sum_assignment(cost)         # candidat cand[j] -> slot slot[j]
        out[i, slot] = fid[i, cand]
        perms.append(slot)
    return out, perms

fid_sorted, perms = reorder_fiducials(fid)

for i, p in enumerate(perms):
    if not np.array_equal(p, [0, 1, 2]):
        print(f"Substrat {i}: points réordonnés (affectation {p})")

fid=fid_sorted
crown = np.array([np.nan if c is None else c for c in crown_indices], dtype=float)
valid = np.isfinite(crown)

norm = plt.Normalize(np.nanmin(crown), np.nanmax(crown))
cmap = plt.get_cmap('rainbow')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, ax in enumerate(axes):
    ax.scatter(fid[~valid, k, 0], fid[~valid, k, 1], s=20, color='lightgrey',
               label='aucun' if k == 0 else None)
    sc = ax.scatter(fid[valid, k, 0], fid[valid, k, 1], s=20,
                    c=crown[valid], cmap=cmap, norm=norm)
    ax.set_title(f"Fiducial {k + 1}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

fig.colorbar(sc, ax=axes, label="Crown index", shrink=0.8)
if (~valid).any():
    axes[0].legend(loc="best")
plt.show()



crown = np.array([np.nan if c is None else c for c in crown_indices], dtype=float)
valid = np.isfinite(crown)
levels = np.unique(crown[valid])
#
norm = plt.Normalize(levels.min(), levels.max())
cmap = plt.get_cmap('rainbow')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, ax in enumerate(axes):
    means = np.array([np.nanmean(fid[crown == lvl, k, :], axis=0) for lvl in levels])
    ax.plot(means[:, 0], means[:, 1], color='black', lw=0.8, zorder=4)
    sc = ax.scatter(means[:, 0], means[:, 1], s=120, c=levels, cmap=cmap, norm=norm,
                    linewidths=1, zorder=5)

    ax.set_title(f"Fiducial {k + 1}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

fig.colorbar(sc, ax=axes, label="Crown index", shrink=0.8)
plt.show()



