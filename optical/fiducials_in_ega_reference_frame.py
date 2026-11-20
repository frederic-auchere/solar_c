import os
import glob
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from optical.zygo import SagData, EGAFit
from optics.surfaces import Substrate, CircularAperture, Standard, Sphere
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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
# excel_file='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN9/Zygo/Form/20260622/substrates_template_FM_form_casquette_29062026.xlsm' #sw9 kapton
# excel_file = r"Y:/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN9/Zygo/Form/20260622/substrates_template_FM_form_casquette_29062026.xlsm"
excel_file = r"C:\Users\fauchere\Desktop\20260622\substrates_template_FM_form_casquette_29062026.xlsm"
    #'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW_livraison_finale/FM_SW_SN9/Zygo/Form/20261109_new_fiducials/substrates_template_FM_form_casquette.xlsm'
    #'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW_livraison_finale/FM_SW_SN3/Zygo/Tilt/20261023/substrates_template_FM_form_casquette.xlsm' #sw3 tilt
    # '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN10/Zygo/Form/20260630/substrates_template_FM_form_casquette_user_1.xlsm'
    # '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN9/Zygo/Form/20260622/substrates_template_FM_form_casquette_29062026.xlsm'
    # '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN3/Zygo/Form/20260814_correction/substrates_template_FM_form_casquette_SWSN3_without_crown.xlsm'
# excel_file='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN3/Zygo/Form/20260814_correction/substrates_template_FM_form_casquette_SWSN3.xlsm'
fitter=EGAFit.from_xlsx(excel_file)
print(fitter.sag_data)
print(fitter.substrate.surface)

fiducials_in_data=[]
fiducials_in_substrate=[]
wb = load_workbook(excel_file, read_only=True, data_only=True)
rows = wb.active.iter_rows(max_col=20)
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
            coordinates = [
                (row.pop('x1'), row.pop('y1')),
                (row.pop('x2'), row.pop('y2')),
                (row.pop('x3'), row.pop('y3')),
            ]
            fiducials_in_data.append(coordinates)
            fiducials_in_substrate.append([fitter.sag_data[i].to_substrate(*c) for c in coordinates])
wb.close()
fid = np.array(fiducials_in_substrate, dtype=float)   # forme (n_substrats, 3, 2)
fid_data = np.array(fiducials_in_data, dtype=float)   # forme (n_substrats, 3, 2)
from scipy.optimize import linear_sum_assignment

def reorder_fiducials(fid, fid_data):
    """Réordonne les 3 points de chaque substrat pour qu'ils correspondent
    à ceux des substrats précédents (moyenne courante comme référence)."""
    out = fid.copy()
    out_data = fid_data.copy()
    perms = [np.arange(3)]
    for i in range(1, len(fid)):
        ref = out[:i].mean(axis=0)                       # (3, 2)
        cost = np.linalg.norm(fid[i][:, None, :] - ref[None, :, :], axis=2)  # (3, 3)
        cand, slot = linear_sum_assignment(cost)         # candidat cand[j] -> slot slot[j]
        out[i, slot] = fid[i, cand]
        out_data[i, slot] = fid_data[i, cand]
        perms.append(slot)
    return out, out_data, perms

fid_sorted, fid_sorted_data, perms = reorder_fiducials(fid, fid_data)

for i, p in enumerate(perms):
    if not np.array_equal(p, [0, 1, 2]):
        print(f"Substrat {i}: points réordonnés (affectation {p})")

fid=fid_sorted
fid_data = fid_sorted_data
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

fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

half = 0.25  # half-width in mm, so each panel is a 0.5 x 0.5 mm window

for k, ax in enumerate(axes):
    means = np.array([np.nanmean(fid[crown == lvl, k, :], axis=0) for lvl in levels])
    ax.plot(means[:, 0], means[:, 1], color='black', lw=0.8, zorder=4)
    sc = ax.scatter(means[:, 0], means[:, 1], s=120, c=levels, cmap=cmap, norm=norm,
                    linewidths=1, zorder=5)

    # Centre of this fiducial's points, snapped so the edges land on ticks
    cx = 0.5 * (np.nanmax(means[:, 0]) + np.nanmin(means[:, 0]))
    cy = 0.5 * (np.nanmax(means[:, 1]) + np.nanmin(means[:, 1]))
    cx = np.round(cx / 0.05) * 0.05
    cy = np.round(cy / 0.05) * 0.05
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.set_title(f"Fiducial {k + 1}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

fig.colorbar(sc, ax=axes, label="Crown index", shrink=0.8)
fig.savefig('fiducials.png')
plt.show()

# Longest side of the fiducial triangle, per image and per crown (mm, after to_substrate)
sides = np.linalg.norm(fid - np.roll(fid, -1, axis=1), axis=2)          # (n, 3): f1-f2, f2-f3, f3-f1
sides_data = np.linalg.norm(fid_data - np.roll(fid_data, -1, axis=1), axis=2)          # (n, 3): f1-f2, f2-f3, f3-f1
longest = np.max(np.nan_to_num(sides, nan=-1.0), axis=1)
longest[~valid] = np.nan
longest_data = np.max(np.nan_to_num(sides_data, nan=-1.0), axis=1)
longest_data[~valid] = np.nan

which = np.argmax(np.nan_to_num(sides, nan=-1.0), axis=1)
names = ['f1-f2', 'f2-f3', 'f3-f1']
print("longest side is:", {names[k]: int(np.sum(which[valid] == k)) for k in range(3)}, "(images)")

lm = np.array([np.nanmean(longest[crown == lvl]) for lvl in levels])
le = np.array([np.nanstd(longest[crown == lvl], ddof=1) / np.sqrt(np.sum(crown == lvl)) for lvl in levels])

lm_data = np.array([np.nanmean(longest_data[crown == lvl]) for lvl in levels])
le_data = np.array([np.nanstd(longest_data[crown == lvl], ddof=1) / np.sqrt(np.sum(crown == lvl)) for lvl in levels])


fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
ax.plot(crown[valid], longest[valid], '.', color='grey', ms=4, alpha=0.5, label='single images')
ax.errorbar(levels, lm, yerr=le, color='black', lw=0.8, capsize=2, zorder=2, label='crown mean')
ax.scatter(levels, lm, s=60, c=levels, cmap=cmap, norm=norm,
           edgecolors='white', linewidths=1, zorder=3)
ax.set_xticks(levels)
ax.set_xlabel("Crown index")
ax.set_ylabel("longest side (mm)")

ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3)
plt.show()

fig.savefig('longest_mm.png')

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
ax.plot(crown[valid], longest_data[valid], '.', color='grey', ms=4, alpha=0.5, label='single images')
ax.errorbar(levels, lm_data, yerr=le_data, color='black', lw=0.8, capsize=2, zorder=2, label='crown mean')
ax.scatter(levels, lm_data, s=60, c=levels, cmap=cmap, norm=norm,
           edgecolors='white', linewidths=1, zorder=3)
ax.set_xticks(levels)
ax.set_xlabel("Crown index")
ax.set_ylabel("longest side (pix)")

ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3)
plt.show()

fig.savefig('longest_data.png')

means = []
for k in (0, 1, 2):
    means.append([np.nanmean(fid_data[crown == lvl, k, :], axis=0) for lvl in levels])
means = np.asarray(means)
means = np.concatenate((means, np.expand_dims(means[0, :, :], axis=0)), axis=0)

fig, ax = plt.subplots(1, 1)
for lvl in levels[0:1]:
    ax.plot(means[:, int(lvl)-1, 0], means[:, int(lvl)-1, 1])
ax.set_aspect("equal")
