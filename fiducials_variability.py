import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import linear_sum_assignment
import os

THRESHOLD = 10  # pixels

def reassign_points(sub):
    """
    For each row, reassign (xc_1,yc_1), (xc_2,yc_2), (xc_3,yc_3) so that
    each point label consistently tracks the same physical point across rows.
    Uses the first row as reference, then for each subsequent row finds the
    optimal assignment minimizing total distance to the current running mean.
    """
    cols_x = ['xc_1', 'xc_2', 'xc_3']
    cols_y = ['yc_1', 'yc_2', 'yc_3']
    data = sub[cols_x + cols_y].values.copy()  # shape (n, 6)
    n = len(data)

    assigned = np.zeros_like(data)
    assigned[0] = data[0]

    for i in range(1, n):
        ref_x = assigned[:i, :3].mean(axis=0)   # running mean x of each point
        ref_y = assigned[:i, 3:].mean(axis=0)   # running mean y of each point

        cand_x = data[i, :3]
        cand_y = data[i, 3:]

        # Cost matrix: distance from candidate j to reference slot k
        cost = np.zeros((3, 3))
        for j in range(3):
            for k in range(3):
                cost[j, k] = np.sqrt((cand_x[j] - ref_x[k])**2 + (cand_y[j] - ref_y[k])**2)

        row_ind, col_ind = linear_sum_assignment(cost)
        # col_ind[j] = which reference slot candidate j maps to
        # We want: slot k <- candidate perm[k]
        perm = np.argsort(col_ind)  # perm[k] = which candidate goes to slot k
        assigned[i, :3] = cand_x[perm]
        assigned[i, 3:] = cand_y[perm]

    result = sub.copy()
    for k, cx, cy in zip(range(3), cols_x, cols_y):
        result[cx] = assigned[:, k]
        result[cy] = assigned[:, 3+k]
    return result


def is_mixed(sub, threshold=THRESHOLD):
    """Returns True if any point column has range > threshold."""
    for cx, cy in [('xc_1','yc_1'),('xc_2','yc_2'),('xc_3','yc_3')]:
        rx = sub[cx].max() - sub[cx].min()
        ry = sub[cy].max() - sub[cy].min()
        if rx > threshold or ry > threshold:
            return True
    return False


excel_path = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/npy_exports/fiducials_measurements.xlsx'
df = pd.read_excel(excel_path)
orientations = sorted(df['mirror'].unique())
n_orientations = len(orientations)

points = [
    ('Point 1', 'xc_1', 'yc_1'),
    ('Point 2', 'xc_2', 'yc_2'),
    ('Point 3', 'xc_3', 'yc_3'),
]

fig, axes = plt.subplots(n_orientations, 3, figsize=(16, 4.5 * n_orientations))
fig.suptitle('Fiducial Point Positions by Orientation', fontsize=16, fontweight='bold', y=1.01)

if n_orientations == 1:
    axes = axes[np.newaxis, :]

cmap = plt.cm.viridis

for row, mirror in enumerate(orientations):
    sub_raw = df[df['mirror'] == mirror].reset_index(drop=True)
    mixed = is_mixed(sub_raw)
    if mixed:
        sub = reassign_points(sub_raw)
        print(f"Mirror {int(mirror)}: points reassigned (range > {THRESHOLD}px detected)")
    else:
        sub = sub_raw
    n = len(sub)

    for col, (title, xcol, ycol) in enumerate(points):
        ax = axes[row, col]
        x, y = sub[xcol].values, sub[ycol].values
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(), y.std()

        sc = ax.scatter(x, y, c=range(n), cmap=cmap, s=80,
                        edgecolors='black', linewidths=0.6, zorder=4)
        ax.scatter(mx, my, color='red', s=150, marker='+',
                   linewidths=2.5, zorder=5, label='Mean')

        ellipse = mpatches.Ellipse((mx, my), width=2*sx, height=2*sy,
                                   fill=False, edgecolor='red', linestyle='--',
                                   linewidth=1.5, zorder=3,
                                   label=f'1σ ellipse\n(σx={sx:.3f}, σy={sy:.3f})')
        ax.add_patch(ellipse)

        margin = max(sx, sy, 0.5) * 4
        ax.set_xlim(mx - margin, mx + margin)
        ax.set_ylim(my - margin, my + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X (pixels)', fontsize=9)
        ax.tick_params(labelsize=8)

        if row == 0:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        if col == 0:
            ax.set_ylabel(f'Mirror {int(mirror)}\nY (pixels)', fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('Y (pixels)', fontsize=9)

        ax.annotate(f'σx={sx:.3f}\nσy={sy:.3f}\nn={n}',
                    xy=(0.03, 0.97), xycoords='axes fraction', fontsize=8,
                    va='top', color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label('Measurement #', fontsize=7)
        cb.ax.tick_params(labelsize=7)

        # ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(excel_path))
base = os.path.splitext(os.path.basename(excel_path))[0]
# plt.savefig(os.path.join(out_dir, base + '.pdf'), dpi=150,
#             bbox_inches='tight', facecolor='white', format='pdf')
# plt.savefig(os.path.join(out_dir, base + '.png'), dpi=150,
#             bbox_inches='tight', facecolor='white')
print(f"Figures saved to: {out_dir}")
plt.show()
print("Done.")