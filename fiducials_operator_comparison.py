import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
EXCEL_PATH_1 = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN7/Zygo/Form/07012026/npy_exports/fiducials_measurements_nina.xlsx'  # ← path to first operator's file
EXCEL_PATH_2 = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN7/Zygo/Form/07012026/npy_exports/fiducials_measurements_axel.xlsx'   # ← path to second operator's file
EXCEL_PATH_3= '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN7/Zygo/Form/07012026/npy_exports/fiducials_measurements_sathyne.xlsx'
OPERATOR_LABELS = ['Nina', 'Axel','Sathyne']
COLORS = ['steelblue', 'darkorange','red']
THRESHOLD = 10  # pixels — triggers point reassignment if exceeded

# ──────────────────────────────────────────────────────────────────────────────

POINTS = [
    ('Point 1', 'xc_1', 'yc_1'),
    ('Point 2', 'xc_2', 'yc_2'),
    ('Point 3', 'xc_3', 'yc_3'),
]


def reassign_points(sub):
    cols_x = ['xc_1', 'xc_2', 'xc_3']
    cols_y = ['yc_1', 'yc_2', 'yc_3']
    data = sub[cols_x + cols_y].values.copy()
    n = len(data)
    assigned = np.zeros_like(data)
    assigned[0] = data[0]
    for i in range(1, n):
        ref_x = assigned[:i, :3].mean(axis=0)
        ref_y = assigned[:i, 3:].mean(axis=0)
        cand_x, cand_y = data[i, :3], data[i, 3:]
        cost = np.array([[np.sqrt((cand_x[j]-ref_x[k])**2 + (cand_y[j]-ref_y[k])**2)
                          for k in range(3)] for j in range(3)])
        _, col_ind = linear_sum_assignment(cost)
        perm = np.argsort(col_ind)
        assigned[i, :3] = cand_x[perm]
        assigned[i, 3:] = cand_y[perm]
    result = sub.copy()
    for k, cx, cy in zip(range(3), cols_x, cols_y):
        result[cx] = assigned[:, k]
        result[cy] = assigned[:, 3+k]
    return result


def is_mixed(sub, threshold=THRESHOLD):
    for cx, cy in [('xc_1','yc_1'), ('xc_2','yc_2'), ('xc_3','yc_3')]:
        if sub[cx].max()-sub[cx].min() > threshold or sub[cy].max()-sub[cy].min() > threshold:
            return True
    return False


def load_and_clean(path, label):
    df = pd.read_excel(path)
    cleaned = []
    for mirror, sub_raw in df.groupby('mirror'):
        sub_raw = sub_raw.reset_index(drop=True)
        if is_mixed(sub_raw):
            sub = reassign_points(sub_raw)
            print(f"{label} — Mirror {int(mirror)}: points reassigned (range > {THRESHOLD}px detected)")
        else:
            sub = sub_raw
        sub = sub.copy()
        sub['mirror'] = mirror
        cleaned.append(sub)
    return pd.concat(cleaned, ignore_index=True)


# Load both datasets
df1 = load_and_clean(EXCEL_PATH_1, OPERATOR_LABELS[0])
df2 = load_and_clean(EXCEL_PATH_2, OPERATOR_LABELS[1])
df3 = load_and_clean(EXCEL_PATH_3, OPERATOR_LABELS[2])

# Cross-operator alignment: for each orientation, reorder df2's points so they
# match df1's points by proximity of their means (Hungarian algorithm).
def align_operators(df_ref, df_other, label_other):
    result_rows = []
    cols_x = ['xc_1', 'xc_2', 'xc_3']
    cols_y = ['yc_1', 'yc_2', 'yc_3']
    for mirror in sorted(df_ref['mirror'].unique()):
        sub_ref   = df_ref[df_ref['mirror'] == mirror]
        sub_other = df_other[df_other['mirror'] == mirror].copy()
        if sub_other.empty:
            result_rows.append(sub_other)
            continue

        ref_means   = np.array([[sub_ref[cx].mean(), sub_ref[cy].mean()]
                                 for cx, cy in zip(cols_x, cols_y)])   # (3, 2)
        other_means = np.array([[sub_other[cx].mean(), sub_other[cy].mean()]
                                 for cx, cy in zip(cols_x, cols_y)])   # (3, 2)

        cost = np.linalg.norm(other_means[:, None, :] - ref_means[None, :, :], axis=2)  # (3,3)
        _, col_ind = linear_sum_assignment(cost)
        # col_ind[j] = which ref slot candidate j maps to → perm[k] = which candidate goes to slot k
        perm = np.argsort(col_ind)

        if not np.array_equal(perm, [0, 1, 2]):
            print(f"{label_other} — Mirror {int(mirror)}: points reordered to match "
                  f"{OPERATOR_LABELS[0]} (permutation {perm})")
            for k, (cx, cy) in enumerate(zip(cols_x, cols_y)):
                sub_other[cx] = df_other[df_other['mirror'] == mirror][cols_x[perm[k]]].values
                sub_other[cy] = df_other[df_other['mirror'] == mirror][cols_y[perm[k]]].values

        result_rows.append(sub_other)
    return pd.concat(result_rows, ignore_index=True)

df2 = align_operators(df1, df2, OPERATOR_LABELS[1])
df3 = align_operators(df1, df3, OPERATOR_LABELS[2])

# Common orientations
orientations = sorted(set(df1['mirror'].unique()) | set(df2['mirror'].unique()))
n_orientations = len(orientations)

fig, axes = plt.subplots(n_orientations, 3, figsize=(16, 4.5 * n_orientations))
fig.suptitle(f'Fiducial Point Comparison — {OPERATOR_LABELS[0]} vs {OPERATOR_LABELS[1]} and {OPERATOR_LABELS[2]}',
             fontsize=16, fontweight='bold', y=1.01)

if n_orientations == 1:
    axes = axes[np.newaxis, :]

for row, mirror in enumerate(orientations):
    for col, (title, xcol, ycol) in enumerate(POINTS):
        ax = axes[row, col]

        all_x, all_y = [], []
        offsets = [0.03, 0.36, 0.69]
        for (df, color, label), offset_x in zip(zip([df1, df2, df3], COLORS, OPERATOR_LABELS), offsets):
            sub = df[df['mirror'] == mirror]
            if sub.empty:
                continue

            x, y = sub[xcol].values, sub[ycol].values
            mx, my = x.mean(), y.mean()
            sx = float(x.std()) if len(x) > 1 else 0.0
            sy = float(y.std()) if len(y) > 1 else 0.0
            n = len(x)

            all_x.extend([mx - max(3*sx, 0.5), mx + max(3*sx, 0.5)])
            all_y.extend([my - max(3*sy, 0.5), my + max(3*sy, 0.5)])

            # Mean marker
            ax.scatter(mx, my, color=color, s=120, marker='o', zorder=5,
                       edgecolors='black', linewidths=0.8,
                       label=f'{label} (n={n})')

            # 1-sigma ellipse
            ellipse = mpatches.Ellipse((mx, my), width=2*sx, height=2*sy,
                                       fill=True, facecolor=color, alpha=0.2,
                                       edgecolor=color, linestyle='--',
                                       linewidth=1.5, zorder=3)
            ax.add_patch(ellipse)

            # Stats annotation offset to avoid overlap

            ax.annotate(f'{label}\nμx={mx:.2f}, μy={my:.2f}\nσx={sx:.3f}, σy={sy:.3f}',
                        xy=(offset_x, 0.97), xycoords='axes fraction',
                        fontsize=7, va='top', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85,
                                  edgecolor=color, linewidth=0.8))

        # Axis limits centered on data
        all_x = [v for v in all_x if np.isfinite(v)]
        all_y = [v for v in all_y if np.isfinite(v)]
        if all_x and all_y:
            cx, cy = np.mean(all_x), np.mean(all_y)
            span = max(np.ptp(all_x), np.ptp(all_y), 1.0) * 0.7
            if np.isfinite(span) and span > 0:
                ax.set_xlim(cx - span, cx + span)
                ax.set_ylim(cy - span, cy + span)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X (pixels)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc='lower right')

        if row == 0:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        if col == 0:
            ax.set_ylabel(f'Mirror {int(mirror)}\nY (pixels)', fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('Y (pixels)', fontsize=9)



# ── Bias analysis ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  Bias analysis (t-test: is mean Δ significantly different from 0?)")
print("=" * 65)

def significance(p):
    if p < 0.001: return "*** (p<0.001) → significant bias"
    if p < 0.01:  return "**  (p<0.01)  → significant bias"
    if p < 0.05:  return "*   (p<0.05)  → significant bias"
    return "ns  (p≥0.05)  → no significant bias"

for df_other, label_other in zip([df2, df3], OPERATOR_LABELS[1:]):
    print(f"\n  ── {OPERATOR_LABELS[0]} vs {label_other} ──")
    for pt_title, xcol, ycol in POINTS:
        deltas_x, deltas_y = [], []
        for mirror in orientations:
            sub1 = df1[df1['mirror'] == mirror]
            sub_other = df_other[df_other['mirror'] == mirror]
            if sub1.empty or sub_other.empty:
                continue
            deltas_x.append(sub_other[xcol].mean() - sub1[xcol].mean())
            deltas_y.append(sub_other[ycol].mean() - sub1[ycol].mean())

        if len(deltas_x) < 2:
            print(f"\n  {pt_title}: not enough orientations for t-test (n={len(deltas_x)})")
            continue

        deltas_x = np.array(deltas_x)
        deltas_y = np.array(deltas_y)
        t_x, p_x = stats.ttest_1samp(deltas_x, popmean=0)
        t_y, p_y = stats.ttest_1samp(deltas_y, popmean=0)

        print(f"\n  {pt_title} (n={len(deltas_x)} orientations):")
        print(f"    Δx: mean={deltas_x.mean():+.3f} px,  std={deltas_x.std():.3f},  "
              f"t={t_x:.3f},  p={p_x:.4f}  {significance(p_x)}")
        print(f"    Δy: mean={deltas_y.mean():+.3f} px,  std={deltas_y.std():.3f},  "
              f"t={t_y:.3f},  p={p_y:.4f}  {significance(p_y)}")

print()
print("=" * 65)
print()

# ── Print mean position differences ───────────────────────────────────────────
for df_other, label_other in zip([df2, df3], OPERATOR_LABELS[1:]):
    print()
    print("=" * 65)
    print(f"  Mean position differences: {OPERATOR_LABELS[0]} vs {label_other}")
    print("=" * 65)

    for mirror in orientations:
        sub1 = df1[df1['mirror'] == mirror]
        sub_other = df_other[df_other['mirror'] == mirror]
        if sub1.empty or sub_other.empty:
            continue
        print(f"\nMirror {int(mirror)}:")
        print(f"  {'Point':<10}  {'Δx (px)':>10}  {'Δy (px)':>10}  {'Δr (px)':>10}")
        print(f"  {'-'*46}")
        for pt_title, xcol, ycol in POINTS:
            mx1, my1 = sub1[xcol].mean(), sub1[ycol].mean()
            mx2, my2 = sub_other[xcol].mean(), sub_other[ycol].mean()
            dx = mx2 - mx1
            dy = my2 - my1
            dr = np.sqrt(dx**2 + dy**2)
            print(f"  {pt_title:<10}  {dx:>+10.3f}  {dy:>+10.3f}  {dr:>10.3f}")

    print()
    print("=" * 65)
    print()
plt.tight_layout()

# Save next to the first Excel file
out_dir = os.path.dirname(os.path.abspath(EXCEL_PATH_1))
base = 'fiducials_comparison'
plt.savefig(os.path.join(out_dir, base + '.pdf'), dpi=150, bbox_inches='tight',
            facecolor='white', format='pdf')
plt.savefig(os.path.join(out_dir, base + '.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
print(f"Figures saved to: {out_dir}")
plt.show()
print("Done.")