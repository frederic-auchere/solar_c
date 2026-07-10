import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import statistics
from matplotlib.transforms import Bbox
import os

def place_text_smart(ax, x, y, text, existing_boxes, fontsize=9):
    """
    Try to place text at data coords (x, y).
    If box overlaps with existing boxes, move it to safe axes positions.
    """
    # --- First try the requested (on-line) position ---
    txt = ax.text(x, y, text, fontsize=fontsize,
                  bbox=dict(facecolor='white', alpha=0.5))

    # Render to get its size
    ax.figure.canvas.draw()
    bbox = txt.get_window_extent()

    # Check overlap with any existing boxes
    for b in existing_boxes:
        if Bbox.intersection(bbox, b) is not None:
            txt.remove()   # remove the conflicting one
            break
    else:
        # No overlap → accept this placement
        existing_boxes.append(bbox)
        return txt

    # --- Fallback positions (axes coordinates) ---
    fallback_positions = [
        (0.02, 0.98),   # top-left
        (0.98, 0.98),   # top-right
        (0.02, 0.02),   # bottom-left
        (0.98, 0.02),   # bottom-right
    ]

    for px, py in fallback_positions:
        txt = ax.text(px, py, text, fontsize=fontsize,
                      transform=ax.transAxes, va='top', ha='left',
                      bbox=dict(facecolor='white', alpha=0.5))

        ax.figure.canvas.draw()
        bbox = txt.get_window_extent()

        # check again
        if not any(Bbox.intersection(bbox, b) is not None for b in existing_boxes):
            existing_boxes.append(bbox)
            return txt

    return txt  # fallback even if overlapping (very unlikely)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})
# Lire le fichier Excel
fichier_excel = input("path to radius data: ").strip().strip("'").strip('"')#"/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/02 - Radius/radius_curvature_LW.xlsx"
print(fichier_excel)

dossier_excel = os.path.dirname(fichier_excel)
# Colonnes à chercher dans chaque feuille
col_jeu1 = ["lambda power confocal", "Deplacement confocal"]
col_jeu2 = ["lambda power catseye", "Deplacement catseye"]

# Fonctions utilitaires
def prepare_data(series):
    return series.astype(str).str.replace(',', '.').astype(float)

def faire_regression(x, y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred, model.coef_[0], model.intercept_, r2_score(y, y_pred)

# Lecture de toutes les feuilles
sheets = pd.read_excel(fichier_excel, sheet_name=None, skiprows=2)



intercepts_diff = []
delta_all=[]
# Préparer les deux subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
existing_boxes_ax1 = []
existing_boxes_ax2 = []
# === Parcours de chaque feuille ===
for feuille, df in sheets.items():

    # ⛔ Ignorer les feuilles dont le nom commence par "ignore" (insensible à la casse)
    if feuille.lower().startswith("ignore"):
        continue
    df.columns = df.columns.str.strip()

    intercept1 = intercept2 = None  # initialiser

    # === Jeu 1 ===
    if all(col in df.columns for col in col_jeu1):
        x1 = prepare_data(df[col_jeu1[0]].dropna().copy())
        y1 = prepare_data(df[col_jeu1[1]].dropna().copy())
        if len(x1) == len(y1) and len(x1) > 1:
            y_pred1, slope1, intercept1, r2_1 = faire_regression(x1.values, y1.values)
            ax1.scatter(x1, y1, label=f"{feuille}")
            ax1.plot(x1, y_pred1, linestyle='--', label=f"{feuille} (régr.)")
            x_line_pos1 = x1.min() + 0.8 * (x1.max() - x1.min())
            y_line_pos1 = slope1 * x_line_pos1 + intercept1

            place_text_smart(
                ax1,
                x_line_pos1,
                y_line_pos1,
                f"{feuille}: y={slope1:.2f}x+{intercept1:.2f}, R²={r2_1:.3f}",
                existing_boxes_ax1,
                fontsize=9
            )

    # === Jeu 2 ===
    if all(col in df.columns for col in col_jeu2):
        x2 = prepare_data(df[col_jeu2[0]].dropna().copy())
        y2 = prepare_data(df[col_jeu2[1]].dropna().copy())
        if len(x2) == len(y2) and len(x2) > 1:
            y_pred2, slope2, intercept2, r2_2 = faire_regression(x2.values, y2.values)
            ax2.scatter(x2, y2, label=f"{feuille}")
            ax2.plot(x2, y_pred2, linestyle='--', label=f"{feuille} (régr.)")
            x_line_pos = x2.min() + 0.8 * (x2.max() - x2.min())
            y_line_pos = slope2 * x_line_pos + intercept2

            place_text_smart(
                ax2,
                x_line_pos,
                y_line_pos,
                f"{feuille}: y={slope2:.2f}x+{intercept2:.2f}, R²={r2_2:.3f}",
                existing_boxes_ax2,
                fontsize=9
            )

    # === Stocker la différence si les deux intercepts sont valides
    if intercept1 is not None and intercept2 is not None:
        delta = intercept2 - intercept1
        intercepts_diff.append(f"{feuille}: Δb = {delta:.2f}")
        delta_all.append(delta)
# === Mise en forme finale ===
ax1.set_title("Confocal")
ax1.set_xlabel("lambda power")
ax1.set_ylabel("Displacement")
ax1.grid(True)
ax1.legend()

ax2.set_title("Catseye")
ax2.set_xlabel("lambda power")
ax2.set_ylabel("Displacement")
ax2.grid(True)
ax2.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Calcul RMS uniquement si au moins 1 delta
if len(delta_all) > 0:
    rms = math.sqrt(sum(x**2 for x in delta_all) / len(delta_all))
else:
    rms = float('nan')

# Calcul écart-type uniquement si au moins 2 valeurs
if len(delta_all) > 1:
    std_dev = statistics.stdev(delta_all)
else:
    std_dev = float('nan')

print(f"RMS: {rms:.6f}")
print(f"Écart-type: {std_dev:.6f}")
titre_global = "\n".join(intercepts_diff)
plt.suptitle("Differences between regression intercepts (catseye − confocal)\n" + titre_global,
             fontsize=12, y=1.02)
# plt.tight_layout()
output_path = os.path.join(dossier_excel, f'{feuille.strip("ignore")}_radius.png')
plt.savefig(output_path, bbox_inches='tight')
# plt.show()
# plt.close()