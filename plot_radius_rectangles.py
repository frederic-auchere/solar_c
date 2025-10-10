import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import statistics

# Lire le fichier Excel
fichier_excel = "/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/02 - Radius/radius_curvature_LW.xlsx"

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# === Parcours de chaque feuille ===
for feuille, df in sheets.items():
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
            ax1.text(x1.min(), y1.max(), f"{feuille}: y={slope1:.2f}x+{intercept1:.2f}, R²={r2_1:.3f}",
                     fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    # === Jeu 2 ===
    if all(col in df.columns for col in col_jeu2):
        x2 = prepare_data(df[col_jeu2[0]].dropna().copy())
        y2 = prepare_data(df[col_jeu2[1]].dropna().copy())
        if len(x2) == len(y2) and len(x2) > 1:
            y_pred2, slope2, intercept2, r2_2 = faire_regression(x2.values, y2.values)
            ax2.scatter(x2, y2, label=f"{feuille}")
            ax2.plot(x2, y_pred2, linestyle='--', label=f"{feuille} (régr.)")
            ax2.text(x2.min(), y2.max(), f"{feuille}: y={slope2:.2f}x+{intercept2:.2f}, R²={r2_2:.3f}",
                     fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    # === Stocker la différence si les deux intercepts sont valides
    if intercept1 is not None and intercept2 is not None:
        delta = intercept2 - intercept1
        intercepts_diff.append(f"{feuille}: Δb = {delta:.2f}")
        delta_all.append(delta)
# === Mise en forme finale ===
ax1.set_title("Jeu 1 : Confocal")
ax1.set_xlabel("lambda power")
ax1.set_ylabel("Déplacement")
ax1.grid(True)
ax1.legend()

ax2.set_title("Jeu 2 : Catseye")
ax2.set_xlabel("lambda power")
ax2.set_ylabel("Déplacement")
ax2.grid(True)
ax2.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])

rms = math.sqrt(sum(x**2 for x in delta_all) / len(delta_all))

# Calcul écart-type (standard deviation)
std_dev = statistics.stdev(delta_all)

print(f"RMS: {rms:.6f}")
print(f"Écart-type: {std_dev:.6f}")
titre_global = "\n".join(intercepts_diff)
plt.suptitle("Différences entre ordonnées à l'origine (catseye - confocal)\n" + titre_global,
             fontsize=12, y=1.02)
plt.show()
