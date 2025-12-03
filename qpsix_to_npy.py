import os
import matplotlib.pyplot as plt
from optics.zygo import ZygoData
import numpy as np

# --- paramètres ---
input_folder =input("️  Enter folder path containing datx files: ").strip().strip("'\"")
output_folder = os.path.join(input_folder, "npy_exports")

# Crée le dossier de sortie s’il n’existe pas
os.makedirs(output_folder, exist_ok=True)
# --- nom(s) de dossier à ignorer ---
IGNORED_FOLDERS = {"shutter"}
# --- parcours de tous les fichiers .datx ---
for dirpath, dirnames, filenames in os.walk(input_folder):

    # retirer les dossiers à ignorer
    dirnames[:] = [d for d in dirnames if d not in IGNORED_FOLDERS]
    for filename in filenames:
        if filename.lower().endswith(".qpsix"):
            filepath = os.path.join(dirpath, filename)
            print(f"Traitement de : {filepath}")

            try:
                zdata = ZygoData(filepath)
                img = np.max(zdata.data, axis=0)  # ou [1]
                # img=zdata.data[1]
                # plt.imshow(img, origin='lower')
                # chemin relatif pour nom unique
                rel_path = os.path.relpath(filepath, input_folder)
                safe_name = rel_path.replace(os.sep, "_").replace(".qpsix", "_qpsix_max.npy")
                output_path = os.path.join(output_folder, safe_name)

                # sauvegarde des données brutes
                np.save(output_path, img)
                print(f"  → Données sauvegardées : {output_path}")

            except Exception as e:
                print(f" Erreur pour {filepath} : {e}")
