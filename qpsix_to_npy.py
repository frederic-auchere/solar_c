import os
import matplotlib.pyplot as plt
from optics.zygo import ZygoData
import numpy as np

# --- paramètres ---
input_folder = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/Old-avant_reprise_Bertin_Sept_2025/FM_LW_SN1_old/Zygo/Tilt/20250902'
output_folder = os.path.join(input_folder, "png_exports")

# Crée le dossier de sortie s’il n’existe pas
os.makedirs(output_folder, exist_ok=True)

# --- parcours de tous les fichiers .datx ---
for dirpath, dirnames, filenames in os.walk(input_folder):
    for filename in filenames:
        if filename.lower().endswith(".datx"):
            filepath = os.path.join(dirpath, filename)
            print(f"Traitement de : {filepath}")

            try:
                zdata = ZygoData(filepath)
                img = zdata.data[0]  # ou [1]

                # chemin relatif pour nom unique
                rel_path = os.path.relpath(filepath, input_folder)
                safe_name = rel_path.replace(os.sep, "_").replace(".datx", ".npy")
                output_path = os.path.join(output_folder, safe_name)

                # sauvegarde des données brutes
                np.save(output_path, img)
                print(f"  → Données sauvegardées : {output_path}")

            except Exception as e:
                print(f" Erreur pour {filepath} : {e}")
