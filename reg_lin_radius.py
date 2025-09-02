import numpy as np
import matplotlib.pyplot as plt
import csv

abs_vals = []  # valeurs des focus (en lambda)
ord_vals = []  # valeurs des positions (en um)

with open("C:\\Users\\mart1\\OneDrive\\Documents\\stage IAS\\radius_1_LW.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)  # sauter la ligne d'en-tête si elle existe

    for row in csv_reader:
        modif_coma_abs = row[0].replace(",",".")
        modif_coma_ord = row[1].replace(",",".")
        
# supposons que la 1ère colonne est focus et la 2ème colonne la position
        abs_vals.append(float(modif_coma_abs))
        ord_vals.append(float(modif_coma_ord))

# Ajustement d'une droite y = a*x + b
a, b = np.polyfit(abs_vals, ord_vals, 1)

# # Calcul des valeurs ajustées (régression)
reg = [a * x + b for x in abs_vals]

# Affichage
plt.plot(abs_vals, ord_vals, 'o', label="Données")
plt.plot(abs_vals, reg, label=f"Fit: y = {a:.2f}x + {b:.2f}")
plt.xlabel('Focus (en lambda)')
plt.ylabel('Position (en um)')
plt.title('Focus en fonction de la position par rapport au télémètre')
plt.legend()

# Ajouter un texte indiquant la position confocale
print('la position confocale (respectivement catseye) est:', b)

plt.show()