import numpy as np
import matplotlib.pyplot as plt
import csv

abs_vals_confocal = []  # valeurs des focus (en lambda)
ord_vals_confocal = []  # valeurs des positions (en um)
abs_vals_catseye = []  # valeurs des focus (en lambda)
ord_vals_catseye = []  # valeurs des positions (en um)
valeurs_radius = []
for k in range(3):
    with open("C:\\Users\\mart1\\OneDrive\\Documents\\stage IAS\\radius_curvature_LW_"+str(k)+".csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)  # sauter la ligne d'en-tête si elle existe
        next(csv_reader)  # sauter la deuxième ligne
        next(csv_reader)  # sauter la troisième ligne

        for row in csv_reader:
            modif_coma_abs_confocal = row[0].replace(",",".")
            modif_coma_ord_confocal = row[1].replace(",",".")
            modif_coma_abs_catseye = row[2].replace(",",".")
            modif_coma_ord_catseye = row[3].replace(",",".")
            print(modif_coma_abs_confocal, modif_coma_ord_confocal, modif_coma_abs_catseye, modif_coma_ord_catseye)
    # supposons que la 1ère colonne est focus et la 2ème colonne la position
            abs_vals_confocal.append(float(modif_coma_abs_confocal))
            ord_vals_confocal.append(float(modif_coma_ord_confocal))
            abs_vals_catseye.append(float(modif_coma_abs_catseye))
            ord_vals_catseye.append(float(modif_coma_ord_catseye))
    # Ajustement d'une droite y = a*x + b
    a_confocal, b_confocal = np.polyfit(abs_vals_confocal, ord_vals_confocal, 1)
    a_catseye, b_catseye = np.polyfit(abs_vals_catseye, ord_vals_catseye, 1)
    valeurs_radius.append(b_catseye-b_confocal)

    print('la position confocal est:',b_confocal,'(um)',' et la position du catseye est:',b_catseye,'(um)')
    print('le rayon de courbure vaut:',b_catseye-b_confocal)
    # Calcul des valeurs ajustées (régression)
    reg_confocal = [a_confocal * x + b_confocal for x in abs_vals_confocal]
    reg_catseye = [a_catseye * x + b_catseye for x in abs_vals_catseye]


    # Affichage
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(abs_vals_confocal,ord_vals_confocal, 'o')
    ax1.plot(abs_vals_confocal,reg_confocal)
    ax1.set_xlabel('focus(en lambda)')
    ax1.set_ylabel('position (en um)')
    ax1.set_title('position confocale')

    ax2.plot(abs_vals_catseye,ord_vals_catseye, 'o')
    ax2.plot(abs_vals_catseye,reg_catseye)
    ax2.set_xlabel('focus(en lambda)')
    ax2.set_ylabel('position (en um)')
    ax2.set_title('position catseye')
print('le rayon de courbure moyen vaut:', np.mean(valeurs_radius),'(um)')
print("l'écart-type de mesure vaut:", np.std(valeurs_radius),'(um)')
plt.show()