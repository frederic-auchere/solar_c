from scipy.interpolate import griddata
from astropy.io import fits
import matplotlib.pyplot as plt
from optical import rectangular_sw_substrate
from optics.zygo import SagData
from optics import surfaces
import numpy as np
from optical.utils import write_zygo_dat

files_zygo = r'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin_SW1_ZU.fits'

# Lecture du fichier FITS Zygo : pas de rotation (theta=0), pas de binning,
# recadrage automatique sur la zone mesurée
zygo_data = SagData(files_zygo, theta=0, binning=1, auto_crop=True)

# ── Construction de la surface mesurée ──────────────────────────────────────
# Récupération des angles de tip/tilt/rotation nominaux du substrat SW rectangulaire,
# définis par rapport à la normale théorique
alpha, beta, gamma = rectangular_sw_substrate.tip_tilt_from_normal()

# Construction d'une MeasuredSurface qui applique
# la correction de basculement définie par alpha, beta, gamma
measured_surface = surfaces.MeasuredSurface(zygo_data, alpha=alpha, beta=beta, gamma=gamma)

# Création de l'objet Substrate associé à la surface mesurée
measured_substrate = surfaces.Substrate(measured_surface, rectangular_sw_substrate.aperture, rectangular_sw_substrate.useful_area)

grid_step = 0.1   # pas de la grille (sortie) [mm]

# Nombre de points en x et y, calculé à partir des limites physiques du substrat
nx = int((rectangular_sw_substrate.limits[1] - rectangular_sw_substrate.limits[0]) / grid_step)
ny = int((rectangular_sw_substrate.limits[3] - rectangular_sw_substrate.limits[2]) / grid_step)

# Coordonnées des bords de la grille (alignées sur le pas)
x_min = rectangular_sw_substrate.limits[0]
x_max = x_min + grid_step * nx
y_min = rectangular_sw_substrate.limits[2]
y_max = y_min + grid_step * ny


# Grille 2D régulière dans le repère nominal du substrat [mm]
ox, oy = np.meshgrid(np.linspace(x_min, x_max, nx + 1), np.linspace(y_min, y_max, ny + 1))

# ── Sag nominal sur la grille ────────────────────────────────────────────────
# Calcul du sag théorique en chaque point de la grille
nominal_sag = rectangular_sw_substrate.sag((ox, oy)).data


# ── Projection sur la sphère de référence ────────────────────────────────────
# Construction du vecteur homogène (x, y, z, 1) pour chaque point de la grille,
# z étant le sag nominal
xyz = np.stack((ox.ravel(), oy.ravel(), nominal_sag.ravel(), np.ones(ox.size)))

# Application de la matrice de rotation de la surface mesurée :
# passage du repère nominal au repère mesuré (incliné)
x, y, z = measured_substrate.surface.rotation_matrix() @ xyz

# Centre et rayon de la meilleure sphère ajustée au substrat nominal
best_sphere = rectangular_sw_substrate.best_sphere

# Résidu mesuré (écart au nominal) en chaque point, converti de µm en mm
dr = measured_substrate.sag((x, y)).data / 1e6

# Coordonnées du centre de la meilleure sphère, exprimées dans le repère mesuré
# (le centre est décalé de r selon z pour partir du sommet de la sphère)
sx, sy, sz = best_sphere.dx, best_sphere.dy, best_sphere.dz + best_sphere.r
sx, sy, sz = measured_substrate.surface.rotation_matrix() @ np.array((sx, sy, sz, 1))

# Distance entre chaque point de la grille et le centre de la sphère de référence
# ratio = fraction du rayon vecteur correspondant au résidu mesuré dr
ratio = dr / np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)

# Déplacement de chaque point le long du rayon vecteur vers le centre de la sphère,
# d'une amplitude dr : projection du résidu sur la sphère de référence
x += (sx - x) * ratio
y += (sy - y) * ratio
z += (sz - z) * ratio

# ── Retour dans le repère nominal par rotation inverse ────────────────────────
xyz = np.stack((x, y, z, np.ones(x.size)))
nx, ny, nz = measured_substrate.surface.inverse_rotation_matrix() @ xyz

# ── Interpolation sur la grille régulière ────────────────────────────────────
# Certains points peuvent être invalides (NaN) après rotation (en dehors des bords du substrat)
valid = np.logical_and(np.isfinite(nx), np.isfinite(ny))

# Interpolation cubique des points mesurés (irréguliers après rotation)
# vers la grille régulière (ox, oy) (lourd)
sag = griddata((nx[valid], ny[valid]), nz[valid], (ox, oy), method='cubic', rescale=False).reshape(ox.shape)

fig, axes = plt.subplots(1, 3)
# Panneau 1 : sag mesuré reprojeté sur la grille nominale [mm]
im1 = axes[0].imshow(sag, origin='lower', extent=rectangular_sw_substrate.limits)
plt.colorbar(im1, ax=axes[0], label='z sag [mm]')
sag_difference = (sag - nominal_sag) * 1e6
# Panneau 2 : différence sag mesuré – sag nominal [nm]
# (résidu de forme après soustraction de la forme théorique)
im2 = axes[1].imshow(sag_difference, origin='lower', extent=rectangular_sw_substrate.limits, vmin=-20000, vmax=20000)
plt.colorbar(im2, ax=axes[1], label = r'z sag difference [nm]')

# Panneau 3 : écart entre le résidu Zygo brut et le résidu reprojeté [nm]
# (évalue la cohérence de la reprojection sur la sphère de référence)
im3 = axes[2].imshow(measured_substrate.sag((ox, oy)) - sag_difference, origin='lower', extent=rectangular_sw_substrate.limits, vmin=-50, vmax=50)
plt.colorbar(im3, ax=axes[2], label = r'z sag difference difference [nm]')

print(grid_step)

# ── Export DAT pour BERTIN───────────────────────────────────────────────────────────────
# Écriture du résidu de forme (sag_difference converti en mm) dans un fichier DAT
# write_zygo_dat('/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin_SW1_ZU_bascule.dat',sag_difference / 1e6,grid_step)


# ── Export FITS ───────────────────────────────────────────────────────────────
# Écriture du résidu de forme (sag_difference converti en mm) dans un fichier FITS,

outfile=r'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin_SW1_ZU_bascule.fits'
header = fits.header.Header()
header['GX'] = grid_step # pas de grille en x [mm]
header['GY'] = grid_step # pas de grille en y [mm]
header['DX'] = -rectangular_sw_substrate.aperture.limits[0] / header['GX']  # indice x de l'origine
header['DY'] = -rectangular_sw_substrate.aperture.limits[2] / header['GY'] # indice y de l'origine
header['THETA'] = 0 # angle de rotation appliqué (ici nul mais necessaire pour ouvrir le fits)

fits.writeto(outfile, sag_difference / 1e6, header=header, overwrite=True)