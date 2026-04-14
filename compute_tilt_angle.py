import numpy as np
from optics.geometry import NormalVector

from optical import lw_surface, sw_surface
from scipy.spatial.transform import Rotation
from optics.zygo import SagData
from optics import surfaces
from optical import rectangular_sw_substrate
from optics.geometry import NormalVector,inverse_rotation_matrix



# files_zygo = ('/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin_for_decentering.fits')
# dx=31.845+13.888 #sw1
# dy=15.073+13.303#sw1
# radius=516.753 #sw1
# dx=32.289+13.86 #sw5
# dy=16.023+13.251#sw5
# radius=516.783 #sw5
dx=32.506+14.119 #sw7
dy=16.29+13.130#sw7
radius=516.823 #sw7 nominal pour l'instant (pour ce substrat)
normal_vector=NormalVector(-dx,-dy,np.sqrt(radius**2-dx**2-dy**2)) #vecteur normal avec les dx et dy (ecart entre vertex et centre asphere)
#en z on a la racine rayon au carré moins x carré et y carré (equation cercle)
alpha = -np.arctan2(normal_vector.y, normal_vector.z)
beta = np.arctan2(normal_vector.x, np.sqrt(normal_vector.y ** 2 + normal_vector.z ** 2))


rotation = Rotation.from_euler('xyz', (-alpha, -beta, 0)).as_matrix()
translation = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
rotation_matrix=rotation@translation

# for surface, name in zip([lw_surface, sw_surface], ['LW', 'SW']):

grating_to_ega_matrix =  inverse_rotation_matrix(rotation_matrix)


x = np.array((1, 0, 0))
y = np.array((0, 1, 0))
z = np.array((0, 0, 1))

# coordinates of unit vectors in EGA
xi_x = grating_to_ega_matrix[:, 0:3] @ x
xi_y = grating_to_ega_matrix[:, 0:3] @ y
xi_z = grating_to_ega_matrix[:, 0:3] @ z

Tx = np.degrees(np.arccos(np.dot(xi_z, x)))
Ty = np.degrees(np.arccos(np.dot(xi_z, y)))
Tz = np.degrees(np.arccos(np.dot(xi_z, z)))

x_proj_xi_x_to_xy = np.degrees(np.arctan(np.dot(xi_x, y)/np.dot(xi_x, x)))
y_proj_xi_y_to_xy = np.degrees(np.arctan(np.dot(xi_y, x)/np.dot(xi_y, y)))

y_proj_xi_y_to_yz = np.degrees(np.arctan(np.dot(xi_y, z)/np.dot(xi_y, y)))
z_proj_xi_z_to_yz = np.degrees(np.arctan(np.dot(xi_z, y)/np.dot(xi_z, z)))

x_proj_xi_x_to_xz = np.degrees(np.arctan(np.dot(xi_x, z)/np.dot(xi_x, x)))
z_proj_xi_z_to_xz = np.degrees(np.arctan(np.dot(xi_z, x)/np.dot(xi_z, z)))


prefix = 'xi'
print(f'Tx={Tx:.5f}°')
print(f'Ty={Ty:.5f}°')
print(f'Tz={Tz:.5f}°')
print(f'Plan XY: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xy:.5f}°')
print(f'Plan XY: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_xy:.5f}°')
print(f'Plan YZ: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_yz:.5f}°')
print(f'Plan YZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_yz:.5f}°')
print(f'Plan XZ: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xz:.5f}°')
print(f'Plan XZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_xz:.5f}°')

# ega_to_global = np.array([[-1, 0, 0, 0],
#                           [ 0, 0.9969180, 0.0784504, -243.5351],
#                           [ 0, 0.0784504, -0.9969180, 299.0754],
#                           [0, 0, 0, 1]])
#
# ega_to_mechanical = np.array([[1, 0, 0, 0],
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 52],
#                               [0, 0, 0, 1]])
#
# reference_to_mechanical = np.array([[1, 0, 0, -22],
#                                     [0, 1, 0, -23.0940],
#                                     [0, 0, 1, 32],
#                                     [0, 0, 0, 1]])
#
# mechanical_to_ega = np.linalg.inv(ega_to_mechanical)
# reference_to_global = ega_to_global @ mechanical_to_ega @ reference_to_mechanical
#
# np.set_printoptions(4, suppress=True)
# print(reference_to_global[0:3, :])

# gamma, beta, alpha = Rotation.from_matrix(reference_to_global[0:3, 0:3]).as_euler('zyx', degrees=True)
# print(f'alpha: {alpha:.4f} beta: {beta:.4f} gamma: {gamma:.4f} ')
#
# alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
# rx = rotationmatrix(alpha, 2)
# ry = rotationmatrix(beta, 1)
# rz = rotationmatrix(gamma, 0)
#
# print(rx @ ry @ rz)

import numpy as np

def incertitude_angle(dx1, dx2, rayon, sigma_dx1=0.01, sigma_dx2=0.02):
    u = (dx1 + dx2) / rayon
    sigma_u = np.sqrt(sigma_dx1**2 + sigma_dx2**2)  # = 0.02236
    sigma_angle = sigma_u / (rayon * np.sqrt(1 - u**2))
    # ou de manière équivalente :
    # sigma_angle = sigma_u / np.sqrt(rayon**2 - (dx1+dx2)**2)
    return sigma_angle  # en radians

incertitude_angle(31.86,13.88,516.753)