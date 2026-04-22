import numpy as np
from scipy.spatial.transform import Rotation
from optics.geometry import NormalVector, inverse_rotation_matrix
from optical import lw_surface, sw_surface, rectangular_sw_substrate
from optics.zygo import SagData
from optics import surfaces

# --- Paramètres des substrats ---
SUBSTRATES = {
    'SW_SN1': dict(dx=31.845 + 13.888, dy=15.073 + 13.303, radius=516.753),
    'SW_SN5': dict(dx=32.289 + 13.86,  dy=16.023 + 13.251, radius=516.783),
    'SW_SN7': dict(dx=32.506 + 14.119, dy=16.29  + 13.130, radius=516.823), #rayon nominal pour l'instant, non mesuré
}

# --- Sélection ---
substrate_name = 'SW_SN7'
p = SUBSTRATES[substrate_name]
dx, dy, radius = p['dx'], p['dy'], p['radius']
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
print(f'alpha Plan YZ: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_yz:.5f}°')
print(f'Plan YZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_yz:.5f}°')
print(f'Plan XZ: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xz:.5f}°')
print(f'beta Plan XZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_xz:.5f}°')

