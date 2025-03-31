import numpy as np
from optical import lw_surface, sw_surface
from scipy.spatial.transform import Rotation


def rotationmatrix(angle, axis):
    """
    Returns a rotation matrix about the specified axis (z=0, y=1, x=2) for the
    specififed angle (in radians).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)

    if axis == 0:  # Rz
        matrix = np.array([[cos, -sin, 0],
                           [sin, cos, 0],
                           [0, 0, 1]])
    elif axis == 1:  # Ry
        matrix = np.array([[cos, 0, sin],
                           [0, 1, 0],
                           [-sin, 0, cos]])
    elif axis == 2:  # Rx
        matrix = np.array([[1, 0, 0],
                           [0, cos, -sin],
                           [0, sin, cos]])

    return matrix


for surface, name in zip([lw_surface, sw_surface], ['LW', 'SW']):

    grating_to_ega_matrix = surface.inverse_rotation_matrix()

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

    print(name)
    prefix = 'xi' if name == 'SW' else 'eta'
    print(f'Tx={Tx:.5f}°')
    print(f'Ty={Ty:.5f}°')
    print(f'Tz={Tz:.5f}°')
    print(f'Plan XY: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xy:.5f}°')
    print(f'Plan XY: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_xy:.5f}°')
    print(f'Plan YZ: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_yz:.5f}°')
    print(f'Plan YZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_yz:.5f}°')
    print(f'Plan XZ: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xz:.5f}°')
    print(f'Plan XZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_xz:.5f}°')

ega_to_global = np.array([[-1, 0, 0, 0],
                          [ 0, 0.9969180, 0.0784504, -243.5351],
                          [ 0, 0.0784504, -0.9969180, 299.0754],
                          [0, 0, 0, 1]])

ega_to_mechanical = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 52],
                              [0, 0, 0, 1]])

reference_to_mechanical = np.array([[1, 0, 0, -22],
                                    [0, 1, 0, -23.0940],
                                    [0, 0, 1, 32],
                                    [0, 0, 0, 1]])

mechanical_to_ega = np.linalg.inv(ega_to_mechanical)
reference_to_global = ega_to_global @ mechanical_to_ega @ reference_to_mechanical

np.set_printoptions(4, suppress=True)
print(reference_to_global[0:3, :])

gamma, beta, alpha = Rotation.from_matrix(reference_to_global[0:3, 0:3]).as_euler('zyx', degrees=True)
print(f'alpha: {alpha:.4f} beta: {beta:.4f} gamma: {gamma:.4f} ')

alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
rx = rotationmatrix(alpha, 2)
ry = rotationmatrix(beta, 1)
rz = rotationmatrix(gamma, 0)

print(rx @ ry @ rz)
