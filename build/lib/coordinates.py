import numpy as np
from optical import lw_surface, lw_surface_torus, sw_surface

for surface, name in zip([sw_surface, lw_surface, lw_surface_torus], ['SW', 'LW', 'LW torus']):

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

    print('__________________________________________')
    print(name)
    np.set_printoptions(precision=6, suppress=True)
    print('Grating to EGA')
    print(grating_to_ega_matrix)
    print('EGA to grating')
    print(surface.rotation_matrix())
    np.set_printoptions(precision=9, suppress=True)
    prefix = 'xi' if name == 'SW' else 'eta'
    print(f'Tx={Tx:.5f}°')
    print(f'Ty={Ty:.5f}°')
    print(f'Tz={Tz:.5f}°')
    print(f'Roll {np.degrees(np.arccos(np.dot(xi_x, x))):.5f}°')
    print(f'Plan XY: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xy:.5f}°')
    print(f'Plan XY: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_xy:.5f}°')
    print(f'Plan YZ: angle (Y, $\\{prefix}_y$)={y_proj_xi_y_to_yz:.5f}°')
    print(f'Plan YZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_yz:.5f}°')
    print(f'Plan XZ: angle (X, $\\{prefix}_x$)={x_proj_xi_x_to_xz:.5f}°')
    print(f'Plan XZ: angle (Z, $\\{prefix}_z$)={z_proj_xi_z_to_xz:.5f}°')
