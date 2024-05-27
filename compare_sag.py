from optical import sw_substrate, lw_substrate, dummy_sw_substrate, dummy_lw_substrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates

# X, Y, Z, define in substrate coordinates (origin at center of substrate)
bertin_lw = (
    (0,    0,        0.000000,    0.000000),
    (-30,    0,        1.890950,    1.890941),
    (-20,    0,        1.070040,    1.070034),
    (-10,    0,        0.439822,    0.439819),
    (10,    0,        -0.249660,    -0.249657),
    (20,    0,        -0.309336,    -0.309330),
    (30,    0,        -0.179164,    -0.179154),
    (0,    -40,        1.428573,    1.428574),
    (0,    -30,        0.787123,    0.787125),
    (0,    -20,        0.335261,    0.335261),
    (0,    -10,        0.072893,    0.072894),
    (0,    10,        0.116605,    0.116605),
    (0,    20,        0.422786,    0.422785),
    (0,    30,        0.918681,    0.918680),
    (0,    40,        1.604480,    1.604479),
    (-30,    -40,        3.321228,    3.321220),
    (-30,    40,        3.498400,    3.498389),
    (30,    40,        1.424347,    1.424355),
    (30,    -40,        1.248401,    1.248412)
)
bertin_sw = (
    (0,    0,        0.000000,    0.000000),
    (-30,    0,        -0.588570,    -0.584555),
    (-20,    0,        -0.585490,    -0.582812),
    (-10,    0,        -0.389507,    -0.388167),
    (10,    0,        0.583767,    0.582423),
    (20,    0,        1.362641,    1.359948),
    (30,    0,        2.337563,    2.333514),
    (0,    -40,        0.516657,    0.516640),
    (0,    -30,        0.097620,    0.097610),
    (0,    -20,        -0.128316,    -0.128322),
    (0,    -10,        -0.160960,    -0.160962),
    (0,    10,        0.354970,    0.354971),
    (0,    20,        0.904463,    0.904463),
    (0,    30,       1.649103,    1.649102),
    (0,    40,        2.589622,    2.589618),
    (-30,    -40,        -0.078176,    -0.074173),
    (-30,    40,        1.997276,    2.001307),
    (30,    40,        4.939174,    4.935101),
    (30,    -40,        2.857503,    2.853433)
)


dummy_lw_substrate.surface = lw_substrate.surface
dummy_sw_substrate.surface = sw_substrate.surface


# Compare the computed sag with that from Bertin

fig, axes = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))

for name, substrate, bertin_sag, ax in zip(['LW', 'SW'],
                                           [dummy_lw_substrate, dummy_sw_substrate],
                                           [bertin_lw, bertin_sw],
                                           axes):
    substrate.useful_area = None
    dx = dy = 1  # mm
    bertin_x = [b[0] for b in bertin_sag]
    bertin_y = [b[1] for b in bertin_sag]
    bertin_z = [b[3] for b in bertin_sag]
    x_min, x_max = min(bertin_x) - 2 * dx, max(bertin_x) + 2 * dx
    y_min, y_max = min(bertin_y) - 2 * dy, max(bertin_y) + 2 * dy
    nx = (x_max - x_min) // dx + 1
    ny = (y_max - y_min) // dy + 1
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    x_ega = x + substrate.aperture.dx
    y_ega = y + substrate.aperture.dy
    ega_grid = np.meshgrid(x_ega, y_ega)
    sag = substrate.sag(ega_grid)

    ix = [int(( len(x) - 1 ) * (bx - x_min ) /  ( x_max - x_min )) for bx in bertin_x]
    iy = [int(( len(y) - 1 ) * (by - y_min ) /  ( y_max - y_min )) for by in bertin_y]
    interpolated_sag = sag[(iy, ix)]
    delta = (interpolated_sag - np.array(bertin_z)) * 1e3
    print(np.nanmax(delta) - np.nanmin(delta), np.nanstd(delta))
    sc = ax.scatter(bertin_x, bertin_y, c=delta)
    bar = fig.colorbar(sc, ax=ax)
    bar.set_label('Sag difference [$\mu$m]')
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.set_xlabel('EGA x (mm)')
    ax.set_ylabel('EGA y (mm)')
    # Draw aperture of dummy substrate
    rect = Rectangle((substrate.aperture.dx - substrate.aperture.x_width / 2,
                      substrate.aperture.dy - substrate.aperture.y_width / 2),
                     substrate.aperture.x_width,
                     substrate.aperture.y_width,
                     color='red', fill=None)
    ax.add_patch(rect)
