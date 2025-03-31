import copy
from optics import Standard, EllipticalGrating, PieAperture, RectangularAperture, Substrate
from optics.geometry import Point

STEP = None  # mm

# Useful area defined in the (x, y) plane of the EGA coordinate system, i.e. in the middle of the two halves

# LW definitions

lw_useful_area = PieAperture(34.2 / 2, 90, -90)

lw_surface = EllipticalGrating(1 / 1008.9554166, 1 / 1010.2811818, 1933.255026,
                               dx=0, dy=2.0386372, dz=0,
                               alpha=0.346498, beta=0.7733422, gamma=0, degrees=True)
lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2 - 0.15 / 2)
lw_substrate = Substrate(lw_surface,
                         lw_aperture,
                         lw_useful_area,
                         name='LW',
                         x_grid_step=STEP)
# Dummy substrate with spherical surface
spherical_lw_aperture = RectangularAperture(61.925, 86.188, dx=-61.925 / 2 - 0.15 / 2 + 20)
spherical_lw_substrate = Substrate(lw_substrate.best_sphere,
                               copy.deepcopy(spherical_lw_aperture),
                               name='Spherical LW',
                                   x_grid_step=STEP)
# Rectangular substrate pre-cutting to octagon
rectangular_lw_substrate = Substrate(copy.deepcopy(lw_substrate.surface),
                                     copy.deepcopy(spherical_lw_aperture),
                                     name='Rectangular LW',
                                     x_grid_step=STEP)

xy1 = 17.925, 41.094
xy2 = -40, 0
xy3 = 17.925, -41.094
rectangular_lw_substrate_fiducials = [Point(*xy, float(rectangular_lw_substrate.sag(xy).data)) for xy in [xy1, xy2, xy3]]


# SW definitions

sw_useful_area = PieAperture(34.2 / 2, -90, 90)

sw_surface = Standard(516.0274, -0.522870,
                      dx=31.8380922, dy=15.0690780, dz=2.446336,
                      alpha=3.1515375, beta=-5.0875302, gamma=0, degrees=True)
sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2 + 0.15 / 2)
sw_substrate = Substrate(sw_surface,
                         sw_aperture,
                         sw_useful_area,
                         name='SW',
                         x_grid_step=STEP)
# Dummy substrate with spherical surface
spherical_sw_aperture = RectangularAperture(61.925, 86.188, dx=61.925 / 2 + 0.15 / 2 - 20)
spherical_sw_substrate = Substrate(sw_substrate.best_sphere,
                                   copy.deepcopy(spherical_sw_aperture),
                                   name='Spherical SW',
                                   x_grid_step=STEP)
# Rectangular substrate pre-cutting to octagon
rectangular_sw_substrate = Substrate(copy.deepcopy(sw_substrate.surface),
                                     copy.deepcopy(spherical_sw_aperture),
                                     name='Rectangular SW',
                                     x_grid_step=STEP)

xy1 = -17.925, 41.094
xy2 = 40, 0
xy3 = -17.925, -41.094
rectangular_sw_substrate_fiducials = [Point(*xy, float(rectangular_lw_substrate.sag(xy).data)) for xy in [xy1, xy2, xy3]]
