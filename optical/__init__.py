import copy
from optics import Standard, EllipticalGrating, CircularAperture, RectangularAperture, Substrate

# Useful area defined in the (x, y) plane of the EGA coordinate system, i.e. in the middle of the two halves
useful_area = CircularAperture(34.2)

lw_surface = EllipticalGrating(1 / 1008.9554166, 1 / 1010.2811818, 1933.255026,
                               dx=0, dy=2.0386372, dz=0,
                               alpha=0.346498, beta=0.7733422, gamma=0, degrees=True)
lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2 - 0.15 / 2)
lw_substrate = Substrate(lw_surface, lw_aperture, useful_area)
# Dummy substrate with spherical surface
dummy_lw_aperture = RectangularAperture(61.925, 86.188, dx=-61.925 / 2 - 0.15 / 2 + 20)
dummy_lw_substrate = Substrate(lw_substrate.best_sphere, copy.deepcopy(dummy_lw_aperture))
# Rectangular substrate pre-cutting to octagon
rectangular_lw_substrate = Substrate(copy.deepcopy(lw_substrate.surface), copy.deepcopy(dummy_lw_aperture))

sw_surface = Standard(516.0274, -0.522870,
                      dx=31.8380922, dy=15.0690780, dz=2.446336,
                      alpha=3.1515375, beta=-5.0875302, gamma=0, degrees=True)
sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2 + 0.15 / 2)
sw_substrate = Substrate(sw_surface, sw_aperture, useful_area)
# Dummy substrate with spherical surface
dummy_sw_aperture = RectangularAperture(61.925, 86.188, dx=61.925 / 2 + 0.15 / 2 - 20)
dummy_sw_substrate = Substrate(sw_substrate.best_sphere, copy.deepcopy(dummy_sw_aperture))
# Rectangular substrate pre-cutting to octagon
rectangular_sw_substrate = Substrate(copy.deepcopy(sw_substrate.surface), copy.deepcopy(dummy_sw_aperture))

