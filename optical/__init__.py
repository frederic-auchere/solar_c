from optics import Standard, EllipticalGrating, CircularAperture, RectangularAperture, Substrate

# Useful area defined in the (x, y) plane of the EGA coordinate system, i.e. in the middle of the two halves
useful_area = CircularAperture(34.2)

# Old definitions
# lw_surface = EllipticalGrating(1 / 845.139, 1 / 846.299, 1356.579,
#                                dy=2.002, alpha=0.3424, beta=0.7735, degrees=True)
# lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2)
# lw_substrate = Substrate(lw_surface, lw_aperture, useful_area)
#
# sw_surface = EllipticalGrating(1 / 730.504, 1 / 730.941, 1035.820,
#                                dx=24.50720, dy=19.04215, dz=1.988, alpha=3.59312, beta=-4.05954, degrees=True)
# sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2)
# sw_substrate = Substrate(sw_surface, sw_aperture, useful_area)

lw_surface = EllipticalGrating(1 / 1008.95542, 1 / 1010.28118, 1933.25503,
                               dx=0, dy=2.03864, dz=0,
                               alpha=0.34650, beta=0.77334, gamma=0, degrees=True)
lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2)
lw_substrate = Substrate(lw_surface, lw_aperture, useful_area)

sw_surface = Standard(516.0144, -0.517006,
                      dx=33.05438, dy=15.31495, dz=2.6241,
                      alpha=3.18010, beta=-5.31673, gamma=0, degrees=True)
sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2)
sw_substrate = Substrate(sw_surface, sw_aperture, useful_area)
