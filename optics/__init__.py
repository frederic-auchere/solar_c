from .surfaces import EllipticalGrating, Standard, CircularAperture, RectangularAperture, Substrate, Sphere

# Useful area defined in the (x, y) plane of the EGA coordinate system, i.e. in the middle of the two halves
useful_area = CircularAperture(34.8)

lw_surface = EllipticalGrating(1 / 845.139, 1 / 846.299, 1356.579, 0.0, 2.002)
lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2)
lw_substrate = Substrate(lw_surface, lw_aperture, useful_area)

sw_surface = EllipticalGrating(1 / 730.504, 1 / 730.941, 1035.820, 19.042, 24.507)
sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2)
sw_substrate = Substrate(sw_surface, sw_aperture, useful_area)
