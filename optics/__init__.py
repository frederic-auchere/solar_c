from .surfaces import EllipticalGrating, Standard, CircularAperture, RectangularAperture, Substrate, Sphere

# Useful area defined in the (x, y) plane of the EGA coordinate system, i.e. in the middle of the two halves
# Values from SOLAR-C(EUVST)_Optical_Design_Summary(v20230707).pdf

# Note that footprint is not exactly circular, but the true footprint is inscribed in this circle
useful_area = CircularAperture(34.2)

lw_surface = EllipticalGrating(1 / 1008.9554, 1 / 1010.2812, 1933.2550, 0.0, 2.0386)
lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2)
lw_substrate = Substrate(lw_surface, lw_aperture, useful_area)

sw_surface = Standard(516.0144, -0.517006, 33.0544, 15.3149)
sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2)
sw_substrate = Substrate(sw_surface, sw_aperture, useful_area)
