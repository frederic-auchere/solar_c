import numpy as np

solar_constant = 1421  # [W.m^-2]

mirror_radius = 0.14  # [m]

mirror_surface = np.pi * mirror_radius ** 2
mirror_reflectivity = 0.4

slit_width = [0.2, 0.4, 0.8, 1.6, 40]  # [arcseconds]
slit_length = 300  # [arcseconds]
slit_area = [slit_length * sw for sw in slit_width]

solar_radius = 0.266 * 3600  # [arcseconds]

total_power = solar_constant * mirror_surface * mirror_reflectivity

solar_surface_arcseconds = np.pi * solar_radius ** 2

power = [total_power * sa / solar_surface_arcseconds for sa in slit_area]

print(power)

