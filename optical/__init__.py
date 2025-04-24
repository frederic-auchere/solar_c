import copy
from optics import Standard, EllipticalGrating, PieAperture, RectangularAperture, Sphere
from optical.surfaces import EGASubstrate

STEP = None  # [mm] step size used to sample the optical surfaces of the substrates

# Useful area defined in the (x, y) plane of the EGA coordinate system, i.e. in the middle of the two halves

# LW definitions

lw_useful_area = PieAperture(34.2 / 2, 90, -90)  # useful area is a half circle

# From Zemax definitions in EUVST_v20230909.zmx
# Description of the design in RSC-2022021C_SOLAR-C(EUVST)_Optical_Design_Summary.pdf
lw_surface = EllipticalGrating(1 / 1008.9554166, 1 / 1010.2811818, 1933.255026,
                               dx=0, dy=2.0386372, dz=0,
                               alpha=0.346498, beta=0.7733422, gamma=0, degrees=True)
# Aperture is modeled as a rectangle
lw_aperture = RectangularAperture(17.4, 34.8, dx=-17.4 / 2 - 0.15 / 2)
lw_substrate = EGASubstrate(lw_surface,
                            lw_aperture,
                            lw_useful_area,
                            name='LW',
                            x_grid_step=STEP)
# Prototype rectangular substrate with spherical surface
spherical_lw_aperture = RectangularAperture(61.925, 86.188, dx=-61.925 / 2 - 0.15 / 2 + 20)
spherical_lw_substrate = EGASubstrate(lw_substrate.best_sphere,
                                      copy.deepcopy(spherical_lw_aperture),
                                      name='Spherical LW',
                                      x_grid_step=STEP)
# Rectangular substrate pre-cutting to octagon
rectangular_lw_substrate = EGASubstrate(copy.deepcopy(lw_substrate.surface),
                                        copy.deepcopy(spherical_lw_aperture),
                                        name='Rectangular LW',
                                        x_grid_step=STEP,
                                        fiducials=((17.925, 41.094), (-40, 0), (17.925, -41.094)))

dx, dy = rectangular_lw_substrate.limits[1] - 12.845, 41.960 - rectangular_lw_substrate.limits[3]
surface = Sphere(527.97, dx, dy)
surface.dz += 20.14 - 20 + surface.sag((dx, dy))
bertin_lw1_spherical = EGASubstrate(surface,
                                    copy.deepcopy(spherical_lw_aperture),
                                    name='LW1',
                                    x_grid_step=STEP)

# SW definitions

sw_useful_area = PieAperture(34.2 / 2, -90, 90)

sw_surface = Standard(516.0274, -0.522870,
                      dx=31.8380922, dy=15.0690780, dz=2.446336,
                      alpha=3.1515375, beta=-5.0875302, gamma=0, degrees=True)
sw_aperture = RectangularAperture(17.4, 34.8, dx=17.4 / 2 + 0.15 / 2)
sw_substrate = EGASubstrate(sw_surface,
                            sw_aperture,
                            sw_useful_area,
                            name='SW',
                            x_grid_step=STEP)
# Dummy substrate with spherical surface
spherical_sw_aperture = RectangularAperture(61.925, 86.188, dx=61.925 / 2 + 0.15 / 2 - 20)
spherical_sw_substrate = EGASubstrate(sw_substrate.best_sphere,
                                      copy.deepcopy(spherical_sw_aperture),
                                      name='Spherical SW',
                                      x_grid_step=STEP)
# Rectangular substrate pre-cutting to octagon
rectangular_sw_substrate = EGASubstrate(copy.deepcopy(sw_substrate.surface),
                                        copy.deepcopy(spherical_sw_aperture),
                                        name='Rectangular SW',
                                        x_grid_step=STEP,
                                        fiducials=((-17.925, 41.094), (40, 0), (-17.925, -41.094)))

dx, dy = rectangular_sw_substrate.limits[0] + 5.773, 29.840 - rectangular_sw_substrate.limits[3]
surface = Sphere(518.59, dx, dy)
surface.dz += 19.99 - 20 + surface.sag((dx, dy))
bertin_sw3_spherical = EGASubstrate(surface,
                                    copy.deepcopy(spherical_sw_aperture),
                                    name='SW3',
                                    x_grid_step=STEP)

bertin_lw_sphericals = bertin_lw1_spherical,
bertin_sw_sphericals = bertin_sw3_spherical,

# Measured angles between mirrors of the rotation stage
# See mirror_crown_measurements.xlsx
mirror_crown_angles = [0, 022.5405, 45.022, 67.5556,
                       90.0603, 112.4738, 134.9973, 157.5418,
                       179.9889, 202.5545, 224.9925, 247.5120,
                       269.9254, 292.5007, 315.0222, 337.5255]
