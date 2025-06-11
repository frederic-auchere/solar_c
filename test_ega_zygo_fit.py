from astropy.io import fits
from optical.zygo import SagData, EGAFit
from optical import rectangular_sw_substrate
from optics.surfaces import Substrate, Sphere, MeasuredSurface, make_sub_parametric
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

file = 'simulated_sw_sag.fit'
substrate = rectangular_sw_substrate
# useful_area = RectangularAperture(55, 80, substrate.aperture.dx, substrate.aperture.dy)
# substrate.useful_area = useful_area
# useful_area = RectangularAperture(substrate.aperture.x_width, substrate.aperture.y_width, substrate.aperture.dx, substrate.aperture.dy)
# substrate = Substrate(substrate.surface,
#                       substrate.aperture, useful_area, x_grid_step=0.25)
# surface = substrate.surface
# substrate = Substrate(surface, aperture)

# aperture = RectangularAperture(61.925, 86.188, dx=0)
# # useful_area = RectangularAperture(aperture.x_width - 20*substrate.x_grid_step, aperture.y_width - 20*substrate.y_grid_step, dx=0)
# useful_area = RectangularAperture(50, 70, dx=0)
# surface = Standard(520, -1)
# # surface.alpha = np.radians(5)
# surface.dx = 0
# surface.dy = 0
# surface.dz = 0
# substrate = Substrate(surface, aperture, useful_area)

# bs = substrate.best_sphere
#
# uts = Substrate(substrate.surface, substrate.aperture)
# ubs = uts.find_best_surface(Sphere).best_surface
# sag = uts.sag().data - ubs.sag(uts.grid()).data
#
# tbs = ubs.tilt_to(substrate.surface)
# ts = uts.surface.tilt_to(substrate.surface)
# tsag = ts.sag(substrate.grid()) - tbs.sag(substrate.grid())
# plt.imshow(1e6 * (sag - tsag), origin='lower')
# plt.colorbar()

#
ox, oy = substrate.grid()  # assumed to be in the normal ref
xyz = np.stack((ox.ravel(), oy.ravel(), np.zeros(ox.size), np.ones(ox.size)))
x, y, z = substrate.matrix_from_normal() @ xyz  # rotate to substrate
x, y, _ = substrate.surface.rotation_matrix() @ np.stack((x, y, z, np.ones_like(x)))  # rotate to intrinsic
z = substrate.surface._zemax_sag(x, y)  # sag in intrinsic
x, y, z = substrate.surface.inverse_rotation_matrix() @ np.stack((x, y, z, np.ones(x.size)))  # rotate back to substrate
x, y, z = substrate.matrix_to_normal() @ np.stack((x, y, z, np.ones(x.size)))  # rotate back to normal
ox, oy = substrate.grid(limits=substrate.limits)  # assumed to be in the normal ref
oz = griddata((x, y), z, (ox, oy), method='cubic', rescale=False).reshape(ox.shape)  # interpolate at (ox, oy)
#
best_sphere = substrate.best_sphere
bsx, bsy, bsz = substrate.matrix_to_normal() @ np.array((best_sphere.dx, best_sphere.dy, best_sphere.dz + best_sphere.r, 1))
r = np.sqrt((ox - bsx) ** 2 + (oy - bsy) ** 2 + (oz - bsz) ** 2)
sag = (best_sphere.r - r).reshape(ox.shape)
# sag = substrate.sag_from(best_sphere, (ox, oy))
# sag[ox.shape[0] //2, :] = 0
# sag[:, ox.shape[1] //2] = 0

sag2 = substrate.sag() - best_sphere.sag(substrate.grid())

header = fits.Header()
header['WAVELNTH'] = 1e3 * 632.8e-9
header['GX'] = ox[0, 1] - ox[0, 0]
header['GY'] = oy[1, 0] - oy[0, 0]
fits.writeto(file, sag, header=header, overwrite=True)
# dx=31.725 [mm] dy=14.751 0
sag_data = SagData(file,
                   dx=(0 - substrate.limits[0]) / header['GX'],
                   dy=(0 - substrate.limits[2]) / header['GY'],
                   theta=0,
                   binning=1,
                   auto_crop=False)

alpha, beta, gamma = substrate.tip_tilt_from_normal()
measured_surface = MeasuredSurface(sag_data, alpha=alpha, beta=beta, gamma=gamma)
measured_substrate = Substrate(measured_surface, substrate.aperture, substrate.useful_area)
sub_class = make_sub_parametric(substrate.surface.__class__, Sphere)
initial_parameters = list(substrate.surface.parameters.values()), list(best_sphere.parameters.values())
nsag = 1e6 * measured_substrate.sag_from(sub_class(*initial_parameters))
plt.imshow(nsag, origin='lower', vmin=-10, vmax=10)
plt.colorbar()
print(np.nanstd(nsag))

# substrate.surface.alpha = 0
# substrate.surface.beta = 0
measurement_substrate = Substrate(substrate.surface, substrate.aperture, substrate.useful_area)
fitted_parameters = ['dx', 'dy']

fitter = EGAFit(sag_data, measurement_substrate, fitted_parameters, best_sphere,
                floating_reference=True, tol=1e-9, objective='std', method='powell'
                )

fit = fitter.fit()
print(fit[0].best_surface.surface1)
# print(fit[0].best_surface.surface1.tilt_to(substrate.surface))
print(fit[0].rms)
print(substrate.surface)
#
# fitter.make_report()
