from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata
from astropy.io import fits
import matplotlib.pyplot as plt
from optical import rectangular_sw_substrate
from optics.zygo import SagData
from optics import surfaces
import numpy as np
from optics.geometry import NormalVector

files_zygo = r'Y:\02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin.fits'

zygo_data = SagData(files_zygo, theta=0, binning=1, auto_crop=True)
alpha, beta, gamma = rectangular_sw_substrate.tip_tilt_from_normal()
measured_surface = surfaces.MeasuredSurface(zygo_data, alpha=alpha, beta=beta, gamma=gamma)

measured_substrate = surfaces.Substrate(measured_surface, rectangular_sw_substrate.aperture, rectangular_sw_substrate.useful_area)

grid_step = 0.5  # [mm]
nx = int((rectangular_sw_substrate.limits[1] - rectangular_sw_substrate.limits[0]) / grid_step)
ny = int((rectangular_sw_substrate.limits[3] - rectangular_sw_substrate.limits[2]) / grid_step)
x_min = rectangular_sw_substrate.limits[0]
x_max = x_min + grid_step * nx
y_min = rectangular_sw_substrate.limits[2]
y_max = y_min + grid_step * ny
ox, oy = np.meshgrid(np.linspace(x_min, x_max, nx + 1), np.linspace(y_min, y_max, ny + 1))

nominal_sag = rectangular_sw_substrate.sag((ox, oy)).data

xyz = np.stack((ox.ravel(), oy.ravel(), nominal_sag.ravel(), np.ones(ox.size)))

x, y, z = measured_substrate.surface.rotation_matrix() @ xyz
z += measured_substrate.sag((x,y)).data / 1e6

xyz = np.stack((x, y, z, np.ones(x.size))) #shape 4x10201


dx = 0.173  # écart point violet mesuré
dy = -0.03
p = np.sqrt(dx ** 2 + dy **2)
dz = p * np.tan(np.acos(p / rectangular_sw_substrate.best_sphere.r))
normal_vector = NormalVector(dx, dy, dz)
delta_alpha = -np.arctan2(normal_vector.y, normal_vector.z)
delta_beta = np.arctan2(normal_vector.x, np.sqrt(normal_vector.y ** 2 + normal_vector.z ** 2))
print(60 * np.degrees(delta_alpha), 60 * np.degrees(delta_beta))

correction_tilt = Rotation.from_euler('xyz', (-delta_alpha, -delta_beta, 0)).as_matrix() #shape 3x3
translation = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
correction_tilt = correction_tilt @ translation

nx, ny, nz = correction_tilt @ xyz  #[:3, :] # shape 3x3 @ shape 3x10201 -> 3x10201

xyz = np.stack((nx, ny, nz, np.ones(x.size))) #shape 4x10201

nx, ny, nz = measured_substrate.surface.inverse_rotation_matrix() @ xyz

valid = np.logical_and(np.isfinite(nx), np.isfinite(ny))
sag = griddata((nx[valid], ny[valid]), nz[valid], (ox, oy), method='cubic', rescale=False).reshape(ox.shape)

fig, axes = plt.subplots(1, 3)
im1 = axes[0].imshow(sag, origin='lower', extent=rectangular_sw_substrate.limits)
plt.colorbar(im1, ax=axes[0], label='z sag [mm]')
sag_difference = (sag - nominal_sag) * 1e6
im2 = axes[1].imshow(sag_difference, origin='lower', extent=rectangular_sw_substrate.limits, vmin=-20000, vmax=20000)
plt.colorbar(im2, ax=axes[1], label = r'z sag difference [nm]')
im3 = axes[2].imshow(measured_substrate.sag((ox, oy)) - sag_difference, origin='lower', extent=rectangular_sw_substrate.limits, vmin=-50, vmax=50)
plt.colorbar(im3, ax=axes[2], label = r'z sag difference difference [nm]')

outfile=r'Y:/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin_bascule_nouveau_code_test_nina.fits'
header = fits.header.Header()
header['GX'] = zygo_data.x_step
header['GY'] = zygo_data.y_step
header['DX'] = -rectangular_sw_substrate.aperture.limits[0] / header['GX']
header['DY'] = -rectangular_sw_substrate.aperture.limits[2] / header['GY']
header['THETA'] = 0

fits.writeto(outfile, nz, header=header, overwrite=True)