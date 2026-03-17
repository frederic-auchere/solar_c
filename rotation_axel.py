import os.path
import numpy as np
import h5py
import inspect
from factory import Parameterized, Parameter
from optics import surfaces
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import sigma_clip
from astropy.io import fits
import warnings
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator

from astropy.io import fits
import matplotlib.pyplot as plt
from optical import rectangular_sw_substrate
from optics.zygo import SagData
import os
from optics import surfaces
import numpy as np

files_zygo = '/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin.fits'
dx_zygo = 200
dy_zygo=432

zygo_data = SagData(files_zygo,dx=dx_zygo, dy=dy_zygo, theta=0, binning=1, auto_crop=True)
measured_surface = surfaces.MeasuredSurface(zygo_data, alpha=0, beta=0, gamma=0) #est ce que on laisse ces angles ? self.tip_tilt_to_normal ?

measured_substrate = surfaces.Substrate(measured_surface,rectangular_sw_substrate.aperture,rectangular_sw_substrate.useful_area)


#grille nominal surface
valid_only = True
ox, oy = rectangular_sw_substrate.grid()

xy=None
valid = np.logical_and(~rectangular_sw_substrate.sag(xy).mask, ~np.isnan(rectangular_sw_substrate.sag(xy).data)) if valid_only else Ellipsis
x, y = ox, oy

xyz = np.stack((x.ravel(), y.ravel(), np.zeros(x.size), np.ones(x.size)))

x, y, _= measured_substrate.surface.rotation_matrix() @ xyz #shape 3xN
sag_in_normal=measured_substrate.sag((x,y)) #x et y tournés depuis EGA-O vers normal et sag calculé en ces points là
plt.imshow(sag_in_normal.reshape(ox.shape), origin = 'lower', extent = measured_substrate.aperture.limits)
valid_mask = np.isfinite(sag_in_normal)
x_valid = x[valid_mask]
y_valid = y[valid_mask]
sag_valid = sag_in_normal[valid_mask]
print("Shape:", sag_valid.shape)
print("Nombre de NaN:", np.count_nonzero(np.isnan(sag_valid)))
xyz = np.stack((x_valid, y_valid, sag_valid, np.ones(x_valid.size))) #shape 4x10201

print(np.count_nonzero(np.isnan(sag_valid)))
print(np.min(sag_in_normal), np.max(sag_valid))

alpha_corr = np.deg2rad(0.0001)
beta_corr = np.deg2rad(10)
correction_tilt = Rotation.from_euler('xyz', (alpha_corr, beta_corr, 0)).as_matrix() #shape 3x3

xyz = np.asarray(xyz) #sinon probleme avec le masque
nx, ny, nz = correction_tilt @ xyz[:3, :] # shape 3x3 @ shape 3x10201 -> 3x10201
print(np.count_nonzero(np.isnan(nx)))
print(np.count_nonzero(np.isnan(ny)))
print(np.count_nonzero(np.isnan(nz)))


points = np.column_stack((nx.ravel(), ny.ravel()))
values = nz.ravel()


print("ox range:", ox.min(), ox.max())
print("oy range:", oy.min(), oy.max())

print("nx range:", np.nanmin(nx), np.nanmax(nx))
print("ny range:", np.nanmin(ny), np.nanmax(ny))
plt.scatter(nx.ravel(), ny.ravel(), c=nz.ravel(), s=5)
plt.scatter(ox.ravel(), oy.ravel(), marker='x', color='red')
plt.show()
sag_corrige_tilt_interpolated_substrate_grid = griddata(points, values, (ox,oy), method='cubic', rescale = False)


outfile='/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/SW/FM_SW_SN1/Zygo/Form/20260223/substrates_template_FM_form_casquette_Bertin_bascule_nouveau_code_test_nina.fits'
header = fits.header.Header()
header['GX'] = zygo_data.x_step
header['GY'] = zygo_data.y_step
header['DX'] = -rectangular_sw_substrate.aperture.limits[0] / header['GX']
header['DY'] = -rectangular_sw_substrate.aperture.limits[2] / header['GY']
header['THETA'] = 0

fits.writeto(outfile, sag_corrige_tilt_interpolated_substrate_grid, header=header, overwrite=True)