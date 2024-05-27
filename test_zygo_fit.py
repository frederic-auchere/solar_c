from optics.zygo import SagData
from optics.surfaces import MeasuredSurface, Substrate, CircularAperture, StandardSubSphere
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm


files= [r"data\HRI-P2_022.5d_174744_002.asc"]
R = 1518.1
gx = ((R + 3.54350) / R) * 0.078797149
gy = ((R + 3.54350) / R) * 0.078717716
aperture = CircularAperture(66)
useful_area = CircularAperture(54)
off_axis = 80.0
theta = 22.5

# files = [r"C:\Users\fauchere\Desktop\Zygo\HRIP3\20231123\HRIP3_20231123-103645_270d_036.asc"]
# R = 1518.1
# gx = 66 / 147.0
# gy = 66 / 147.0
# aperture = CircularAperture(66.0)
# useful_area = CircularAperture(54.0)
# off_axis = 80.0
# theta = 0.0

# files = glob.glob(r"C:\Users\fauchere\Desktop\Zygo\Benoit_200\*.asc")[0:1]
# # files = [r"C:\Users\fauchere\Desktop\Zygo\Benoit_200\Benoit_20231123-150628001_.asc"]
# R = 2478.0
# gx = 207 / 266.0
# gy = 207 / 266.0
# aperture = CircularAperture(207)
# useful_area = CircularAperture(200)
# off_axis = 0
# theta = 0

average_residual = None

for file in tqdm(files):

    sag_data = SagData(file, gx=gx, gy=gy, binning=1, theta=theta)

    measured_surface = MeasuredSurface(sag_data)

    substrate = Substrate(measured_surface, aperture, useful_area)
    initial_parameters = [R, -1, 0, off_axis, R, 0, off_axis, 0]
    no_bounds = (None, None)
    bounds = [(R, R), (-1, -1), no_bounds, no_bounds, no_bounds, no_bounds, no_bounds, (0, 0)]
    best_surface = substrate.find_best_surface(StandardSubSphere,
                                               initial_parameters=initial_parameters, bounds=bounds, objective='std')
    residual = 1e6*(substrate.sag() - best_surface.sag(substrate.grid()))
    residual -= np.nanmean(residual)

    if average_residual is None:
        average_residual = residual
    else:
        average_residual += residual

average_residual /= len(files)

print(best_surface)

wavelength = 632.8
pv = np.nanmax(residual) - np.nanmin(residual)
print(f'PV: {pv} nm - $\lambda${wavelength / pv}')
std = np.nanstd(residual)
print(f'RMS: {std} nm - $\lambda${wavelength / std}')
fig, ax = plt.subplots(1, 1)
im = ax.imshow(
    residual,
    origin='lower',
    extent=[substrate.grid()[0].min(),
            substrate.grid()[0].max(),
            substrate.grid()[1].min(),
            substrate.grid()[1].max()]
)
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
cbar = fig.colorbar(im, label='Difference from best parabola [nm]')
