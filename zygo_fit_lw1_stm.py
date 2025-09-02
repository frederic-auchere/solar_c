import glob
import os
from optical import spherical_lw_substrate
from optical.zygo import EGAFit
from optics.zygo import SagData
from optics.surfaces import MeasuredSurface, Substrate, Sphere
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from time import time

path = r'C:\Users\fauchere\Desktop\LW1_STM\20250619\000_degrees'
files = glob.glob(os.path.join(path, '*.datx'))
best_surfaces = []

for file in tqdm(files[0:1]):

    sag_data = SagData(file, gx=0.07, binning=1, theta=0)
    measured_surface = MeasuredSurface(sag_data)

    t0 = time()
    initial_surface = Sphere(528)
    fitted_parameters = []

    fitter = EGAFit(sag_data, spherical_lw_substrate, fitted_parameters, Sphere(528), floating_reference=True, objective='std')
    best_surface = fitter.fit()
    best_surfaces.append(best_surface)
    print(time() - t0)

print(best_surface[0])
residual = best_surface[0].residuals
residual -= np.nanmean(residual)
print(np.nanstd(residual))
print(632 / np.nanstd(residual))
print(632 / (np.nanmax(residual) - np.nanmin(residual)))

fig, axes = plt.subplots(1, 2)
im = axes[0].imshow(sag_data.sag, origin='lower')
plt.colorbar(im)
im = axes[1].imshow(residual, origin='lower')
plt.colorbar(im)
