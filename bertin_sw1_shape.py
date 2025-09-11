import os
from optical import lw_substrate
from optics.zygo import SagData
from optical.zygo import EGAFit
from optics import surfaces
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

base_path = r"Y:\02- Engineering\08 - Metrology\01 - Optics\08 - Bertin\LW_SN1"
zygo_file = "LW-SN1_Zygo_mes_1.datx"
nanomefos_file = "LW_SN1_2_ZU_wotiltnina2.datx"

gx = 0.207
gy = gx
dx = 14 + (61.925 - 19.925) / gx
dy = 400 - 86.188 / 2 / gy
sag_data = SagData(os.path.join(base_path, zygo_file), gx=gx, dx=dx, dy=dy, theta=0, binning=1, auto_crop=True)

nanomefos_data = SagData(os.path.join(base_path, nanomefos_file), dx=60, dy=60, theta=0, binning=1, auto_crop=True)

measured_surface = surfaces.MeasuredSurface(nanomefos_data, alpha=0, beta=0, gamma=0)
measured_substrate = surfaces.Substrate(measured_surface,
                                        lw_substrate.aperture,
                                        lw_substrate.useful_area)
surface = deepcopy(lw_substrate.surface)
surface.alpha += 0.0002031
surface.beta += 0.0002002
surface.a = 1 / (1 / surface.a + 0.025)
# surface.dx += 0.1247
# surface.dy -= 0.1075
# surface.c -= 0.005
residual = measured_substrate.sag() + lw_substrate.surface.sag(measured_substrate.grid()) - surface.sag(measured_substrate.grid())
residual *= 1e6
mini, maxi = np.nanpercentile(residual.data, (1, 99))
valid = (residual.data > mini) & (residual.data < maxi)
residual -= np.nanmedian(residual[valid])
print(np.nanstd(residual[valid]))
plt.imshow(residual, origin='lower', vmin=-25, vmax=25)


# fitted_parameters = ['dx', 'dy']
#
# print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
# fitter = EGAFit(sag_data, lw_substrate, fitted_parameters, lw_substrate.best_sphere,
#                 floating_reference=True, tol=1e-11, objective='std', method="Powell")
#
# fit = fitter.fit()
#
# fitter.make_report(outfile='LW-SN1_bertin.pdf')
# print(lw_substrate.surface)
# print(fit[0].best_surface.surface1)
# residual = fit[0].residuals
# mini, maxi = np.nanpercentile(residual.data, (1, 99))
# valid = (residual > mini) & (residual < maxi)
# residual -= np.nanmean(residual[valid])
# print(np.nanstd(residual[valid]))
# #print(np.nanmax(residual[valid]) - np.nanmin(residual[valid]))
# print(1 / fit[0].best_surface.surface1.a)
# print(1 / lw_substrate.surface.a)
#
# da = 1 / lw_substrate.surface.a - 1 / fit[0].best_surface.surface1.a
# db = 1 / lw_substrate.surface.b - 1 / fit[0].best_surface.surface1.b
# dc = lw_substrate.surface.c - fit[0].best_surface.surface1.c
# dx = lw_substrate.surface.dx - fit[0].best_surface.surface1.dx
# dy = lw_substrate.surface.dy - fit[0].best_surface.surface1.dy
# dz = lw_substrate.surface.dz - fit[0].best_surface.surface1.dz
# dalpha = lw_substrate.surface.alpha - fit[0].best_surface.surface1.alpha
# dbeta = lw_substrate.surface.beta - fit[0].best_surface.surface1.beta
# dgamma = lw_substrate.surface.gamma - fit[0].best_surface.surface1.gamma
# print(
#     f'da={da:.3f} [mm] db={db:.3f} [mm] dc={dc:.3f} [mm] ' + \
#     f'dx={dx:.3f} [mm] dy={dy:.3f} [mm] dz={dz:.3f} [mm] ' + \
#     f'dalpha={np.degrees(dalpha):.4f} [°] ' + \
#     f'dbeta={np.degrees(dbeta):.4f} [°] ' + \
#     f'dgamma={np.degrees(dgamma):.4f} [°]'
# )
#
# plt.figure()
# plt.imshow(residual, vmin=-25, vmax=25, origin='lower')
