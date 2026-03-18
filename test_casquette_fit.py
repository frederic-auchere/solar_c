import numpy as np
from optical.zygo import EGAFit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# path = r'C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics\Metrology\Zygo\Casquette\20241206_casquette_by_flipping\Sequence2'
#




#fitter = EGAFit.from_xlsx(r"/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/Old-avant_reprise_Bertin_Sept_2025/FM_SW_SN1_old/Zygo/Tilt/20250901/substrates_template_SW_SN2.xlsm")
#fitter = EGAFit.from_xlsx(r"Y:\02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/Old-avant_reprise_Bertin_Sept_2025/FM_SW_SN1_old/Zygo/Tilt/20250901/substrates_template_SW_SN2.xlsm")
# fitter = EGAFit.from_xlsx(r'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/STM/SW1_STM/Zygo/20250619/20250619_sw1_stm.xlsx')
fitter = EGAFit.from_xlsx(r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN5\Zygo\Tilt\20251125\substrates_FM_casquette_SW_SN5_test.xlsm")
# fitter = EGAFit.from_xlsx(r"Y:\02- Engineering\08 - Metrology\01 - Optics\07 - Measurements\FM\SW\FM_SW_SN5\Zygo\Tilt\20251127\substrates_FM_casquette_SW_SN5.xlsm")

print(fitter.substrate.surface)

print('XXXXXXXX')
fit = fitter.fit()
# print(fit[0].best_surface)
# print(fit[0].rms)
#

xc = []
yc = []
for sag_data, result in zip(fitter.sag_data, fitter.result):
    if result is not None:
        x, y  = sag_data.to_data(result.best_surface.surface1.dx, result.best_surface.surface1.dy, raw=True)
        x_ega, y_ega = sag_data.to_data(0, 0, raw=True)
        xc.append(x - x_ega)
        yc.append(y - y_ega)

from optical.utils import fit_circle_to_points
cx, cy, r = fit_circle_to_points(xc, yc)

for sag_data, result in zip(fitter.sag_data, fitter.result):
    if result is not None:
        x_ega, y_ega = sag_data.to_data(0, 0, raw=True)
        dx, dy = sag_data.to_substrate(x_ega - cx, y_ega - cy, raw=True)
        result.best_surface.surface1.dx += dx
        result.best_surface.surface1.dy += dy

fitter.make_report()

print(cx * fitter.sag_data[0].x_step, cy * fitter.sag_data[0].y_step, r * fitter.sag_data[0].x_step)
print(np.sqrt(fitter.substrate.surface.dx ** 2 + fitter.substrate.surface.dy ** 2))

circle = Circle((cx, cy), r, fill=False, color='cyan', linewidth=2, linestyle='--')

fig, ax = plt.subplots()
ax.scatter(xc, yc)
ax.add_patch(circle)
ax.axis('equal')


# fig, axes = plt.subplots(2, 2, figsize=(20 / 2.54, 10 / 2.54))
# width = 3
#
# x_means, y_means = [], []
# for orientation, guess, ax in zip(['000_0', '180_0'], ((-16, 13), (11, -11)), axes):
#
#     files = glob.glob(os.path.join(path, orientation, '*.asc'))
#     best_surfaces = []
#
#     for file in tqdm(files):
#
#         sag_data = SagData(file, gx=0.084, binning=1, theta=0, auto_crop=True)
#         aperture = RectangularAperture(50, 50)
#         mask = CircularAperture(5, dx=guess[0], dy=guess[1])
#         initial_surface = Sphere(528, dx=guess[0], dy=guess[1])
#         substrate = Substrate(initial_surface, aperture, mask)
#
#         fitted_parameters = ['r', 'dx', 'dy']
#         fitter = EGAFit(sag_data, substrate, fitted_parameters, Flat(), floating_reference=False, objective='std', method='powell')
#         best_surface = fitter.fit()
#         best_surfaces.append(best_surface)
#
#     print(np.mean([b.surface1.r for b in best_surfaces]))
#     x_means.append(np.median([b.surface1.dx for b in best_surfaces]))
#     y_means.append(np.median([b.surface1.dy for b in best_surfaces]))
#     print(1e3 * np.std([b.surface1.dx for b in best_surfaces]))
#     print(1e3 * np.std([b.surface1.dy for b in best_surfaces]))
#
#     residual = 1e6 * (substrate.sag() - best_surface.sag(substrate.grid()))
#     residual -= np.nanmean(residual)
#
#     x_lims = x_means[-1] - width / 2, x_means[-1] + width / 2
#     y_lims = y_means[-1] - width / 2, y_means[-1] + width / 2
#
#     sag = (substrate.sag() - np.nanmean(substrate.sag())) * 1e6
#
#     for a, data in zip(ax, (sag, residual)):
#         im = a.imshow(data, origin='lower', extent=substrate.aperture.limits)
#         a.set_xlim(x_lims)
#         a.set_ylim(y_lims)
#         a.set_xlabel('[mm]')
#         a.set_ylabel('[mm]')
#         plt.colorbar(im)
#
# print(np.sqrt((x_means[1] - x_means[0]) ** 2 + (y_means[1] - y_means[0]) ** 2) / 2)
