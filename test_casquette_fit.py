import os
import numpy as np
import glob
from optics.zygo import SagData
from optical.zygo import EGAFit
from optics.surfaces import MeasuredSurface, Substrate, Sphere, Flat, RectangularAperture, CircularAperture
from tqdm import tqdm
import matplotlib.pyplot as plt


# path = r'C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics\Metrology\Zygo\Casquette\20241206_casquette_by_flipping\Sequence2'
#




#fitter = EGAFit.from_xlsx(r"/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/Old-avant_reprise_Bertin_Sept_2025/FM_SW_SN1_old/Zygo/Tilt/20250901/substrates_template_SW_SN2.xlsm")
fitter = EGAFit.from_xlsx(r"Y:\02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/FM/Old-avant_reprise_Bertin_Sept_2025/FM_SW_SN1_old/Zygo/Tilt/20250901/substrates_template_SW_SN2.xlsm")
# fitter = EGAFit.from_xlsx(r'/Volumes/solarc/02- Engineering/08 - Metrology/01 - Optics/07 - Measurements/STM/SW1_STM/Zygo/20250619/20250619_sw1_stm.xlsx')
print(fitter.sag_data)
print(fitter.substrate.surface)

print('XXXXXXXX')
fit = fitter.fit()
print(fit[0].best_surface)
print(fit[0].rms)
#
fitter.make_report()




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
