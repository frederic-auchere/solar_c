import os
import glob
from optical.zygo import SagData, EGAFit
from optics.surfaces import Substrate, CircularAperture, Standard, Sphere

path = r'C:\Users\fauchere\Documents\01-Projects\02-Space\Solar Orbiter\EUI\Design\HRI\Optics\Alignment\Interferometry\HRI-P\HRI-P2_20170125_IAS\Mesures'
files = glob.glob(os.path.join(path, '*.asc'))[::8]

gx = ((1518.1 + 3.54350) / 1518.1) * 0.078797149
gy = ((1518.1 + 3.54350) / 1518.1) * 0.078717716

sag_data = []
for file in files:
    theta = float(os.path.basename(file).split('_')[1][0:5])
    sag_data.append(
        SagData(file, gx=gx, gy=gy, theta=theta, binning=4, auto_crop=False)
    )

initial_surface = Standard(1518.1253, -1, 0, 80)
aperture = CircularAperture(33)
useful_area = CircularAperture(27)
substrate = Substrate(initial_surface, aperture, useful_area)

fitted_parameters = ['dx', 'dy']

fitter = EGAFit(sag_data, substrate, fitted_parameters, Sphere(1518, 0, 80),
                floating_reference=True, tol=1e-9, objective='std', method='powell')

# fitter = EGAFit.from_xlsx(r"C:\Users\fauchere\Documents\02-Programmes\Python\scripts\solar_c\sw_substrates_template.xlsx")
#
fit = fitter.fit()
print(fit[0].best_surface)
print(fit[0].rms)
#
fitter.make_report()
