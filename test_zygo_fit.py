from optics.zygo import SagData
from optics.surfaces import MeasuredSurface, Substrate, CircularAperture, StandardSubSphere
import matplotlib.pyplot as plt

file = r"C:\Users\fauchere\Desktop\HRI-P2_022.5d_174744_002.asc"
# file = r"C:\Users\fauchere\Desktop\HRI-P2_000.0d_145220_002.asc"

gx = ((1518.1 + 3.54350)/1518.1)*0.078797149
gy = ((1518.1 + 3.54350)/1518.1)*0.078717716
sag_data = SagData(file, gx=gx, gy=gy, theta=22.5, binning=1)

measured_surface = MeasuredSurface(*sag_data.grid, sag_data.sag)
aperture = CircularAperture(66)
useful_area = CircularAperture(54)

substrate = Substrate(measured_surface, aperture, useful_area)
initial_parameters = [1518.1253, -1, 0, 80, 1518.1253, 0, 80, 0]
no_bounds = (None, None)
bounds = [(1518.1253, 1518.1253), (-1, -1), no_bounds, no_bounds, no_bounds, no_bounds, no_bounds, (0, 0)]
best_surface = substrate.find_best_surface(StandardSubSphere,
                                           initial_parameters=initial_parameters, bounds=bounds, objective='std')
print(best_surface)
residual = 1e6*(substrate.sag() - best_surface.sag(substrate.grid()))
residual -= residual.mean()
plt.imshow(residual, origin='lower')
plt.colorbar()
