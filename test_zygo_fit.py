from optics.zygo import read_asc
from optics.surfaces import MeasuredSurface, Substrate, CircularAperture, StandardSubSphere

file = r"C:\Users\fauchere\Desktop\HRI-P3_000.0d_135709_001.asc"

_, sag = read_asc(file)

measured_surface = MeasuredSurface(x, y, sag)
aperture = CircularAperture(66)
useful_area = CircularAperture(54)

substrate = Substrate(measured_surface, aperture, useful_area)
substrate.find_best_surface(StandardSubSphere)
