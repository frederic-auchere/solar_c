import numpy as np
from skimage.filters import window
import matplotlib.pyplot as plt
from optics.zygo import ZygoData

basepath = r"C:\Users\fauchere\Documents\01-Projects\02-Space\Solar C\EPSILON\Optics\Substrates\Bertin"
file = r"\LW_SN1\LW_SN1_2_ZU_raw.datx"

zygo_data = ZygoData(basepath + file)
data = zygo_data.data[1]
data -= np.nanmean(data)
data *= 1e-6
variance = np.nanvar(data)
print(np.nanstd(data))
print(np.nanmax(data) - np.nanmean(data))
data = np.nan_to_num(data, 0)

win = window('hann', data.shape)

width = 1e6 * zygo_data.gx * data.shape[0]  # Convert to nm
norm = data.shape[0] * data.shape[1] * (width / data.shape[0])*(width / data.shape[1]) / variance

psd = norm * np.abs(np.fft.fft2(win * data))**2

nu = np.linspace(0, data.shape[1] // 2 - 1, data.shape[1] // 2) / width

spec = 2e10 * (1e6 * nu[1:]) ** -3  # specification given with nu expressed in mm-1

plt.step(nu[1:], psd[1:psd.shape[1] // 2, 0])
plt.plot(nu[1:], spec)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-8, 1e-5)
#plt.ylim(0.1, 1e10)
plt.xlabel('Spatial frequency [nm$^{-1}$]')
plt.ylabel('PSD [nm$^4$]')
