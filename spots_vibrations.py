import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from astropy.table import Table
from utils.fitting import gauss_2dfit

from scipy import optimize
from scipy.signal import correlate2d


def model(x, b, a):
    return b * x + a

def parabolic(cc):
    cy, cx = np.unravel_index(np.argmax(cc, axis=None), cc.shape)
    if cx == 0 or cy == 0 or cx == cc.shape[1] - 1 or cy == cc.shape[0] - 1:
        return cx, cy
    else:
        xi = [cx - 1, cx, cx + 1]
        yi = [cy - 1, cy, cy + 1]
        ccx2 = cc[[cy, cy, cy], xi] ** 2
        ccy2 = cc[yi, [cx, cx, cx]] ** 2

        xn = ccx2[2] - ccx2[1]
        xd = ccx2[0] - 2 * ccx2[1] + ccx2[2]
        yn = ccy2[2] - ccy2[1]
        yd = ccy2[0] - 2 * ccy2[1] + ccy2[2]

        if xd != 0:
            dx = xi[2] - (xn / xd + 0.5)
        else:
            dx = cx
        if yd != 0:
            dy = yi[2] - (yn / yd + 0.5)
        else:
            dy = cy

        return dx, dy

data = fits.getdata(r"C:\Users\fauchere\Desktop\SolarC\tests\2023-11-17-1906_4-U-G-Jup.fit")

yc, xc = np.unravel_index(np.argmax(data[0]), data[0].shape)
half_width = 21

x = []
y = []
for frame in tqdm(data):
    # cc = correlate2d(frame,
    #                        data[0, yc - half_width:yc + half_width, xc - half_width:xc + half_width])
    cc = cv2.matchTemplate(frame,
                           data[0, yc - half_width:yc + half_width, xc - half_width:xc + half_width],
                           cv2.TM_CCOEFF)
    dx, dy = parabolic(cc)
    # x0, y0 = np.unravel_index(np.argmax(frame), frame.shape)
    # guess = [frame.max(), x0, y0, 10, 10, 0, 0]
    # p, _ = gauss_2dfit(frame, guess)
    # dx, dy = p[1], p[2]
    x.append(dx)
    y.append(dy)

n = len(x)

dt = 120 / n
time = np.linspace(0, (len(x) - 1) * dt, len(x))
p, _ = optimize.curve_fit(model, time, x, p0=[1, 0])
x_fit = model(time, *p)
x -= x_fit

p, _ = optimize.curve_fit(model, time, y, p0=[1, 0])
y_fit = model(time, *p)
y -= y_fit

# table = Table([time, x, y], names=['Time (seconds)', 'X (pixels)', 'Y (pixels)'])
# table.write('ega_shifts.csv', format='csv', overwrite=True)

x = np.array(x)
x -= x.mean()
y = np.array(y)
y -= y.mean()

fig, ax = plt.subplots(2, 1)
ax[0].plot(time, x, label='x')
ax[0].plot(time, y, label='y')
ax[0].set_xlabel('Time (seconds)')
ax[0].set_ylabel('Pixel')
ax[0].legend()

x_std = x.std()
normed_x = x / x_std
y_std = y.std()
normed_y = y / y_std

frequencies = np.fft.fftfreq(n, dt)
x_power = n * abs(np.fft.fft(normed_x, norm='forward')) ** 2
y_power = n * abs(np.fft.fft(normed_y, norm='forward')) ** 2
ax[1].plot(frequencies[1:n // 2], x_power[1:n // 2], label='x')
ax[1].plot(frequencies[1:n // 2], y_power[1:n // 2], label='y')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_ylim(1e-4, 1e3)
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Power')
ax[1].legend()

plt.tight_layout()

cutoff = 0.1  # Hz

print(f'R RMS {np.std(np.sqrt(x ** 2 + y ** 2))} pixels')
print(f'X RMS {x_std} pixels')
print(f'Y RMS {y_std} pixels')
filtered = np.where(frequencies > cutoff)[0]
print(f'Filtered X RMS {np.sum(x_std * x_power[filtered] / n)} pixels')
print(f'Filtered Y RMS {np.sum(y_std * y_power[filtered] / n)} pixels')
