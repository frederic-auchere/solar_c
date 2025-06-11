import numpy as np
from numpy.random import random_sample, standard_normal


def make_random_time_series_ifft(n, dt, params):
    freq = np.fft.fftfreq(n, d=dt)

    # freq = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (dt*n)   if n is even
    # freq = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (dt*n)   if n is odd

    nfreq = len(freq)

    freq2 = freq ** 2

    # call_procedure, func, freq, params, pow
    alpha = params

    powe = np.zeros_like(freq2)
    powe[1:] = freq2[1:] ** (alpha / 2)  # /2 because freq2 is the square of f

    comp = np.zeros(nfreq, dtype='complex')

    if (n % 2 == 0):
        # Even case :
        # freq = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (dt*n)
        comp[1:(n // 2)] = np.sqrt(0.5) * (standard_normal(size=n // 2 - 1) + standard_normal(size=n // 2 - 1) * 1j)
        comp[(n // 2) + 1:] = np.conjugate(comp[1:(n // 2)])[::-1]
        comp[(n // 2)] = np.sqrt(2) * standard_normal(size=1)
    else:
        # Odd case :
        # freq = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (dt*n)
        comp[1:((n + 1) // 2)] = np.sqrt(0.5) * (
                    standard_normal(size=(n - 1) // 2) + standard_normal(size=(n - 1) // 2) * 1j)
        comp[((n + 1) // 2):] = np.conjugate(comp[1:((n + 1) // 2)])[::-1]

    fft_coeffs = np.sqrt(powe) * comp

    sst = np.fft.ifft(fft_coeffs)

    return (np.real(n * sst))  # Take the real part, but by construction the complex has a null imaginary part


def make_random_2d_power_law_ifft(n, dt, params):
    freq = np.fft.fftfreq(n, d=dt)

    # freq = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (dt*n)   if n is even
    # freq = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (dt*n)   if n is odd

    nfreq = len(freq)

    fx, fy = np.meshgrid(freq, freq, sparse=True)
    r_freq = np.sqrt(fx ** 2 + fy ** 2)

    # call_procedure, func, freq, params, pow
    alphaxy, alphat = params

    powe = np.zeros_like(r_freq)
    mask_notnul = r_freq > 0
    powe[mask_notnul] = r_freq[mask_notnul] ** alphaxy

    comp = np.zeros((nfreq, nfreq), dtype='complex')

    if n % 2 == 0:
        # Even case :
        # freq = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (dt*n)

        # Trick : use the code of the odd case to build the inner right shifted matrix
        # And then manage the Nyquist freq
        inner_comp = comp[1:n, 1:n]
        m = n - 1

        inner_comp[0:(m - 1) // 2, (m - 1) // 2] = np.sqrt(0.5) * (
                   standard_normal(size=(m - 1) // 2) + standard_normal(size=(m - 1) // 2) * 1j)
        inner_comp[(m - 1) // 2, (m - 1) // 2] = 0
        inner_comp[(m + 1) // 2:, (m - 1) // 2] = np.conjugate(inner_comp[0:(m - 1) // 2, (m - 1) // 2])[::-1]

        # Now draw column by column:
        for i in range((m + 1) // 2, m):
            inner_comp[:, i] = np.sqrt(0.5) * (standard_normal(size=m) + standard_normal(size=m) * 1j)
            inner_comp[:, m - 1 - i] = np.conjugate(inner_comp[:, i])[::-1]

        # Now the Nyquist freq on both axes:
        comp[0, 1:n // 2] = np.sqrt(0.5) * (standard_normal(size=n // 2 - 1) + standard_normal(size=n // 2 - 1) * 1j)
        comp[1:n // 2, 0] = np.sqrt(0.5) * (standard_normal(size=n // 2 - 1) + standard_normal(size=n // 2 - 1) * 1j)
        comp[0, (n // 2 + 1):] = np.conjugate(comp[0, 1:n // 2])[::-1]
        comp[n // 2 + 1:, 0] = np.conjugate(comp[1:n // 2, 0])[::-1]
        comp[0, n // 2] = np.sqrt(0.5) * standard_normal(size=1)
        comp[n // 2, 0] = np.sqrt(0.5) * standard_normal(size=1)
        comp[0, 0] = np.sqrt(2) * standard_normal(size=1)

    else:
        # Odd case :
        # freq = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (dt*n)
        # Work on shifted coeff is easier, comp is the shifted matrix here :

        # First draw the fx=0 coefficients, they are in the middle (because coeff are shifted here) :
        comp[0:(n - 1) // 2, (n - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=(n - 1) // 2) + standard_normal(size=(n - 1) // 2) * 1j)
        comp[(n - 1) // 2, (n - 1) // 2] = 0
        comp[(n + 1) // 2:, (n - 1) // 2] = np.conjugate(comp[0:((n - 1) // 2), (n - 1) // 2])[::-1]

        # Now draw column by column:
        for i in range((n + 1) // 2, n):
            comp[:, i] = np.sqrt(0.5) * (standard_normal(size=n) + standard_normal(size=n) * 1j)
            comp[:, n - 1 - i] = np.conjugate(comp[:, i])[::-1]

    # print (np.array_repr(comp, max_line_width=200))

    fft_coeffs = np.sqrt(powe) * np.fft.ifftshift(comp)

    # print (np.array_repr(np.fft.ifftshift(comp), max_line_width=200))

    sst = np.fft.ifft2(fft_coeffs)

    return np.real(n * n * sst)  # Take the real part, but by construction the complex has a null imaginary part


def make_random_3d_power_law(nxy, nt, dxy, dt, params):
    # WARNING : this version is here for historical purpose, but should not be used
    # because it draws too many random coefficients and returns a complex cube (not real only)

    freqx = np.fft.fftfreq(nxy, d=dxy)
    freqy = np.fft.fftfreq(nxy, d=dxy)
    freqt = np.fft.fftfreq(nt, d=dt)

    nfreqx = len(freqx)
    nfreqy = len(freqy)
    nfreqt = len(freqt)

    fx, fy, ft = np.meshgrid(freqx, freqy, freqt, sparse=True)
    freq2 = fx ** 2 + fy ** 2 + ft ** 2

    (alphaxy, alphat) = params
    powe = freq2 ** (alphaxy / 2)  # /2 because freq2 is the square of f

    # to be verified : sqrt(0.5) en 2D or 3D ??!!
    comp = np.sqrt(0.5) * (
                standard_normal(size=(nfreqx, nfreqy, nfreqt)) + standard_normal(size=(nfreqx, nfreqy, nfreqt)) * 1j)

    powe[0, 0, 0] = 0

    fft_coeffs = np.sqrt(powe) * comp

    sst = np.fft.ifftn(fft_coeffs)

    return (nxy * nxy * nt ** sst)  # *2 for symmetric part of FFT


def make_random_3d_power_law_ifft(nxyt, dxyt, params):
    freqx = np.fft.fftfreq(nxyt, d=dxyt)
    freqy = np.fft.fftfreq(nxyt, d=dxyt)
    freqt = np.fft.fftfreq(nxyt, d=dxyt)

    nfreqx = len(freqx)
    nfreqy = len(freqy)
    nfreqt = len(freqt)
    nfreq = nfreqx
    n = nfreq

    fx, fy, ft = np.meshgrid(freqx, freqy, freqt, sparse=True)
    freq2 = fx ** 2 + fy ** 2 + ft ** 2

    (alphaxy, alphat) = params
    # powe = freq2**(alphaxy/2) # /2 because freq2 is the square of f
    # vari = total(pow, /double)/nfreq

    powe = np.zeros_like(freq2)
    mask_notnul = freq2 > 0
    powe[mask_notnul] = freq2[mask_notnul] ** (alphaxy / 2)  # /2 because freq2 is the square of f

    comp = np.zeros((nfreq, nfreq, nfreq), dtype='complex')

    # idee : remplir la moitie du cube (l'autre à 0)
    # puis remplir la symetrique avec
    # ms = ms+np.conjugate(ms[::-1,::-1,::-1])

    if (n % 2 == 0):
        # Even case :
        # freq = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (dt*n)

        # Trick : use the code of the odd case to build the inner right shifted matrix
        # And then manage the Nyquist freq
        inner_comp = comp[1:n, 1:n, 1:n]
        m = n - 1

        inner_comp[0:((m - 1) // 2), :, :] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2, m, m)) + standard_normal(size=((m - 1) // 2, m, m)) * 1j)
        # the slide comp[((m-1)//2),:,:] must be treated separatly
        inner_comp[((m - 1) // 2), 0:((m - 1) // 2), :] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2, m)) + standard_normal(size=((m - 1) // 2, m)) * 1j)
        inner_comp[((m - 1) // 2), ((m - 1) // 2), 0:((m - 1) // 2)] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)

        inner_comp += np.conjugate(inner_comp[::-1, ::-1, ::-1])

        inner_slide1 = comp[0, 1:n, 1:n]  # m x m slide with half coeff conjugate from symetric (m odd)
        inner_slide1[0:(m - 1) // 2, :] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2, m)) + standard_normal(size=((m - 1) // 2, m)) * 1j)
        inner_slide1[(m - 1) // 2, 0:(m - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)
        inner_slide1 += np.conjugate(inner_slide1[::-1, ::-1])
        inner_slide1[(m - 1) // 2, (m - 1) // 2] = np.sqrt(0.5) * standard_normal(size=(1))

        inner_slide2 = comp[1:n, 0, 1:n]
        inner_slide2[0:(m - 1) // 2, :] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2, m)) + standard_normal(size=((m - 1) // 2, m)) * 1j)
        inner_slide2[(m - 1) // 2, 0:(m - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)
        inner_slide2 += np.conjugate(inner_slide2[::-1, ::-1])
        inner_slide2[(m - 1) // 2, (m - 1) // 2] = np.sqrt(0.5) * standard_normal(size=(1))

        inner_slide3 = comp[1:n, 1:n, 0]
        inner_slide3[0:(m - 1) // 2, :] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2, m)) + standard_normal(size=((m - 1) // 2, m)) * 1j)
        inner_slide3[(m - 1) // 2, 0:(m - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)
        inner_slide3 += np.conjugate(inner_slide3[::-1, ::-1])
        inner_slide3[(m - 1) // 2, (m - 1) // 2] = np.sqrt(0.5) * standard_normal(size=(1))

        row1 = comp[0, 0, 1:n]
        row1[0:(m - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)
        row1 += np.conjugate(row1[::-1])
        row1[(m - 1) // 2] = np.sqrt(0.5) * standard_normal(size=(1))

        row2 = comp[1:n, 0, 0]
        row2[0:(m - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)
        row2 += np.conjugate(row2[::-1])
        row2[(m - 1) // 2] = np.sqrt(0.5) * standard_normal(size=(1))

        row3 = comp[0, 1:n, 0]
        row3[0:(m - 1) // 2] = np.sqrt(0.5) * (
                    standard_normal(size=((m - 1) // 2)) + standard_normal(size=((m - 1) // 2)) * 1j)
        row3 += np.conjugate(row3[::-1])
        row3[(m - 1) // 2] = np.sqrt(0.5) * standard_normal(size=(1))

        comp[0, 0, 0] = np.sqrt(0.5) * standard_normal(size=(1))

    else:
        # Odd case :
        # freq = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (dt*n)
        # Work on shifted coeff is easier, comp is the shifted matrix here :

        comp[0:((n - 1) // 2), :, :] = np.sqrt(0.5) * (
                    standard_normal(size=((n - 1) // 2, n, n)) + standard_normal(size=((n - 1) // 2, n, n)) * 1j)
        # the slide comp[((n-1)//2),:,:] must be treated separatly
        comp[((n - 1) // 2), 0:((n - 1) // 2), :] = np.sqrt(0.5) * (
                    standard_normal(size=((n - 1) // 2, n)) + standard_normal(size=((n - 1) // 2, n)) * 1j)
        comp[((n - 1) // 2), ((n - 1) // 2), 0:((n - 1) // 2)] = np.sqrt(0.5) * (
                    standard_normal(size=((n - 1) // 2)) + standard_normal(size=((n - 1) // 2)) * 1j)

        comp += np.conjugate(comp[::-1, ::-1, ::-1])

        # RQ comp[((n-1)//2),(n-1)//2,(n-1)//2]=0 by design... So that the mean of the signal is 0

    # print (np.array_repr(comp, max_line_width=200))

    fft_coeffs = np.sqrt(powe) * np.fft.ifftshift(comp)

    sst = np.fft.ifftn(fft_coeffs)

    return (np.real((n * n * n * sst)))  # Take the real part, but by construction the complex has a null imaginary part
