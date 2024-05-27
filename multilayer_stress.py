import numpy as np

sigma = -0.5e9, -1e9  # Pa
nu_s = 0.17
E = 7.06e10  # Pa
es = 0.02    # m
r = 0.023    # m
ef = 75e-9  # m

dh = [s / ( (E / (1 - nu_s) ) * (1 / (3 * ef) ) * (es / r) ** 2 ) for s in sigma]

print(dh)

R = 0.522  # m
f = ( R - np.sqrt(R ** 2 - r ** 2) )
Rp = [( (f + d) ** 2 + r ** 2) / (2 * (f + d) ) for d in dh]

print(Rp)