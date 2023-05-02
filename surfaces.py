import numpy as np


class Surface:
    def __init__(self):
        pass

    def sag(self, x, y):
        raise NotImplementedError


class Toroidal(Surface):
    def __init__(self, rc, rr):
        super().__init__()
        self.rc = rc
        self.c = 1/rc
        self.rr = rr

    def sag(self, x, y):
        y2 = y**2
        zy = y2 * self.c / (1 + np.sqrt(1 - (1 + self.k) * y2 * self.c**2))
        zx = self.rr - zy - np.sqrt((self.rr - zy)**2 - x**2)
        return zx + zy


class EllipticalGrating(Surface):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def sag(self, x, y):
        u2 = (self.a * x)**2 + (self.b * x)**2
        return u2 * self.c / (1 + np.sqrt(1 - u2 * self.c ** 2))


class Standard(Surface):
    def __init__(self, r, k):
        super().__init__()
        self.r = r
        self.c = 1/r
        self.k = k

    def sag(self, x, y):
        r2 = x**2 + y**2
        return r2 * self.c / (1 + np.sqrt(1 - (1 + self.k) * r2 * self.c**2))
