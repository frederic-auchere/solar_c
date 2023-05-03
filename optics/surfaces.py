import numpy as np
from numpy import ma
from scipy.optimize import minimize


class Point:

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    @property
    def radius(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


def sphere_from_four_points(p1, p2, p3, p4):
    U = lambda a, b, c, d, e, f, g, h: (a.z - b.z)*(c.x*d.y - d.x*c.y) - (e.z - f.z)*(g.x*h.y - h.x*g.y)
    D = lambda x, y, a, b, c: a.__getattribute__(x)*(b.__getattribute__(y) - c.__getattribute__(y)) +\
                              b.__getattribute__(x)*(c.__getattribute__(y) - a.__getattribute__(y)) +\
                              c.__getattribute__(x)*(a.__getattribute__(y) - b.__getattribute__(y))
    E = lambda x, y: (r1*D(x, y, p2, p3, p4) - r2*D(x, y, p3, p4, p1) +
                      r3*D(x, y, p4, p1, p2) - r4*D(x, y, p1, p2, p3)) / uvw
    u = U(p1, p2, p3, p4, p2, p3, p4, p1)
    v = U(p3, p4, p1, p2, p4, p1, p2, p3)
    w = U(p1, p3, p4, p2, p2, p4, p1, p3)
    uvw = 2 * (u + v + w)
    if uvw == 0.0:
        raise ValueError('The points are coplanar')
    r1, r2, r3, r4 = p1.radius ** 2, p2.radius ** 2, p3.radius ** 2, p4.radius ** 2
    x0, y0, z0 = E('y', 'z'), E('z', 'x'), E('x', 'y')
    p = Point(p1.x - x0, p1.y - y0, p1.z - z0)
    return x0, y0, z0, p.radius


class Surface:
    def __init__(self, dx, dy, dz=0):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def sag(self, x, y):
        raise NotImplementedError


class Toroidal(Surface):
    def __init__(self, rc, rr, dx=0.0, dy=0.0, dz=0.0):
        super().__init__(dx, dy, dz)
        self.rc = rc
        self.rr = rr

    def sag(self, x, y):
        c = 1 / self.rc
        y2 = (y - self.dy) ** 2
        zy = y2 * c / (1 + np.sqrt(1 - y2 * c ** 2))
        zx = self.rr - zy - np.sqrt((self.rr - zy) ** 2 - (x - self.dx) ** 2)
        return zx + zy


class EllipticalGrating(Surface):
    def __init__(self, a, b, c, dx=0.0, dy=0.0, dz=0.0):
        super().__init__(dx, dy, dz)
        self.a = a
        self.b = b
        self.c = c

    def sag(self, x, y):
        u2 = (self.a * (x - self.dx)) ** 2 + (self.b * (y - self.dy)) ** 2
        return u2 * self.c / (1 + np.sqrt(1 - u2))


class Standard(Surface):
    def __init__(self, r, k, dx=0.0, dy=0.0, dz=0.0):
        super().__init__(dx, dy, dz)
        self.r = r
        self.k = k

    def sag(self, x, y):
        c = 1 / self.r
        r2 = (x - self.dx) ** 2 + (y - self.dy) ** 2
        return self.dz + r2 * c / (1 + np.sqrt(1 - (1 + self.k) * r2 * c ** 2))


class Sphere(Standard):
    def __init__(self, r, dx=0.0, dy=0.0, dz=0.0):
        super().__init__(r, 0, dx, dy, dz)


class Aperture:
    def __init__(self, x_width, y_width, dx, dy):
        self.x_width = x_width
        self.y_width = y_width
        self.dx = dx
        self.dy = dy

    def mask(self, x, y):
        raise NotImplementedError


class CircularAperture(Aperture):
    def __init__(self, diameter, dx=0.0, dy=0.0):
        super().__init__(diameter, diameter, dx, dy)
        self.diameter = diameter

    def mask(self, x, y):
        return (x - self.dx) ** 2 + (y - self.dy)**2 > (self.diameter / 2) ** 2


class RectangularAperture(Aperture):
    def __init__(self, x_width, y_width, dx=0.0, dy=0.0):
        super().__init__(x_width, y_width, dx, dy)

    def mask(self, x, y):
        cx = x - self.dx
        cy = y - self.dy
        return (cx > self.x_width / 2) | (cx < (-self.x_width / 2)) |\
               (cy > self.y_width / 2) | (cy < (-self.y_width / 2))


class Substrate:

    def __init__(self, surface, aperture, useful_area=None):
        self.surface = surface
        self.aperture = aperture
        self.useful_area = useful_area
        self._best_sphere = None

    def mask(self, x, y):
        mask = self.aperture.mask(x, y)
        return mask if self.useful_area is None else mask | self.useful_area.mask(x, y)

    def sag(self, x, y):
        return ma.masked_array(self.surface.sag(x, y), self.mask(x, y))

    @property
    def best_sphere(self):
        if self._best_sphere is None:
            self._best_sphere = self._find_best_sphere()
        return self._best_sphere

    def _find_best_sphere(self):
        x, y = self.meshgrid()
        x_mean, y_mean = x.mean(), y.mean()
        points = []
        for px, py in zip((x.min(), x.max(), x_mean, x_mean), (y_mean, y_mean, y.min(), y.max())):
            points.append(Point(px, py, self.surface.sag(px, py)))
        x0, y0, z0, r0 = sphere_from_four_points(*points)
        mini = minimize(self._sphere_min,
                        np.array([r0, x0, y0, z0 - r0]),
                        args=(x, y), method='Powell')
        return Sphere(*mini.x)

    def meshgrid(self, nx=None, ny=None):
        x_min = self.aperture.dx - self.aperture.x_width / 2
        x_max = self.aperture.dx + self.aperture.x_width / 2
        y_min = self.aperture.dy - self.aperture.y_width / 2
        y_max = self.aperture.dy + self.aperture.y_width / 2
        if nx is None:
            nx = int((x_max - x_min) / 0.1)
        if ny is None:
            ny = int((y_max - y_min) / 0.1)
        return np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

    def _sphere_min(self, coefficients, x, y):
        r, dx, dy, dz = coefficients
        if r <= 0:
            return 1e10
        delta = self.sag(x, y) - Sphere(r, dx, dy, dz).sag(x, y)
        return np.mean(delta ** 2)

    def interferogram(self):
        x, y = self.meshgrid()
        delta = self.sag(x, y) - self.best_sphere.sag(x, y)

        wvl = 632e-6
        return (1 + np.cos(2 * np.pi * 2 * delta / wvl)) / 2
