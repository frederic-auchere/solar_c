import numpy as np
from numpy import ma
from scipy.optimize import minimize
from scipy.interpolate import griddata


def rotation_matrix(angle, axis):
    """
    Returns a rotation matrix about the specified axis (z=0, y=1, x=2) for the
    specified angle (in radians).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)

    if axis == 0:  # Rz
        matrix = np.array([[cos, -sin, 0, 0],
                           [sin, cos, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    elif axis == 1:  # Ry
        matrix = np.array([[cos, 0, sin, 0],
                           [0, 1, 0, 0],
                           [-sin, 0, cos, 0],
                           [0, 0, 0, 1]])
    elif axis == 2:  # Rx
        matrix = np.array([[1, 0, 0, 0],
                           [0, cos, -sin, 0],
                           [0, sin, cos, 0],
                           [0, 0, 0, 1]])

    return matrix


def sphere_from_four_points(p1, p2, p3, p4):
    """
    Adapted from https: // stackoverflow.com / questions / 37449046 / how - to - calculate - the - sphere - center -
    with-4 - points
    """
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


class Point:

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    @property
    def radius(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


class BaseSurface:

    def __sub__(self, other):
        return DifferenceSurface(self, other)


class DifferenceSurface(BaseSurface):
    def __init__(self, surface1, surface2):
        super().__init__()
        self.surface1 = surface1
        self.surface2 = surface2

    def __repr__(self):
        return f'Surface 1: {self.surface1}\nSurface 2: {self.surface2}'

    def sag(self, x, y):
        return self.surface2.sag(x, y) - self.surface1.sag(x, y)


class Surface(BaseSurface):

    @classmethod
    def spherical_parameters(cls, r, dx, dy, dz):
        raise NotImplementedError

    def __init__(self, dx=0, dy=0, dz=0, alpha=0, beta=0, gamma=0):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        return f'dx={self.dx:.3f} [mm] dy={self.dy:.3f} [mm] dz={self.dz:.3f} [mm] ' +\
               f'alpha={np.degrees(self.alpha):.2f} [°] ' +\
               f'beta={np.degrees(self.beta):.2f} [°] ' +\
               f'gamma={np.degrees(self.gamma):.2f} [°]'

    def sag(self, x, y):
        if self.alpha == 0 and self.beta == 0 and self.gamma == 0:
            return self.dz + self._zemax_sag(x - self.dx, y - self.dy)
        else:
            z = np.zeros(x.size)
            xyz = np.stack((x.ravel(), y.ravel(), z.ravel(), np.ones(x.size)))

            rx = rotation_matrix(self.alpha, 0)
            ry = rotation_matrix(self.beta, 1)
            rz = rotation_matrix(self.gamma, 2)
            sxyz = np.array([[1, 0, 0, -self.dx],
                             [0, 1, 0, -self.dy],
                             [0, 0, 1, -self.dz],
                             [0, 0, 0, 1]])
            matrix = rz @ ry @ rx @ sxyz
            nx, ny, _, _ = matrix @ xyz

            nz = self._zemax_sag(nx, ny)

            xyz = np.stack((nx, ny, nz, np.ones(x.size)))
            matrix = np.linalg.inv(matrix)
            nx, ny, nz, _ = matrix @ xyz

            return griddata((nx, ny), nz, (x, y), method='nearest')

    def _zemax_sag(self, x, y):
        raise NotImplementedError


class Toroidal(Surface):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, r, *args

    def __init__(self, rc, rr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rc = rc
        self.rr = rr

    def __repr__(self):
        return f'Rc={self.rc:.3f} Rr={self.rr:.3f} ' + super().__repr__()

    def _zemax_sag(self, x, y):
        c = 1 / self.rc
        y2 = y ** 2
        zy = y2 * c / (1 + np.sqrt(1 - y2 * c ** 2))
        zx = self.rr - zy - np.sqrt((self.rr - zy) ** 2 - x ** 2)
        return zx + zy


class EllipticalGrating(Surface):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, r, r, *args

    def __init__(self, a, b, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self):
        return f'a={self.a:.3f} [mm-1] b={self.b:.3f} [mm-1] c={self.b:.3f} [mm] ' + super().__repr__()

    def _zemax_sag(self, x, y):
        u2 = (self.a * x) ** 2 + (self.b * y) ** 2
        return u2 * self.c / (1 + np.sqrt(1 - u2))


class Standard(Surface):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, 0, *args

    def __init__(self, r, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.k = k

    def __repr__(self):
        return f'R={self.r:.3f} [mm] k={self.k:.3f} ' + super().__repr__()

    def _zemax_sag(self, x, y):
        c = 1 / self.r
        r2 = x ** 2 + y ** 2
        return r2 * c / (1 + np.sqrt(1 - (1 + self.k) * r2 * c ** 2))


class Sphere(Standard):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, *args

    def __init__(self, r, *args, **kwargs):
        super().__init__(r, 0, *args, **kwargs)

    def __repr__(self):
        return f'R={self.r:.3f} [mm] ' + super().__repr__()


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

    @property
    def best_sphere(self):
        if self._best_sphere is None:
            self._best_sphere = self.find_best_surface(Sphere)
        return self._best_sphere

    def find_best_surface(self, surface_class=Sphere, tilt=False):
        x, y = self.meshgrid()
        x_mean, y_mean = x.mean(), y.mean()
        points = []
        for px, py in zip((x.min(), x.max(), x_mean, x_mean), (y_mean, y_mean, y.min(), y.max())):
            points.append(Point(px, py, self.surface.sag(px, py)))
        x0, y0, z0, r0 = sphere_from_four_points(*points)
        initial_parameters = surface_class.spherical_parameters(r0, x0, y0, z0 - r0)
        if tilt:
            initial_parameters = initial_parameters + (0.0, 0.0, 0.0)
        mini = minimize(self._surface_min,
                        np.array(initial_parameters),
                        args=(x, y, surface_class), method='Powell')
        return surface_class(*mini.x)

    def _surface_min(self, coefficients, x, y, surface_class):
        try:
            surface_sag = surface_class(*coefficients).sag(x, y)
        except RuntimeWarning:
            return 1e10
        delta = self.sag(x, y) - surface_sag
        return np.mean(delta ** 2)

    def interferogram(self, phase=0):
        x, y = self.meshgrid()
        delta = self.sag(x, y) - self.best_sphere.sag(x, y)

        wvl = 632e-6
        return (1 + np.cos(2 * np.pi * (phase + 2 * delta / wvl))) / 2
