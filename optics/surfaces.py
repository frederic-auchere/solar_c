import numbers
import numpy as np
from numpy import ma
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation


def _lad(array):
    return np.mean(np.abs(array - np.mean(array)))


def _rms(array):
    return np.sqrt(np.mean(array ** 2))


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
    if uvw == 0:
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

    name = ''

    def __sub__(self, other):
        return DifferenceSurface(self, other)

    def sag(self, xy):
        raise NotImplementedError


class DifferenceSurface(BaseSurface):

    def __init__(self, surface1, surface2):
        super().__init__()
        self.surface1 = surface1
        self.surface2 = surface2

    def __repr__(self):
        return f'Surface 1: {self.surface1}\nSurface 2: {self.surface2}'

    def sag(self, xy):
        return self.surface1.sag(xy) - self.surface2.sag(xy)


class MeasuredSurface(BaseSurface):

    def __init__(self, x, y, sag):
        self._sag = sag
        self.grid = x, y

    def sag(self, xy, method='nearest', **kwargs):
        x, y = xy
        if isinstance(x, numbers.Number):
            x, y = np.array([x]), np.array([y])
        xg, yg = self.grid
        return griddata((xg.ravel(), yg.ravel()), self._sag.ravel(), (x.ravel(), y.ravel()),
                        method=method, **kwargs).reshape(x.shape)


class ParametricSurface(BaseSurface):

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
               f'alpha={np.degrees(self.alpha):.4f} [°] ' +\
               f'beta={np.degrees(self.beta):.4f} [°] ' +\
               f'gamma={np.degrees(self.gamma):.4f} [°]'

    def primary_radius(self):
        return NotImplementedError

    def sag(self, xy, method='nearest', **kwargs):
        x, y = xy
        if self.alpha == 0 and self.beta == 0 and self.gamma == 0:
            return self.dz + self._zemax_sag(x - self.dx, y - self.dy)
        else:
            if isinstance(x, numbers.Number):
                x, y = np.array([x]), np.array([y])
            z = np.zeros(x.size)
            xyz = np.stack((x.ravel(), y.ravel(), z, np.ones(x.size)))
            rotation = Rotation.from_euler('zyx', (self.alpha, self.beta, self.gamma)).as_matrix()
            translation = np.array([[1, 0, 0, -self.dx],
                                    [0, 1, 0, -self.dy],
                                    [0, 0, 1, -self.dz]])
            matrix = rotation @ translation
            nx, ny, _ = matrix @ xyz

            nz = self._zemax_sag(nx, ny)

            xyz = np.stack((nx, ny, nz, np.ones(x.size)))
            matrix = np.concatenate((matrix, [[0, 0, 0, 1]]))
            matrix = np.linalg.inv(matrix)[:-1]
            nx, ny, nz = matrix @ xyz

            return griddata((nx, ny), nz, (x, y), method=method, **kwargs).reshape(x.shape)

    def _zemax_sag(self, x, y):
        raise NotImplementedError


class Toroidal(ParametricSurface):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, r, *args

    def __init__(self, rc, rr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rc = rc
        self.rr = rr

    def __repr__(self):
        return f'Rc={self.rc:.3f} [mm] Rr={self.rr:.3f} [mm] {super().__repr__()}'

    def primary_radius(self):
        return self.rc

    def _zemax_sag(self, x, y):
        c = 1 / self.rc
        y2 = y ** 2
        zy = y2 * c / (1 + np.sqrt(1 - y2 * c ** 2))
        zx = self.rr - zy - np.sqrt((self.rr - zy) ** 2 - x ** 2)
        return zx + zy


class EllipticalGrating(ParametricSurface):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, r, r, *args

    def __init__(self, a, b, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = a
        self.b = b
        self.c = c

    def primary_radius(self):
        return self.c

    def __repr__(self):
        return f'a={self.a:.3f} [mm-1] b={self.b:.3f} [mm-1] c={self.b:.3f} [mm] {super().__repr__()}'

    def _zemax_sag(self, x, y):
        u2 = (self.a * x) ** 2 + (self.b * y) ** 2
        return u2 * self.c / (1 + np.sqrt(1 - u2))


class Standard(ParametricSurface):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, 0, *args

    def __init__(self, r, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.k = k

    def primary_radius(self):
        return self.r

    def __repr__(self):
        return f'R={self.r:.3f} [mm] k={self.k:.3f} {super().__repr__()}'

    def _zemax_sag(self, x, y):
        c = 1 / self.r
        r2 = x ** 2 + y ** 2
        return r2 * c / (1 + np.sqrt(1 - (1 + self.k) * r2 * (c ** 2)))


class Sphere(Standard):

    @classmethod
    def spherical_parameters(cls, r, *args):
        return r, *args

    def __init__(self, r, *args, **kwargs):
        super().__init__(r, 0, *args, **kwargs)


class EllipticalGratingSubSphere(DifferenceSurface):

    def __init__(self, a, b, c, dx, dy, rs, dxs, dys, dzs):
        super().__init__(EllipticalGrating(a, b, c, dx, dy), Sphere(rs, dxs, dys, dzs))


class StandardSubSphere(DifferenceSurface):

    def __init__(self, r, k, dx, dy, rs, dxs, dys, dzs):
        super().__init__(Standard(r, k, dx, dy), Sphere(rs, dxs, dys, dzs))


class Aperture:

    def __init__(self, x_width, y_width, dx, dy):
        self.x_width = x_width
        self.y_width = y_width
        self.dx = dx
        self.dy = dy

    def mask(self, x, y):
        raise NotImplementedError

    @property
    def limits(self):
        x_min = self.dx - self.x_width / 2
        x_max = self.dx + self.x_width / 2
        y_min = self.dy - self.y_width / 2
        y_max = self.dy + self.y_width / 2
        return x_min, x_max, y_min, y_max


class CircularAperture(Aperture):

    def __init__(self, diameter, dx=0.0, dy=0.0):
        super().__init__(diameter, diameter, dx, dy)
        self.diameter = diameter

    def mask(self, x, y):
        return (x - self.dx) ** 2 + (y - self.dy) ** 2 > (self.diameter / 2) ** 2


class RectangularAperture(Aperture):

    def __init__(self, x_width, y_width, dx=0.0, dy=0.0):
        super().__init__(x_width, y_width, dx, dy)

    def mask(self, x, y):
        cx = x - self.dx
        cy = y - self.dy
        return (cx > self.x_width / 2) | (cx < (-self.x_width / 2)) |\
               (cy > self.y_width / 2) | (cy < (-self.y_width / 2))


class SquareAperture(RectangularAperture):

    def __init__(self, width, dx=0.0, dy=0.0):
        super().__init__(width, width, dx, dy)


class Substrate:

    def __init__(self, surface, aperture, useful_area=None):
        self.surface = surface
        self.aperture = aperture
        self.useful_area = useful_area
        self._best_sphere = None
        self._grid = None
        self._sag = None

    def four_points(self):
        min_radius = min((self.aperture.x_width, self.aperture.y_width))
        # middle point
        points = [Point(self.aperture.dx, self.aperture.dy, self.surface.sag((self.aperture.dx, self.aperture.dy)))]
        # three points on a circle
        for theta in (0, 120, 240):
            x = self.aperture.dx + min_radius * np.cos(np.radians(theta))
            y = self.aperture.dy + min_radius * np.sin(np.radians(theta))
            z = self.surface.sag((x, y))
            points.append(Point(x, y, z))
        return points

    def mask(self, xy):
        mask = self.aperture.mask(*xy)
        return mask if self.useful_area is None else mask | self.useful_area.mask(*xy)

    def sag(self, xy=None):
        if xy is None:
            if self._sag is None:
                xy = self.grid()
                self._sag = ma.masked_array(self.surface.sag(xy), self.mask(xy))
            return self._sag
        else:
            return ma.masked_array(self.surface.sag(xy), self.mask(xy))

    def grid(self, nx=None, ny=None):
        if nx is not None or ny is not None or self._grid is None:
            x_min, x_max, y_min, y_max = self.aperture.limits
            if nx is None:
                nx = int(self.aperture.x_width / 0.1)
            if ny is None:
                ny = int(self.aperture.y_width / 0.1)
            self._grid = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        return self._grid

    @property
    def best_sphere(self):
        if self._best_sphere is None:
            self._best_sphere = self.find_best_surface(Sphere)
        return self._best_sphere

    def find_best_surface(self, surface_class=Sphere, tilt=False, initial_parameters=None,
                          method='Powell', objective='rms', **kwargs):
        objectives = {'rms': _rms, 'lad': _lad, 'std': np.std}
        if initial_parameters is None:
            x0, y0, z0, r0 = sphere_from_four_points(*self.four_points())
            sign = np.sign(self.surface.primary_radius())
            initial_parameters = surface_class.spherical_parameters(sign * r0, x0, y0, np.abs(z0) - r0)
            if tilt:
                initial_parameters = initial_parameters + (0.0, 0.0, 0.0)

        mini = minimize(self._surface_min, np.array(initial_parameters),
                        args=(surface_class, objectives[objective]), method=method, **kwargs)
        return surface_class(*mini.x)

    def _surface_min(self, coefficients, surface_class, objective=_rms):
        try:
            surface_sag = surface_class(*coefficients).sag(self.grid())
        except RuntimeWarning:
            return 1e10
        surface_sag -= self.sag().data
        return objective(surface_sag[~self.sag().mask])

    def interferogram(self, phase=0, dx=0, dy=0):
        sphere = Sphere(self.best_sphere.r, self.best_sphere.dx + dx, self.best_sphere.dy + dy, self.best_sphere.dz)
        delta = self.sag() - sphere.sag(self.grid())

        wvl = 632e-6
        return (1 + np.cos(2 * np.pi * (phase + 2 * delta / wvl))) / 2
