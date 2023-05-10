import numpy as np
import h5py


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def read_asc(file):
    def read_array():
        data = []
        for line in f:
            if line[0] == '#':
                break
            data.extend(line.split())

        if len(data) == 0:
            return

        array = np.array([float(num) for num in data])
        array[array == 2147483640] = np.nan
        array *= wavelength / 65536
        array = array.reshape(nx, ny)
        array = np.flipud(array)

        return array

    with open(file) as f:
        for i in range(4):
            line = f.readline()

        split = line.split()
        nx = int(split[2])
        ny = int(split[3])

        for i in range(4):
            line = f.readline()

        split = line.split()
        wavelength = 1e3 * float(split[2])  # mm

        for i in range(7):
            f.readline()

        return read_array(), read_array()


def read_psix(file):
    with h5py.File(file, 'r') as f:
        data_frame = f.get('Data/FrameBuffer')
        data_set = data_frame.get(list(data_frame.keys())[0])
        data = data_set[()]
        for d in data:
            d[:] = np.flipud(d)
        return data


class SagData:

    def __init__(self, file, gx, gy=None, dx=0, dy=0, theta=0, binning=1):
        self.file = file
        _, sag = read_asc(file)
        self._sag = sag if binning == 1 else rebin(sag, (sag.shape[0] // binning, sag.shape[1] // binning))
        self.gx = gx * binning
        self.gy = gx if gy is None else gy * binning
        self.dx = dx / binning
        self.dy = dy / binning
        self.theta = np.radians(theta)
        y, x = np.indices(self._sag.shape)
        xy = np.stack((x.ravel(), y.ravel(), np.ones(x.size)))
        cos, sin = np.cos(theta), np.sin(theta)
        translation = np.array([[self.gx, 0, self.dx * self.gx],
                                [0, self.gy, self.dy * self.gy]])
        rotation = np.array([[+cos, sin],
                             [-sin, cos]])
        self.grid = rotation @ translation @ xy
