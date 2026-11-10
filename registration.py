import numpy as np
import cv2


def register(image, diameters=(1, 20)):
    """
    Returns the offset between image and reference computed by cross correlation
    """

    def _vertex_offset(c0, c1, c2):
        """Sub-pixel offset of the vertex of a parabola through three samples.
        Returns 0 if the samples are collinear."""
        c0, c1, c2 = c0 ** 2, c1 ** 2, c2 ** 2
        d = c0 - 2 * c1 + c2
        return 0.0 if d == 0 else (c0 - c2) / (2 * d)

    def parabolic_interpolation_nd(cc):
        """
        Returns the sub-pixel position of the maximum of cc as (x, y, z, ...),
        """
        idx = np.unravel_index(np.argmax(cc), cc.shape)

        if any(i == 0 or i == n - 1 for i, n in zip(idx, cc.shape)):
            return tuple(int(i) for i in idx[::-1])

        pos = []
        for axis, i in enumerate(idx):
            sl = list(idx)
            sl[axis] = slice(i - 1, i + 2)
            c0, c1, c2 = cc[tuple(sl)]
            pos.append(i + _vertex_offset(c0, c1, c2))

        return tuple(pos[::-1])

    ccc = []
    width = 2 * diameters[1]
    for diameter in range(*diameters):
        x = np.linspace(-width / 2, width / 2, width)
        x, y = np.meshgrid(x, x, sparse=False)
        template = np.sqrt(x**2 + y**2)
        radius = diameter / 2
        template[template < radius - 0.5] = radius - 0.5
        template[template > radius + 0.5] = radius + 0.5

        ccc.append(cv2.matchTemplate(np.float32(image),
                               np.float32(template),
                               cv2.TM_CCOEFF_NORMED))

    dx, dy, dz = parabolic_interpolation_nd(np.asarray(ccc))

    dx += width / 2
    dy += width / 2
    dz += diameters[0]

    return dx, dy, dz