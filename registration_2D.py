import numpy as np
import cv2


def register(image, diameter):
    """
    Returns the offset between image and reference computed by cross correlation
    """

    def parabolic_interpolation(cc):
        """
        Returns the position of the extremum of the cross_correlation matrix cc
        assuming a parabolic shape
        """
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

            dx = cx if xd == 0 else xi[2] - xn / xd - 0.5
            dy = cy if yd == 0 else yi[2] - yn / yd - 0.5

            return dx, dy

    width = 2 * diameter
    x = np.linspace(-width / 2, width / 2, width)
    x, y = np.meshgrid(x, x, sparse=False)
    template = np.sqrt(x**2 + y**2)
    radius = diameter / 2
    template[template < radius - 0.5] = radius - 0.5
    template[template > radius + 0.5] = radius + 0.5

    cc = cv2.matchTemplate(np.float32(image),
                           np.float32(template),
                           cv2.TM_CCOEFF)
    dx, dy = parabolic_interpolation(cc)

    dx += width / 2
    dy += width / 2

    return dx, dy