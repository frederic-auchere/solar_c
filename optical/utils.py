import numpy as np


def parabolic_interpolation(cc, extremum=np.nanargmax):
    """
    Returns the position of the extremum of matrix cc
    assuming a parabolic shape
    """
    cy, cx = np.unravel_index(extremum(cc, axis=None), cc.shape)
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


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)



def fit_circle_to_points(x_coords, y_coords):
    """
    Fits a circle to 3 or more points using least squares

    """

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    if len(x_coords) < 3:
        raise ValueError("Need at least 3 points to fit a circle")

    # Setting up the linear system
    A = np.c_[2 * x_coords, 2 * y_coords, np.ones(x_coords.size)]
    b = x_coords ** 2 + y_coords ** 2
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)  # keeping all outputs just in case

    center_x, center_y = coeffs[0], coeffs[1]  # circle center coordinates
    radius = np.sqrt(coeffs[2] + center_x ** 2 + center_y ** 2)

    return center_x, center_y, radius