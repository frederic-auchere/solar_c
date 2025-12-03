import numpy as np

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