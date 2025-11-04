from optical.surfaces import EGASubstrate
from optics.geometry import Polygon, Point
from itertools import permutations
from scipy.stats import circmean, circstd
import numpy as np


def match_polygons(polygons, reference, offset_angles=None, absolute_angles=False, estimator='mean'):
    """
    Computes the transformation between a list of polygons and a reference polygon.

    polygons: list of Polygon objects
    reference: Polygon object

    output: dict
    dx, dy: offset between the origins of the reference frames of the polygon and reference polygon.
    g: scale of the reference with respect to the polygon.
    If the reference vertices coordinates are in [mm] and those of the polygon in [pixels], the scale is in [mm / pixel]
    """

    circ_estimator = {'mean': circmean}[estimator]
    estimator = {'mean': np.mean}[estimator]

    if type(polygons) == Polygon:
        polygons = [polygons]

    # Finds which permutation of vertices matches the reference polygon
    angles = []
    ordered_polygons = []
    for polygon in polygons:

        # Defines all possible permutations of vertices orders
        # This allows the vertices of the polygons to be compared in any order
        perms = permutations(range(len(reference.edges)))
        permuted_polygons = [Polygon([polygon.vertices[p] for p in perm]) for perm in perms]
        # For each permutation, compute the angles between the edges of the polygon and those of the reference polygon
        theta = []
        for p in permuted_polygons:
            edges_theta = []
            for edge2, edge1 in zip(p.edges, reference.edges):
                cross = edge1.v.cross(edge2.v)
                edges_theta.append(
                    np.arctan2(cross.norm * np.sign(cross.z) / (edge1.v.norm * edge2.v.norm), edge1.v.dot(edge2.v))
                )
            theta.append(edges_theta)

        # The best match polygon is the one for which the angles between all pairs of sides have the smallest std
        idx = np.argmin([circstd(edges_theta) for edges_theta in theta])
        ordered_polygons.append(permuted_polygons[idx])
        angles.append(theta[idx])

    if offset_angles is None:  # No angles were passed
        # The roll between the polygon and the reference is the mean of the angles
        angles = [np.degrees(circ_estimator(edges_angles)) for edges_angles in angles]
        angles_std = [np.degrees(circstd(edges_angles)) for edges_angles in angles]
    elif absolute_angles:  # Absolute angles were passed
        angles = offset_angles
        angles_std = [0] * len(angles)
    else:  # Relative angles were passed
        # The roll between the polygons and the reference is the mean of all angles plus the offset_angles
        deltas = [angle - offset for edges_angles, offset in zip(angles, np.radians(offset_angles))
                           for angle in edges_angles]
        delta = np.degrees(circ_estimator(deltas))
        angles = [delta + offset for offset in offset_angles]
        angles_std = [np.degrees(circstd(deltas))] * len(angles)

    # The scale is the mean of the scales between all pairs of sides
    scales = [edge2.v.norm / edge1.v.norm for polygon in ordered_polygons
                   for edge1, edge2 in zip(polygon.edges, reference.edges)]
    g = estimator(scales)
    g_std = np.std(scales)

    output = []
    for polygon, angle, angle_std in zip(ordered_polygons, angles, angles_std):

        # Scale and rotate the reference to the polygon
        cos_g, sin_g = np.cos(np.radians(angle)) / g, np.sin(np.radians(angle)) / g
        rotation_matrix = np.array([[cos_g, -sin_g],
                                    [sin_g, cos_g]])
        xy = np.stack(([v.x for v in reference.vertices], [v.y for v in reference.vertices]))
        x, y = rotation_matrix @ xy

        # Compute the mean translation between the vertices of the polygon and those of the scaled and rotated reference
        dxs = [v.x - x for v, x in zip(polygon.vertices, x)]
        dx = estimator(dxs)
        dx_std = np.std(dxs)
        dys = [v.y - y for v, y in zip(polygon.vertices, y)]
        dy = estimator(dys)
        dy_std = np.std(dys)

        output.append({'gx': g, 'gx_std': g_std,
                       'gy': g, 'gy_std': g_std,
                       'dx': dx, 'dx_std': dx_std,
                       'dy': dy, 'dy_std': dy_std,
                       'theta': angle, 'theta_std': angle_std})

    return output


def ega_from_fiducials(measured_fiducials, substrate: EGASubstrate, to_normal:bool, **kwargs):

    vertices = []
    for v in substrate.fiducials.vertices:
        if to_normal:
            x, y, _ = substrate.matrix_to_normal() @ (v.x, v.y, v.z, 1)
        else:
            x, y = v.x, v.y
        vertices.append(Point(x, y, 0))
    return match_polygons(measured_fiducials, Polygon(vertices), **kwargs)
