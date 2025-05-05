from optical.surfaces import EGASubstrate
from optics.geometry import Polygon, Point
from itertools import permutations
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

    estimator = {'mean': np.mean, 'median': np.median}[estimator]

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
        idx = np.argmin([np.std(edges_theta) for edges_theta in theta])
        ordered_polygons.append(permuted_polygons[idx])
        angles.append(np.degrees(theta[idx]))

    if offset_angles is None:  # No angles were passed
        # The roll between the polygon and the reference is the mean of the angles
        angles = [estimator(edges_angles) for edges_angles in angles]
    elif absolute_angles:  # Absolute angles were passed
        angles = offset_angles
    else:  # Relative angles were passed
        # The roll between the polygons and the reference is the mean of all angles plus the offset_angles
        delta = estimator([angle - offset for edges_angles, offset in zip(angles, offset_angles)
                           for angle in edges_angles])
        angles = [delta + offset for offset in offset_angles]

    # The scale is the mean of the scales between all pairs of sides
    g = estimator([edge2.v.norm / edge1.v.norm for polygon in ordered_polygons
                   for edge1, edge2 in zip(polygon.edges, reference.edges)])

    output = []
    for polygon, angle in zip(ordered_polygons, angles):

        # Scale and rotate the reference to the polygon
        cos_g, sin_g = np.cos(np.radians(angle)) / g, np.sin(np.radians(angle)) / g
        rotation_matrix = np.array([[cos_g, -sin_g],
                                    [sin_g, cos_g]])
        xy = np.stack(([v.x for v in reference.vertices], [v.y for v in reference.vertices]))
        x, y = rotation_matrix @ xy

        # Compute the mean translation between the vertices of the polygon and those of the scaled and rotated reference
        dx = estimator([v.x - x for v in polygon.vertices])
        dy = estimator([v.y - y for v in polygon.vertices])

        output.append({'gx': g, 'gy': g, 'dx': dx, 'dy': dy, 'theta': angle})

    if type(polygons) == Polygon:
        output = output[0]

    return output


def ega_from_fiducials(measured_fiducials, substrate: EGASubstrate, **kwargs):

    vertices = []
    for v in substrate.fiducials.vertices:
        x, y, _ = substrate.matrix_to_normal() @ (v.x, v.y, v.z, 1)
        vertices.append(Point(x, y, 0))
    return match_polygons(measured_fiducials, Polygon(vertices), **kwargs)
