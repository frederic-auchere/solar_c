from optical import rectangular_lw_substrate_fiducials, rectangular_sw_substrate_fiducials
from optics.geometry import Polygon
from itertools import permutations
import numpy as np


def match_polygons(polygon, reference):
    """
    Computes the transformation between a polygon and a reference polygon.

    polygon: Polygon object
    reference: Polygon object

    output: dict
    dx, dy: offset between the origins of the reference frames of the polygon and reference polygon.
    g: scale of the reference with respect to the polygon.
    If the reference vertices coordinates are in [mm] and those of the polygon in [pixels], the scale is in [mm / pixel]
    """

    # Defines all possible permutations of vertices orders
    # This allows the vertices of the polygons to be compared in any order
    perms = permutations(range(len(reference.edges)))
    permuted_polygons = [Polygon([polygon.vertices[p] for p in perm]) for perm in perms]

    # For each permutation, compute the angles between the polygon and the reference polygon
    theta = []
    for polygon in permuted_polygons:
        t = []
        for edge2, edge1 in zip(polygon.edges, reference.edges):
            cross = edge1.v.cross(edge2.v)
            t.append(np.arctan2(cross.norm * np.sign(cross.z) / (edge1.v.norm * edge2.v.norm), edge1.v.dot(edge2.v)))
        theta.append(t)

    # The best match polygon is the one for which the angles between all pairs of sides are the closest (smallest std)
    idx = np.argmin([np.std(t) for t in theta])
    polygon = permuted_polygons[idx]

    theta = np.mean(theta[idx])  # The roll between the polygon and the reference is the mean of the angles

    # The scale is the mean of the scales between the pairs of sides
    g = np.mean([edge2.v.norm / edge1.v.norm for edge1, edge2 in zip(polygon.edges, reference.edges)])
    # Scale and rotate the reference to the polygon
    cos_g, sin_g = np.cos(theta) / g, np.sin(theta) / g
    rotation_matrix = np.array([[cos_g, -sin_g],
                                [sin_g, cos_g]])
    xy = np.stack(([v.x for v in reference.vertices], [v.y for v in reference.vertices]))
    x, y = rotation_matrix @ xy

    # Compute the mean translation between the vertices of the polygon and those of the scaled and rotated reference
    dx = np.mean(np.array([v.x for v in polygon.vertices]) - x)
    dy = np.mean(np.array([v.y for v in polygon.vertices]) - y)

    return {'g': g, 'dx': dx, 'dy': dy, 'theta': np.degrees(theta)}


def ega_from_fiducials(measured_fiducials, grating):

    if grating == 'lw':
        reference_fiducials = rectangular_lw_substrate_fiducials
    elif grating == 'sw':
        reference_fiducials = rectangular_sw_substrate_fiducials
    else:
        raise ValueError('grating must be either lw or sw')

    return match_polygons(Polygon(measured_fiducials), Polygon(reference_fiducials))