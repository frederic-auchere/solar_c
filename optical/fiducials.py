from optical import rectangular_lw_substrate_fiducials, rectangular_sw_substrate_fiducials
from optics.geometry import Polygon
from itertools import permutations
import numpy as np


def match_polygons(polygon, reference):

    perms = permutations(range(len(reference.edges)))
    permuted_polygons = [Polygon([polygon.vertices[p] for p in perm]) for perm in perms]

    theta = []
    for polygon in permuted_polygons:
        t = []
        for edge2, edge1 in zip(polygon.edges, reference.edges):
            cross = edge1.v.cross(edge2.v)
            t.append(np.arctan2(cross.norm * np.sign(cross.z), edge1.v.dot(edge2.v)))
        theta.append(t)

    idx = np.argmin([np.std(t) for t in theta])
    polygon = permuted_polygons[idx]

    theta = np.mean(theta[idx])

    g = np.mean([edge2.v.norm / edge1.v.norm for edge1, edge2 in zip(polygon.edges, reference.edges)])

    cos_g, sin_g = np.cos(theta) / g, np.sin(theta) / g
    rotation_matrix = np.array([[cos_g, -sin_g],
                                [sin_g, cos_g]])
    xy = np.stack(([v.x for v in reference.vertices], [v.y for v in reference.vertices]))
    x, y = rotation_matrix @ xy

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