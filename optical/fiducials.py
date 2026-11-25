from pickle import APPEND

from optical.surfaces import EGASubstrate
from optics.geometry import Polygon, Point
from itertools import permutations
from scipy.stats import circmean, circstd
import numpy as np
from skimage.transform import SimilarityTransform
from scipy.optimize import least_squares
from scipy.linalg import block_diag


def estimate(ordered_polygons, reference, offset_angles=None, isotropic=True, global_magnification=True):

    n_polygons = len(ordered_polygons)
    n_vertices = len(ordered_polygons[0].vertices)

    dst = []
    for i in range(n_polygons):
        xy = np.array([[v.x, v.y] for v in reference.vertices]).T
        dst.extend(xy)
    dst = np.array(dst)

    if offset_angles is not None:
        unique_angles, inverse_unique = np.unique(offset_angles, return_inverse=True)
        n_unique = len(unique_angles)

    def pack_parameters(similarities):

        scales = [s.scale for s in similarities]
        if global_magnification:
            scales = np.mean(scales),
        elif offset_angles is not None:
            sc = []
            for angle in unique_angles:
                sc.append(np.mean([scale for oa, scale in zip(offset_angles, scales) if angle == oa]))
            scales = sc

        angles = [(-s.rotation) % (2 * np.pi) for s in similarities]
        if offset_angles is not None:  # adjust only the delta angle
            deltas = [a - o for a, o in zip(angles, np.radians(offset_angles))]
            angles = circmean(deltas),

        translations = [s.translation for s in similarities]
        if offset_angles is not None:
            tr = []
            for angle in unique_angles:
                x = np.mean([translation[0] for oa, translation in zip(offset_angles, translations) if angle == oa])
                y = np.mean([translation[1] for oa, translation in zip(offset_angles, translations) if angle == oa])
                tr.append((x, y))
            translations = tr
        translations = [coord for t in translations for coord in t]

        scales_bounds = (0,) * len(scales)
        angles_bounds = (-np.inf,) * len(angles)
        translations_bounds = (-np.inf,) * len(translations)

        if isotropic:
            return ([*scales, *angles, *translations],
                    [(*scales_bounds, *angles_bounds, *translations_bounds), np.inf])
        else:
            return ([*scales, *scales, *angles, *translations],
                    [(*scales_bounds, *scales_bounds, *angles_bounds, *translations_bounds), np.inf])

    def unpack_parameters(p):
        if global_magnification:
            gx = (p[0],) * n_polygons
            p = p[1:]
        elif offset_angles is not None:
            gx = (p[0:n_unique])[inverse_unique]
            p = p[n_unique:]
        else:
            gx = p[0:n_polygons]
            p = p[n_polygons:]

        if isotropic:
            gy = gx
        else:
            if global_magnification:
                gy = (p[0],) * n_polygons
                p = p[1:]
            elif offset_angles is not None:
                gy = (p[0:n_unique])[inverse_unique]
                p = p[n_unique:]
            else:
                gy = p[0:n_polygons]
                p = p[n_polygons:]

        if offset_angles is None:
            angles = p[0:n_polygons] % (2 * np.pi)
            p = p[n_polygons:]
        else:
            angles = [(p[0] + o) % (2 * np.pi) for o in np.radians(offset_angles)]
            p = p[1:]

        if offset_angles is None:
            translations = list(zip(p[::2], p[1::2]))
        else:
            translations = list(zip((p[::2])[inverse_unique], (p[1::2])[inverse_unique]))

        return gx, gy, angles, translations

    def residuals(p):

        gxs, gys, angles, translations = unpack_parameters(p)

        matrix = []
        src = []
        for polygon, gx, gy, angle, (dx, dy) in zip(ordered_polygons, gxs, gys, angles, translations):

            cos, sin = np.cos(angle), np.sin(angle)
            translation = np.array([[gx, 0, -dx * gx],
                                    [0, gy, -dy * gy]])
            rotation = np.array([[cos, sin],
                                 [-sin, cos]])
            matrix.append(rotation @ translation)

            xy = np.array([[v.x, v.y] for v in polygon.vertices]).T
            xy = np.concatenate((xy, np.ones((1, xy.shape[1]))))
            src.append(xy)

        matrix = block_diag(*matrix)
        src = block_diag(*src)

        pred = matrix @ src

        blocks = np.array([pred[2 * i:2 * i + 2, n_vertices * i:n_vertices * i + n_vertices] - dst[2 * i:2 * i + 2, :] for i in range(n_polygons)])

        return blocks.ravel()

    xy_reference = np.array([[v.x, v.y] for v in reference.vertices])
    similarities = []
    for polygon in ordered_polygons:  # One similarity per polygon
        xy_polygon = np.array([[v.x, v.y] for v in polygon.vertices])
        s = SimilarityTransform()
        s.estimate(xy_polygon, xy_reference)
        similarities.append(s)

    p0, bounds = pack_parameters(similarities)

    res = least_squares(residuals, p0, bounds=bounds)

    return unpack_parameters(res.x)


def match_polygons(polygons, reference, offset_angles=None, absolute_angles=False, estimator='mean', global_magnification=True, isotropic=True):
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

    res = estimate(ordered_polygons, reference, offset_angles=offset_angles, isotropic=isotropic, global_magnification=global_magnification)
    print(np.degrees(res[2]))

    return [{'gx': gx, 'gy': gy, 'dx': dx, 'dy': dy, 'theta': np.degrees(angle)} for gx, gy, angle, (dx, dy) in zip(*res)]


def ega_from_fiducials(measured_fiducials, substrate: EGASubstrate, to_normal:bool, **kwargs):

    vertices = []
    for v in substrate.fiducials.vertices:
        if to_normal:
            x, y, _ = substrate.matrix_to_normal() @ (v.x, v.y, v.z, 1)
        else:
            x, y = v.x, v.y
        vertices.append(Point(x, y, 0))
    return match_polygons(measured_fiducials, Polygon(vertices), **kwargs)
