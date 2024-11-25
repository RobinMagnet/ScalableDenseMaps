"""
.. note::
    Torch and cuda-compatible implementation of:
        [1] - "Deblurring and Denoising of Maps between Shapes", by Danielle Ezuz and Mirela Ben-Chen.
"""

from tqdm.auto import tqdm

import torch as th
import torch.nn as nn

from .nn_utils import nn_query


def nn_query_precise_torch(
    vert_emb, faces, points_emb, return_dist=False, batch_size=None, clear_cache=True
):
    """
    Project a pointcloud on a p-dimensional triangle mesh

    Parameters
    ----------------------------
    vert_emb      :
        (n1, p) coordinates of the mesh vertices
    faces         :
        (m1, 3) faces of the mesh defined as indices of vertices
    points_emb    :
        (n2, p) coordinates of the pointcloud
    return_dist   : bool
        Whether to return the distance to the nearest vertex
    batch_size    : int, optional
        If precompute_dmin is False, projects batches of points on the surface
    clear_cache   : bool
        Whether to clear cache after computation

    Returns
    ----------------------------
    face_match  : torch.Tensor
        (n2,) - indices of the face assigned to each point.
    bary_coord  : torch.Tensor
        (n2,3) - barycentric coordinates of each point within the face.
    """

    with th.no_grad():
        # n2,  (n2,3)
        face_match, bary_coords = project_pc_to_triangles(
            vert_emb,
            faces,
            points_emb,
            precompute_dmin=batch_size is None,
            batch_size=batch_size,
            verbose=False,
        )

        if vert_emb.is_cuda and clear_cache:
            th.cuda.empty_cache()

    if return_dist:
        targets = (bary_coords.unsqueeze(-1) * vert_emb[faces[face_match]]).sum(
            1
        )  # (n2, p)
        dists = th.linalg.norm(targets - points_emb, dim=-1)

        return face_match, bary_coords, dists

    return face_match, bary_coords


def project_pc_to_triangles(
    vert_emb, faces, points_emb, precompute_dmin=True, batch_size=None, verbose=False
):
    """
    Project a pointcloud on a set of triangles in p-dimension. Projection is defined as barycentric coordinates on one of the triangle.
    Line i for the output has 3 non-zero values at indices j,k and l of the vertices of the triangle point i is projected on.

    Parameters
    ----------------------------
    vert_emb        : torch.Tensor
        (n1, p) coordinates of the mesh vertices
    faces           : torch.Tensor
        (m1, 3) faces of the mesh defined as indices of vertices
    points_emb      : torch.Tensor
        (n2, p) coordinates of the pointcloud
    precompute_dmin : bool
        Whether to precompute all the values of delta_min. Faster but heavier in memory.
    batch_size      : int, optional
        If precompute_dmin is False, projects batches of points on the surface


    Returns
    ----------------------------
    face_match  : torch.Tensor
        (n2,) - indices of the face assigned to each point.
    bary_coord  : torch.Tensor
        (n2,3) - barycentric coordinates of each point within the face.
    """
    assert vert_emb.ndim == 2
    assert points_emb.ndim == 2

    vert_emb = vert_emb.contiguous()
    points_emb = points_emb.contiguous()

    n_points = points_emb.shape[0]

    # Compute the maximum edge length per triangle
    lmax = compute_per_tri_max_edge_length(vert_emb, faces)  # (m1,)

    # Find closest vertex to each point
    if batch_size is not None:
        if precompute_dmin:
            print("WARNING, `precompute_dmin` can't be True if batch size is not None")
        precompute_dmin = False
    if precompute_dmin:
        Deltamin = nn_query(vert_emb, points_emb)  # (n2,)

    # Find closest vertex on each face
    dmin = None
    if precompute_dmin:
        dmin = compute_per_triangle_min_dist(vert_emb, faces, points_emb)  # (n_f1,n2)
        dmin_params = None

    else:
        vert_sqnorms = th.linalg.norm(vert_emb, dim=1) ** 2
        points_sqnorm = th.linalg.norm(points_emb, dim=1) ** 2
        dmin_params = {"vert_sqnorms": vert_sqnorms, "points_sqnorm": points_sqnorm}

    # Iterate along all points
    if precompute_dmin or batch_size is None:
        faceind, bary = project_to_mesh_multi(
            vert_emb,
            faces,
            points_emb,
            th.arange(n_points),
            lmax,
            Deltamin,
            dmin=dmin,
            dmin_params=dmin_params,
        )
        face_match = faceind
        bary_coord = bary

    else:
        # Number of batches
        n_batches = n_points // batch_size + int((n_points % batch_size) > 0)
        iterable = range(n_batches) if not verbose else tqdm(range(n_batches))

        # Initialize output
        face_match = th.zeros(n_points, dtype=int, device=vert_emb.device)
        bary_coord = th.zeros(n_points, 3, device=vert_emb.device)

        for batchind in iterable:
            batch_minmax = [
                batch_size * batchind,
                min(n_points, batch_size * (1 + batchind)),
            ]

            # Compute quentities for face simplification
            dmin_batch = compute_per_triangle_min_dist(
                vert_emb,
                faces,
                points_emb[batch_minmax[0] : batch_minmax[1]],
                vert_sqnorm=vert_sqnorms,
                points_sqnorm=points_sqnorm[batch_minmax[0] : batch_minmax[1]],
            )

            Deltamin_batch = nn_query(
                vert_emb, points_emb[batch_minmax[0] : batch_minmax[1]].contiguous()
            )

            # Get faceinds and barycentric coordinates
            faceind, bary = project_to_mesh_multi(
                vert_emb,
                faces,
                points_emb[batch_minmax[0] : batch_minmax[1]],
                th.arange(*batch_minmax) - batch_minmax[0],
                lmax,
                Deltamin_batch,  # Deltamin[batch_minmax[0]:batch_minmax[1]],
                dmin=dmin_batch,
                dmin_params=dmin_params,
            )

            face_match[th.arange(*batch_minmax)] = faceind
            bary_coord[th.arange(*batch_minmax)] = bary

    return face_match, bary_coord


def compute_per_tri_max_edge_length(vert_emb, faces):
    """
    Computes the maximum edge length per triangle

    Parameters
    ----------------------------
    vert_emb      :
        (n1, p) coordinates of the mesh vertices
    faces         :
        (m1, 3) faces of the mesh defined as indices of vertices

    Returns
    ----------------------------
    lmax : torch.Tensor
        (m1,) maximum edge length
    """
    # print("VEMB", vert_emb.max())

    tri_embs = vert_emb[faces]  # (m1, 3, p)

    lmax = (
        th.linalg.norm(tri_embs - tri_embs[:, [1, 2, 0]], dim=-1).max(dim=1).values
    )  # (m1,)

    return lmax


def mycdist(X, Y, sqnormX=None, sqnormY=None, squared=False):
    """
    Compute pairwise euclidean distance between two collections of vectors in a k-dimensional space

    Parameters
    --------------
    X       :
        (n1, k) first collection
    Y       :
        (n2, k) second collection or (k,) if single point
    squared : bool
        whether to compute the squared euclidean distance

    Returns
    --------------
    distmat : torch.Tensor
        (n1, n2) or (n2,) distance matrix
    """

    if sqnormX is None:
        sqnormX = th.linalg.norm(X, dim=1) ** 2

    if sqnormY is None:
        if Y.ndim == 2:
            sqnormY = th.linalg.norm(Y, dim=1) ** 2
        else:
            sqnormY = th.linalg.norm(Y) ** 2

    distmat = X @ Y.T  # (n1, n2)
    distmat *= -2

    if Y.ndim == 2:
        distmat += sqnormX[:, None]
        distmat += sqnormY[None, :]
    else:
        distmat += sqnormX
        distmat += sqnormY

    th.clamp(distmat, min=0, out=distmat)

    if not squared:
        th.sqrt(distmat, out=distmat)

    return distmat


def compute_per_triangle_min_dist(
    vert_emb, faces, points_emb, vert_sqnorm=None, points_sqnorm=None
):
    r"""
    Given a vertex in the pointcloud and each face on the surface, gives the minimum distance
    to between the vertex and each of the 3 points of the triangle.

    For a given face on the source shape and vertex on the target shape:
    $\delta_min = \min_{i=1\cdots 3} \|A_{c_i,*} - b\|_2$
    with notations from "Deblurring and Denoising of Maps between Shapes".

    Parameters
    ----------------------------
    vert_emb      :
        (n1, p) coordinates of the mesh vertices
    faces         :
        (m1, 3) faces of the mesh defined as indices of vertices
    points_emb    :
        (n2, p) coordinates of the pointcloud
    vertind       :
        index of the vertex for which to compute dmin
    vert_sqnorm   :
        (n1,) squared norm of each vertex
    points_sqnorm :
        (n2,) squared norm of each point

    Returns
    ----------------------------
    delta_min : torch.Tensor
        (m1,n2) delta_min for each face on the source shape.
    """
    emb0 = vert_emb[faces[:, 0]]  # (m1,k1)
    emb1 = vert_emb[faces[:, 1]]  # (m1,k1)
    emb2 = vert_emb[faces[:, 2]]  # (m1,k1)

    if points_sqnorm is None:
        points_sqnorm = th.linalg.norm(points_emb, dim=1) ** 2
    if vert_sqnorm is None:
        vert_sqnorm = th.linalg.norm(vert_emb, dim=1) ** 2

    distmat = mycdist(
        emb0,
        points_emb,
        sqnormX=vert_sqnorm[faces[:, 0]],
        sqnormY=points_sqnorm,
        squared=True,
    )
    th.minimum(
        distmat,
        mycdist(
            emb1,
            points_emb,
            sqnormX=vert_sqnorm[faces[:, 1]],
            sqnormY=points_sqnorm,
            squared=True,
        ),
        out=distmat,
    )
    th.minimum(
        distmat,
        mycdist(
            emb2,
            points_emb,
            sqnormX=vert_sqnorm[faces[:, 2]],
            sqnormY=points_sqnorm,
            squared=True,
        ),
        out=distmat,
    )
    th.sqrt(distmat, out=distmat)
    return distmat  # (m1,n2)


def project_to_mesh_multi(
    vert_emb, faces, points_emb, vertinds, lmax, Deltamin, dmin=None, dmin_params=None
):
    """
    Project a pointcloud on a p-dimensional triangle mesh

    Parameters
    ----------------------------
    vert_emb    :
        (n1, p) coordinates of the mesh vertices
    faces       :
        (m1, 3) faces of the mesh defined as indices of vertices
    points_emb  :
        (n2, p) coordinates of the pointcloud
    vertinds     :
        (l,) - indices of the vertices to project
    lmax        :
        (m1,) value of lmax (max edge length for each face)
    Deltamin    :
        (n2,) or (l,) value of Deltamin (distance to nearest vertex)
    dmin        :
        (m1,n2) or (m1, l) - optional - values of dmin (distance to the nearest vertex of each face
                  for each vertex). Can be computed on the fly
    dmin_params : dict, optional
        if dmin is None, stores 'vert_sqnorm' a (n1,) array of squared norms
        of vertices embeddings, and 'points_sqnorm' a (n2,) array of squared norms
        of points embeddings. Helps speed up computation of dmin

    Returns
    -----------------------------
    min_faceind :
        (l,) index of the face on which the vertex is projected
    min_bary    :
        (l, 3,) - barycentric coordinates on the chosen face
    """
    dmin_params = dict() if dmin_params is None else dmin_params

    # Obtain deltamin
    if dmin is None:
        deltamin = compute_per_triangle_min_dist(
            vert_emb, faces, points_emb[vertinds], **dmin_params
        )  # (m1, l)
    else:
        deltamin = dmin[:, vertinds]  # (m1,l)

    # Get list of potential faces
    query_faceinds = th.where(
        (deltamin - lmax[:, None] < Deltamin[None, vertinds]).any(1)
    )[
        0
    ]  # (p,) p < m1

    # Projection can be done on multiple triangles
    query_triangles = vert_emb[faces[query_faceinds]]  # (p, 3, k1)
    query_points = points_emb[vertinds]  # (l, k1)

    # (l,), (l,k1), (l,3), (l,)
    # dists, proj, bary_coords, argmin_triangle = PointsTriangleProjLayer().forward(triangles=query_triangles, points=query_points, return_bary=True, min_only=True)
    bary_coords, argmin_triangle = PointsTriangleProjLayer().forward(
        triangles=query_triangles,
        points=query_points,
        return_dist=False,
        return_proj=False,
        return_bary=True,
        min_only=True,
    )

    min_faceind = query_faceinds[argmin_triangle]  # .clone().detach()
    min_bary = bary_coords

    return min_faceind, min_bary


class PointsTriangleProjLayer(nn.Module):
    """
    Torch implementation of the projection of points a set of triangles in p-dimension.
    Face pre-selection must be done beforehand, this is bruteforce.

    .. note::
        The notations (and the algorithm) are inspired from:
            [1] "David Eberly, 'Distance Between Point and Triangle in 3D', Geometric Tools, LLC, (1999)"
    """

    def __init__(self):
        # pass
        super().__init__()

    def get_base_regions(self, s, t, det):
        region_t_neg = t < 0
        region_s_neg = s < 0

        # s + t <= det
        region_0345 = s + t <= det  # (n, m) with (m1) True values
        region_126 = ~region_0345  # (n, m) with (m-m1) True values

        # s < 0 & s + t <= det
        region_34 = region_0345 & region_s_neg  # (n, m) with (m11) True values
        region_05 = region_0345 & ~region_34  # (n, m) with (m-m11) True values

        # t < 0 | (s + t <= det) and (s < 0)
        region_4 = region_34 & region_t_neg
        region_3 = region_34 & ~region_4

        # t < 0 | s + t <= det and (s >= 0)
        region_5 = region_05 & region_t_neg
        region_0 = region_05 & ~region_5

        # s < 0 | s + t > det
        region_2 = region_126 & region_s_neg
        region_16 = region_126 & ~region_2

        # t < 0 | (s + t > det) and (s > 0)
        region_6 = region_16 & region_t_neg
        region_1 = region_16 & ~region_6

        return (region_0, region_1, region_2, region_3, region_4, region_5, region_6)

    def process_r4(self, a, c, d, e, f, s, t, verbose=False):

        final_s = th.zeros_like(a)
        final_t = th.zeros_like(a)
        final_dists = th.zeros_like(a)

        region_4_1 = d < 0
        region_4_2 = ~region_4_1

        region_4_11 = region_4_1 & (-d >= a)
        region_4_12 = region_4_1 & ~region_4_11

        region_4_21 = region_4_2 & (e >= 0)
        region_4_22 = region_4_2 & ~region_4_21

        region_4_221 = region_4_22 & (-e >= c)
        region_4_222 = region_4_22 & ~region_4_221

        # Region 4.1
        # final_t[region_4_1] = 0
        # Region 4.1.1
        final_s[region_4_11] = 1.0
        final_dists[region_4_11] = (
            a[region_4_11] + 2.0 * d[region_4_11] + f[region_4_11]
        )
        # Region 4.1.2
        # print('R4', (-d[region_4_12] / a[region_4_12]).max(), (-d[region_4_12] / a[region_4_12]).min())
        final_s[region_4_12] = -d[region_4_12] / a[region_4_12]
        final_dists[region_4_12] = d[region_4_12] * s[region_4_12] + f[region_4_12]

        # Region 4.2
        # final_s[region_4_2] = 0  # Useless already done
        # Region 4.2.1
        # final_t[region_4_21] = 0
        final_dists[region_4_21] = f[region_4_21]
        # Regions 4.2.2
        # Region 4.2.2.1
        final_t[region_4_221] = 1
        final_dists[region_4_221] = (
            c[region_4_221] + 2.0 * e[region_4_221] + f[region_4_221]
        )
        # Region 4.2.2.2
        # print('R4.2', (-e[region_4_222] / c[region_4_222]).max(), (-e[region_4_222] / c[region_4_222]).min())
        final_t[region_4_222] = -e[region_4_222] / c[region_4_222]
        final_dists[region_4_222] = e[region_4_222] * t[region_4_222] + f[region_4_222]
        # print('R4 final', final_s.max(), final_t.max())
        return final_s, final_t, final_dists

    def process_r3(self, a, c, e, f, verbose=False):
        final_s = th.zeros_like(a)
        final_t = th.zeros_like(a)
        final_dists = th.zeros_like(a)

        region_3_1 = e >= 0
        region_3_2 = ~region_3_1

        region_3_21 = region_3_2 & (-e >= c)
        region_3_22 = region_3_2 & ~region_3_21

        # Region 3
        # final_s[region_3] = 0
        # Region 3.1
        # final_t[region_3_1] = 0
        final_dists[region_3_1] = f[region_3_1]
        # Region 3.2
        # Region 3.2.1
        final_t[region_3_21] = 1
        final_dists[region_3_21] = (
            c[region_3_21] + 2.0 * e[region_3_21] + f[region_3_21]
        )
        # Region 3.2.2
        final_t[region_3_22] = -e[region_3_22] / c[region_3_22]
        final_dists[region_3_22] = (
            e[region_3_22] * final_t[region_3_22] + f[region_3_22]
        )  # -e*t ????

        return final_s, final_t, final_dists

    def process_r5(self, a, d, f, verbose=False):
        final_s = th.zeros_like(a)
        final_t = th.zeros_like(a)
        final_dists = th.zeros_like(a)

        region_5_1 = d >= 0
        region_5_2 = ~region_5_1

        region_5_21 = region_5_2 & (-d >= a)
        region_5_22 = region_5_2 & ~region_5_21

        # Region 5
        # final_t[region_5] = 0
        # Region 5.1
        # final_s[region_5_1] = 0
        final_dists[region_5_1] = f[region_5_1]
        # Region 5.2
        # Region 5.2.1
        final_s[region_5_21] = 1
        final_dists[region_5_21] = (
            a[region_5_21] + 2.0 * d[region_5_21] + f[region_5_21]
        )
        # Region 5.2.2
        final_s[region_5_22] = -d[region_5_22] / a[region_5_22]
        final_dists[region_5_22] = (
            d[region_5_22] * final_s[region_5_22] + f[region_5_22]
        )

        return final_s, final_t, final_dists

    def process_r0(self, a, b, c, d, e, f, s, t, det, verbose=False):
        # final_s = th.zeros_like(a)
        # final_t = th.zeros_like(a)
        # final_dists = th.zeros_like(a)

        invDet = 1.0 / th.clamp(det, min=1e-6)
        # print('Det', det.min(), invDet.max())
        final_s = s * invDet
        final_t = t * invDet
        final_dists = (
            final_s * (a * final_s + b * final_t + 2.0 * d)
            + final_t * (b * final_s + c * final_t + 2.0 * e)
            + f
        )

        return final_s, final_t, final_dists

    def process_r2(self, a, b, c, d, e, f, verbose=False):
        final_s = th.zeros_like(a)
        final_t = th.zeros_like(a)
        final_dists = th.zeros_like(a)

        b_plus_d = b + d
        c_plus_e = c + e

        numer = c_plus_e - b_plus_d
        denom = a - 2.0 * b + c

        region_2_1 = c_plus_e > b_plus_d
        region_2_2 = ~region_2_1

        region_2_11 = region_2_1 & (numer >= denom)
        region_2_12 = region_2_1 & ~region_2_11

        region_2_21 = region_2_2 & (c_plus_e <= 0.0)
        region_2_22 = region_2_2 & ~region_2_21

        region_2_221 = region_2_22 & (e >= 0.0)
        region_2_222 = region_2_22 & ~region_2_221

        # Region 2
        #   Region 2.1
        #       Region 2.1.1
        final_s[region_2_11] = 1
        # final_t[region_2_11] = 0
        final_dists[region_2_11] = (
            a[region_2_11] + 2.0 * d[region_2_11] + f[region_2_11]
        )
        #       Region 2.1.2
        final_s[region_2_12] = numer[region_2_12] / denom[region_2_12]
        final_t[region_2_12] = 1 - final_s[region_2_12]
        final_dists[region_2_12] = (
            final_s[region_2_12]
            * (
                a[region_2_12] * final_s[region_2_12]
                + b[region_2_12] * final_t[region_2_12]
                + 2 * d[region_2_12]
            )
            + final_t[region_2_12]
            * (
                b[region_2_12] * final_s[region_2_12]
                + c[region_2_12] * final_t[region_2_12]
                + 2 * e[region_2_12]
            )
            + f[region_2_12]
        )
        #   Region 2.2
        # final_s[region_2_2] = 0.
        #       Region 2.2.1
        final_t[region_2_21] = 1
        final_dists[region_2_21] = (
            c[region_2_21] + 2.0 * e[region_2_21] + f[region_2_21]
        )
        #       Region 2.2.2
        #           Region 2.2.2.1
        # final_t[region_2_221] = 0.
        final_dists[region_2_221] = f[region_2_221]
        #           Region 2.2.2.2
        final_t[region_2_222] = -e[region_2_222] / c[region_2_222]
        final_dists[region_2_222] = (
            e[region_2_222] * final_t[region_2_222] + f[region_2_222]
        )

        return final_s, final_t, final_dists

    def process_r6(self, a, b, c, d, e, f, verbose=False):
        final_s = th.zeros_like(a)
        final_t = th.zeros_like(a)
        final_dists = th.zeros_like(a)

        tmp0 = b + e
        tmp1 = a + d

        region_6_1 = tmp1 > tmp0
        region_6_2 = ~region_6_1

        numer = tmp1 - tmp0
        denom = a - 2.0 * b + c

        region_6_11 = region_6_1 & (numer >= denom)
        region_6_12 = region_6_1 & ~region_6_11

        region_6_21 = region_6_2 & (tmp1 <= 0.0)
        region_6_22 = region_6_2 & ~region_6_21

        region_6_221 = region_6_22 & (d >= 0.0)
        region_6_222 = region_6_22 & ~region_6_221

        # Region 6
        #   Region 6.1
        #       Region 6.1.1
        final_t[region_6_11] = 1
        # final_s[region_6_11] = 0
        final_dists[region_6_11] = (
            c[region_6_11] + 2.0 * e[region_6_11] + f[region_6_11]
        )
        #       Region 6.1.2
        final_t[region_6_12] = numer[region_6_12] / denom[region_6_12]
        final_s[region_6_12] = 1 - final_t[region_6_12]
        final_dists[region_6_12] = (
            final_s[region_6_12]
            * (
                a[region_6_12] * final_s[region_6_12]
                + b[region_6_12] * final_t[region_6_12]
                + 2.0 * d[region_6_12]
            )
            + final_t[region_6_12]
            * (
                b[region_6_12] * final_s[region_6_12]
                + c[region_6_12] * final_t[region_6_12]
                + 2.0 * e[region_6_12]
            )
            + f[region_6_12]
        )
        #   Region 6.2
        # final_t[region_6_2] = 0.
        #       Region 6.2.1
        final_s[region_6_21] = 1
        final_dists[region_6_21] = (
            a[region_6_21] + 2.0 * d[region_6_21] + f[region_6_21]
        )
        #       Region 6.2.2
        #           Region 6.2.2.1
        # final_s[region_6_221] = 0.
        final_dists[region_6_221] = f[region_6_221]
        #           Region 6.2.2.2
        final_s[region_6_222] = -d[region_6_222] / a[region_6_222]
        final_dists[region_6_222] = (
            d[region_6_222] * final_s[region_6_222] + f[region_6_222]
        )

        return final_s, final_t, final_dists

    def process_r1(self, a, b, c, d, e, f, verbose=False):
        final_s = th.zeros_like(a)
        final_t = th.zeros_like(a)
        final_dists = th.zeros_like(a)

        numer = c + e - b - d
        denom = a - 2.0 * b + c

        region_1_1 = numer <= 0
        region_1_2 = ~region_1_1

        region_1_21 = region_1_2 & (numer >= denom)
        region_1_22 = region_1_2 & ~region_1_21

        # Region 1
        #   Region 1.1
        # final_s[region_1_1] = 0
        final_t[region_1_1] = 1
        final_dists[region_1_1] = c[region_1_1] + 2.0 * e[region_1_1] + f[region_1_1]
        #   Region 1.2
        #       Region  1.2.1
        final_s[region_1_21] = 1
        # final_t[region_1_21] = 0
        final_dists[region_1_21] = (
            a[region_1_21] + 2.0 * d[region_1_21] + f[region_1_21]
        )
        #       Region 1.2.2
        final_s[region_1_22] = numer[region_1_22] / denom[region_1_22]
        final_t[region_1_22] = 1 - final_s[region_1_22]
        final_dists[region_1_22] = (
            final_s[region_1_22]
            * (
                a[region_1_22] * final_s[region_1_22]
                + b[region_1_22] * final_t[region_1_22]
                + 2.0 * d[region_1_22]
            )
            + final_t[region_1_22]
            * (
                b[region_1_22] * final_s[region_1_22]
                + c[region_1_22] * final_t[region_1_22]
                + 2.0 * e[region_1_22]
            )
            + f[region_1_22]
        )

        return final_s, final_t, final_dists

    def forward(
        self,
        triangles=None,
        points=None,
        min_only=True,
        return_bary=True,
        return_dist=True,
        return_proj=True,
        verbose=False,
    ):
        """

        returns in order: [final_dists, projections, barycentric_coordinates, argmin_proj]

        - If return_dist is True, output contains the distances
        - If return_proj is True, output contains the projections
        - If return_bary is True, output contains the barycentric coordinates
        - If min_only is True, output contains the index of the closest triangle

        Parameters
        ----------------------------
        triangles :
            (m, 3, p) coordinates of the mesh triangles
        points    :
            (n, p) coordinates of the points to project
        min_only  : bool
             whether to return results only for closest point on the triangle
        return_bary : bool
            whether to return the barycentric coordinates
        return_dist : bool
            whether to return the distances
        return_proj : bool
            whether to return the projections
        verbose   : bool
            whether to print debug information

        Returns
        ----------------------------
        final_dists :
            (n,) or (n,m) distances between the points and the triangles. (n,) if min_only is True
        projections :
            (n,p) or (n,m,p) projections of the points on the triangles. (n,p) if min_only is True
        barycentric_coordinates :
            (n,3) or (n,m,3) barycentric coordinates of the projections. (n,3) if min_only is True
        argmin_proj :
            (n,) index of the closest triangle. Only if min_only is True
        """

        n_points = points.shape[0]
        n_triangles = triangles.shape[0]

        final_s = th.zeros(n_points, n_triangles, device=points.device)  # (n,m)
        final_t = th.zeros(n_points, n_triangles, device=points.device)  # (n,m)
        final_dists = th.zeros(n_points, n_triangles, device=points.device)  # (n, m)

        bases = triangles[:, 0]  # (m,p)
        axis1 = triangles[:, 1] - bases  # (m,p)
        axis2 = triangles[:, 2] - bases  # (m,p)

        diff = bases[None, :] - points[:, None, :]  # (n,m,p)

        #  Precompute quantities

        a = (
            th.einsum("ij,ij->i", axis1, axis1).unsqueeze(0).expand(points.shape[0], -1)
        )  # (n, m)
        b = (
            th.einsum("ij,ij->i", axis1, axis2).unsqueeze(0).expand(points.shape[0], -1)
        )  # (n, m)
        c = (
            th.einsum("ij,ij->i", axis2, axis2).unsqueeze(0).expand(points.shape[0], -1)
        )  # (n, m)
        d = th.einsum("ij,nij->ni", axis1, diff)  # (n, m,)
        e = th.einsum("ij,nij->ni", axis2, diff)  # (n, m,)
        f = th.einsum("nij,nij->ni", diff, diff)  # (n, m,)

        det = a * c - b**2  # (n, m)
        s = b * e - c * d  # (n, m)
        t = b * d - a * e  # (n, m)

        r0, r1, r2, r3, r4, r5, r6 = self.get_base_regions(s, t, det)

        final_s[r4], final_t[r4], final_dists[r4] = self.process_r4(
            a[r4], c[r4], d[r4], e[r4], f[r4], s[r4], t[r4], verbose=verbose
        )
        final_s[r3], final_t[r3], final_dists[r3] = self.process_r3(
            a[r3], c[r3], e[r3], f[r3], verbose=verbose
        )
        final_s[r5], final_t[r5], final_dists[r5] = self.process_r5(
            a[r5], d[r5], f[r5], verbose=verbose
        )
        final_s[r0], final_t[r0], final_dists[r0] = self.process_r0(
            a[r0],
            b[r0],
            c[r0],
            d[r0],
            e[r0],
            f[r0],
            s[r0],
            t[r0],
            det[r0],
            verbose=verbose,
        )
        final_s[r2], final_t[r2], final_dists[r2] = self.process_r2(
            a[r2], b[r2], c[r2], d[r2], e[r2], f[r2], verbose=verbose
        )
        final_s[r6], final_t[r6], final_dists[r6] = self.process_r6(
            a[r6], b[r6], c[r6], d[r6], e[r6], f[r6], verbose=verbose
        )
        final_s[r1], final_t[r1], final_dists[r1] = self.process_r1(
            a[r1], b[r1], c[r1], d[r1], e[r1], f[r1], verbose=verbose
        )

        output = []
        if min_only:
            argmin_proj = final_dists.argmin(-1, keepdims=True)  # (n,1)

        if return_dist:
            if min_only:
                final_dists = th.gather(final_dists, -1, argmin_proj).squeeze(
                    dim=-1
                )  # (n,)
            final_dists[final_dists < 0] = 0
            final_dists = th.sqrt(final_dists)  # (n,) or # (n,m)

            output.append(final_dists)

        if min_only and (return_proj or return_bary):
            final_s = th.gather(final_s, -1, argmin_proj).squeeze(dim=-1)  # (n,)
            final_t = th.gather(final_t, -1, argmin_proj).squeeze(dim=-1)  # (n,)

        if min_only:
            argmin_proj = argmin_proj.squeeze(-1)

        if return_proj:
            if min_only:
                projections = (
                    bases[argmin_proj]
                    + final_s[:, None] * axis1[argmin_proj]
                    + final_t[:, None] * axis2[argmin_proj]
                )  # (n,p)
            else:
                projections = (
                    bases[None, ...]
                    + final_s[..., None] * axis1[None, ...]
                    + final_t[..., None] * axis2[None, ...]
                )  # (n,m,p)

            output.append(projections)

        if return_bary:
            if min_only:
                bary_coords = th.cat(
                    [
                        1 - final_s[:, None] - final_t[:, None],
                        final_s[:, None],
                        final_t[:, None],
                    ],
                    dim=-1,
                )  # (n,m,3)
            else:
                bary_coords = th.cat(
                    [
                        1 - final_s[..., None] - final_t[..., None],
                        final_s[..., None],
                        final_t[..., None],
                    ],
                    dim=-1,
                )  # (n,m,3)

            output.append(bary_coords)

        if min_only:
            output.append(argmin_proj)

        return output
