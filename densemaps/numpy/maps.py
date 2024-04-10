import sys
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.special import logsumexp

# from .point_to_triangle import nn_query_precise_torch

from .nn_utils import knn_query, compute_sqdistmat
from .point_to_triangle import nn_query_precise_np, barycentric_to_precise


class PointWiseMap:
    def __init__(self, array_names=None):
        self.array_names = []
        self._add_array_name(array_names)
        pass

    def _add_array_name(self, names):
        if type(names) is not list:
            names = [names]
        
        for name in names:
            if name not in self.array_names:
                self.array_names.append(name)

    def pull_back(self, f):
        raise NotImplementedError

    def get_nn(self):
        raise NotImplementedError
    
    def __matmul__(self, other):
        return self.pull_back(other)
    
    @property
    def mT(self):
        raise NotImplementedError

class SparseMap(PointWiseMap):
    def __init__(self, map):
        """
        Map represented by a sparse matrix

        Parameters
        -------------------
        map : (N2, N1) or (N2, N1)
        """
        super().__init__(array_names=["map"])

        self.map = map  # (N2, N1)
    @property
    def shape(self):
        return self.map.shape

    @property
    def mT(self):
        return SparseMap(self.map.T)
    
    def pull_back(self, f):
        return self.map @ f

    def get_nn(self):
        return self.map.argmax(-1)  # (N2, )
    
    def _to_sparse(self):
        return self.map

class P2PMap(PointWiseMap):
    """
    Point to point map, as an array or tensor of shape (n2,)
    """
    def __init__(self, p2p_21, n1=None):
        """
        Point to point map from a set S2 to a set S1.

        Parameters
        -------------------
        p2p_21 : (n2,) or (B, n2)
        n1 : int or None
            Number of points in S1. If None, n1 = p2p.max()+1
        """
        super().__init__(array_names=["p2p_21"])

        self.p2p_21 = p2p_21  # (n2, ) or (B, n2)

        self.n2 = self.p2p_21.shape[-1]
        self.n1 = n1

        self.max_ind = self.p2p.max().item() if n1 is None else n1-1
    
    @property
    def shape(self):
        return self.p2p_21.shape

    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """


        if (f.ndim==1 and f.shape[-1] <= self.max_ind):
            raise ValueError(f'Function f doesn\'t have enough entries, need at least {1+self.max_ind} but only has {f.shape[-1]}')
        elif (f.ndim > 1 and f.shape[-2] <= self.max_ind):
            raise ValueError(f'Function f doesn\'t have enough entries, need at least {1+self.max_ind} but only has {f.shape[-2]}')
        
        if f.ndim == 3 and self.p2p_21.ndim == 2:
            assert f.shape[0] == self.p2p_21.shape[0], "Batch size of f and p2p_21 should match"
        
        if f.ndim == 1 or f.ndim == 2:  # (N1,) or (N1, p)
            f_pb = f[self.p2p_21]  # (n2, p) or (B, n2, p)
        elif f.ndim == 3:
            if self.p2p_21.ndim == 1:
                f_pb = f[:, self.p2p_21]  # (B, n2, k)
            else:
                f_pb = f[np.arange(f.shape[0])[:,None], self.p2p_21]
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
    
        return f_pb

    def get_nn(self):
        return self.p2p_21

    @property
    def mT(self):
        assert self.p2p_21.ndim == 1, "Batched version not implemented yet."
        # sparsemat = th.sparse_coo_tensor(th.stack([th.arange(self.n2), self.p2p_21]), th.ones_like(self.p2p_21).float(), (self.n2, self.n1)).coalesce()
        P21 = self._to_sparse()
        return SparseMap(P21.T)
    
    def _to_sparse(self):
        return sparse.csc_matrix((np.ones_like(self.p2p_21), (np.arange(self.n2), self.p2p_21)), shape=(self.n2, self.n1))

class PreciseMap(SparseMap):
    """
    Point to barycentric map, using vertex to face and barycentric coordinates.
    """
    def __init__(self, v2face_21, bary_coords, faces1):
        """
        Point to barycentric map from a set S2 to a surface S1.

        Parameters
        -------------------
        v2face_21 : (n2,)
            Indices of the faces of S1 closest to each point of S2.
        bary_coords : (n2, 3) 
            Barycentric coordinates of the points of S2 in the faces of S1.
        faces1 : (N1, 3)
            All the Faces of S1.
        """
        # super().__init__(array_names=["v2face_21", "bary_coords", "faces1"])
        if v2face_21.ndim == 2:
            raise ValueError('Batched version not implemented yet.')
        
        # self.v2face_21 = v2face_21  # (n2,) or (B, n2)
        # self.bary_coords = bary_coords  # (N2, 3)  or (B, N2, 3)
        # self.faces1 = faces1  # (N1, 3)

        # self.n2 = self.v2face_21.shape[-1]
        # self.n1 = self.faces1.max()+1

        sparse_map = barycentric_to_precise(faces1, v2face_21, bary_coords, n_vertices=None)  # (n2, n1)
        super().__init__(sparse_map)
        self._nn_map = None
    
    @property
    def shape(self):
        return self.sparse_map.shape

class EmbP2PMap(P2PMap):
    """
    Point to point map, computed from embeddings.
    """
    def __init__(self, emb1, emb2, n_jobs=1):
        self.emb1 = emb1 # (N1, K) or (N1, K)
        self.emb2 = emb2  # (N2, K) or (N2, K)
        self.n_jobs = n_jobs

        p2p_21 = knn_query(self.emb1, self.emb2, n_jobs=n_jobs)  # (N2, )
        
        super().__init__(p2p_21, n1=self.emb1.shape[-2])
        self._add_array_name(["emb1", "emb2", "p2p_21"])

class EmbPreciseMap(PreciseMap):
    """
    Point to barycentric map, computed from embeddings.
    """
    def __init__(self, emb1, emb2, faces1, n_jobs=1):
        self.emb1 = emb1  # (N1, K)
        self.emb2 = emb2  # (N2, K)

        
        v2face_21, bary_coords = nn_query_precise_np(self.emb1, faces1, self.emb2, return_dist=False, batch_size=min(2000, emb2.shape[0]), n_jobs=n_jobs)

        # th.cuda.empty_cache()
        super().__init__(v2face_21, bary_coords, faces1)
        self._add_array_name(["emb1", "emb2"])

class KernelDenseDistMap(PointWiseMap):
    def __init__(self, log_matrix, lse_row=None, lse_col=None):
        super().__init__(array_names=["log_matrix"])
        self.log_matrix = log_matrix   # (..., N2, N1)
        self.lse_row = lse_row  # (..., N2)
        self.lse_col = lse_col  # (..., N1)

        self._nn_map = None
        self._inv_nn_map = None

        if lse_row is not None:
            self._add_array_name(["lse_row"])
        if lse_col is not None:
            self._add_array_name(["lse_col"])

    def _to_dense(self):
        if self.lse_row is None:
            self.lse_row = logsumexp(self.log_matrix, dim=-1)  # (..., N2)
        
        return np.exp(self.log_matrix - self.lse_row[...,None])
    
    def pull_back(self, f):
        if type(f) is KernelDenseDistMap:
            return self._to_dense() @ f._to_dense()

        return self._to_dense() @ f

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = self.log_matrix.argmax(-1)
            self._add_array_name(["_nn_map"])
        return self._nn_map

    @property
    def mT(self):
        obj = KernelDenseDistMap(self.log_matrix.T, lse_row=self.lse_col, lse_col=self.lse_row)
        obj._inv_nn_map, obj._nn_map = self._nn_map, self._inv_nn_map
        if obj._nn_map is not None:
            obj._add_array_name(["_nn_map"])
        if obj._inv_nn_map is not None:
            obj._add_array_name(["_inv_nn_map"])
        return obj

    @property
    def shape(self):
        return self.log_matrix.shape

class EmbKernelDenseDistMap(KernelDenseDistMap):
    def __init__(self, emb1, emb2, blur=None, normalize=False, normalize_emb=False, dist_type="sqdist"):
        """

        Parameters
        -------------------
        emb1 : (N1, K)
        emb2 : (N2, K)
        blur : float
            Standard deviation of the Gaussian kernel.
        normalize : bool
            Normalize the blur by the maximum distance.
        dist_type : {"sqdist", "inner"}
            Type of distance to use.
        """

        assert dist_type in ["sqdist", "inner"], "Invalid distance type."

        self.emb1 = emb1  # (N1, p) or (B, N1, p)
        self.emb2 = emb2  # (N2, p) or (B, N2, p)
        if normalize_emb:
            norm1 = np.linalg.norm(self.emb1, axis=-1, keepdims=True)  # (N1, 1) or (B, N1, 1)
            norm2 = np.linalg.norm(self.emb2, axis=-1, keepdims=True)
            self.emb1 = self.emb1 / np.clip(norm1, 1e-6, None)  # (N1, p) or (B, N1, p)
            self.emb2 = self.emb2 / np.clip(norm2, 1e-6, None)  # (N2, p) or (B, N2, p)


        if dist_type == "sqdist":
            dist = compute_sqdistmat(self.emb2, self.emb1, normalized=normalize_emb)  # (N2, N1)  or (B, N2, N1)
        elif dist_type == "inner":
            dist = - self.emb2 @ self.emb1.permute(-2, -1)  # (N2, N1)  or (B, N2, N1)

        self.dist_type = dist_type

        self.blur = 1 if blur is None else blur

        if normalize:
            assert dist_type == "sqdist", "Normalization only supported for sqdist."
            self.blur = blur * np.sqrt(dist.max())

        log_matrix = - dist / (2 * self.blur**2)  # (N2, N1)  or (B, N2, N1)

        super().__init__(log_matrix)
        self._add_array_name(["emb1", "emb2", "blur"])