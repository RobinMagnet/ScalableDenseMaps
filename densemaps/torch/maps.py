import sys
import os
from pathlib import Path

import numpy as np

import torch as th
import torch.nn as nn

import scipy.sparse as sparse

try:
    import pykeops
    KEOPS_AVAILABLE = True
except ImportError:
    KEOPS_AVAILABLE = False

from .point_to_triangle import nn_query_precise_torch

from .nn_utils import nn_query, compute_sqdistmat

from ..numpy.point_to_triangle import barycentric_to_precise

class PointWiseMap:
    def __init__(self, tensor_names=None):
        self.tensor_names = []
        self._add_tensor_name(tensor_names)
        pass

    def _add_tensor_name(self, names):
        if type(names) is not list:
            names = [names]
        
        for name in names:
            if name not in self.tensor_names:
                self.tensor_names.append(name)
    
    def cpu(self):
        for name in self.tensor_names:
            setattr(self, name, getattr(self, name).cpu())
        return self
    
    def cuda(self):
        for name in self.tensor_names:
            setattr(self, name, getattr(self, name).cuda())
        return self
    
    def to(self, device):
        for name in self.tensor_names:
            setattr(self, name, getattr(self, name).to(device))
        return self

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
        map : (N2, N1) or (B, N2, N1)
        """
        super().__init__(tensor_names=["map"])

        self.map = map
    
    @property
    def shape(self):
        return self.map.shape

    @property
    def mT(self):
        return SparseMap(self.map.transpose(-1, -2))
    
    def pull_back(self, f):
        return self.map @ f

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
        super().__init__(tensor_names=["p2p_21"])

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
                # f_pb = f[th.arange(f.shape[0]).unsqueeze(1), self.p2p_21]
                f_pb = th.take_along_dim(f, self.p2p_21.unsqueeze(-1), dim=1)  # (B, n2, k)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
    
        return f_pb

    def get_nn(self):
        return self.p2p_21

    @property
    def mT(self):
        sparsemat = th.sparse_coo_tensor(th.stack([th.arange(self.n2, device=self.p2p_21.device), self.p2p_21]), th.ones_like(self.p2p_21).float(), (self.n2, self.n1)).coalesce()
        return SparseMap(sparsemat).mT
    
    def _to_np_sparse(self):
        assert self.p2p_21.ndim == 1, "Batched version not implemented yet."

        return sparse.csc_matrix((np.ones(self.p2p_21.shape[0]), (np.arange(self.n2), self.p2p_21.cpu().numpy())), shape=(self.n2, self.n1))

class PreciseMap(PointWiseMap):
    """
    Point to barycentric map, using vertex to face and barycentric coordinates.
    """
    def __init__(self, v2face_21, bary_coords, faces1):
        """
        Point to barycentric map from a set S2 to a surface S1.

        Parameters
        -------------------
        v2face_21 : (n2,) or (B, n2)
            Indices of the faces of S1 closest to each point of S2.
        bary_coords : (n2, 3) or (B, n2, 3)
            Barycentric coordinates of the points of S2 in the faces of S1.
        faces1 : (N1, 3)
            All the Faces of S1.
        """
        super().__init__(tensor_names=["v2face_21", "bary_coords", "faces1"])
        if v2face_21.ndim == 2:
            raise ValueError('Batched version not implemented yet.')
        
        self.v2face_21 = v2face_21  # (n2,) or (B, n2)
        self.bary_coords = bary_coords  # (N2, 3)  or (B, N2, 3)
        self.faces1 = faces1  # (N1, 3)

        self.n2 = self.v2face_21.shape[-1]
        self.n1 = self.faces1.max()+1

        self._nn_map = None
    
    @property
    def shape(self):
        return th.Size([self.n2, self.n1])
     
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
        if self.v2face_21.ndim == 1:
            if f.ndim == 1 or f.ndim == 2:
                f_selected = f[self.faces1[self.v2face_21]]  # (N2, 3, p) or (N2, 3)
                if f.ndim == 1:
                    f_pb = (self.bary_coords * f_selected).sum(1)
                else:
                    f_pb = (self.bary_coords.unsqueeze(-1) * f_selected).sum(1)
                    # print('Selected2', f_pb.max())

            elif f.ndim == 3:
                f_selected = f[: self.faces1[self.v2face_21]]  # (B, N2, 3, p)
                f_pb = (self.bary_coords.unsqueeze(0).unsqueeze(-1) * f_selected).sum(1)
        
        else:
            raise NotImplementedError('Batched version not implemented yet.')
    
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = th.take_along_dim(self.faces1[self.v2face_21],
                                             self.bary_coords.argmax(1, keepdims=True),
                                             1).squeeze(-1)
            self._add_tensor_name(["_nn_map"])
        
        return self._nn_map

    @property
    def mT(self):
        target_faces = self.faces1[self.v2face_21]  # (n2, 3)

        In = th.tile(th.arange(self.n2, device=self.v2face_21.device), (3,))  # (3*n2)
        Jn = th.concatenate([target_faces[:,0], target_faces[:,1], target_faces[:,2]])  # (3*n2)
        Sn = th.concatenate([self.bary_coords[:,0], self.bary_coords[:,1], self.bary_coords[:,2]])  # (3*n2)

        precise_map = th.sparse_coo_tensor(th.stack([In, Jn]), Sn, (self.n2, self.n1)).coalesce()
        return SparseMap(precise_map).mT

    def _to_np_sparse(self):
        return barycentric_to_precise(self.faces1.cpu().numpy(), self.v2face_21.cpu().numpy(), self.bary_coords.cpu().numpy())
class EmbP2PMap(P2PMap):
    """
    Point to point map, computed from embeddings.
    """
    def __init__(self, emb1, emb2):
        self.emb1 = emb1.contiguous()  # (N1, K) or (B, N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K) or (B, N2, K)

        p2p_21 = nn_query(self.emb1, self.emb2)
        
        super().__init__(p2p_21, n1=self.emb1.shape[-2])
        self._add_tensor_name(["emb1", "emb2", "p2p_21"])

class EmbPreciseMap(PreciseMap):
    """
    Point to barycentric map, computed from embeddings.
    """
    def __init__(self, emb1, emb2, faces1, clear_cache=True):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        
        v2face_21, bary_coords = nn_query_precise_torch(self.emb1, faces1, self.emb2, return_dist=False, batch_size=min(2000, emb2.shape[0]), clear_cache=clear_cache)

        # th.cuda.empty_cache()
        super().__init__(v2face_21, bary_coords, faces1)
        self._add_tensor_name(["emb1", "emb2"])

class KernelDenseDistMap(PointWiseMap):
    def __init__(self, log_matrix, lse_row=None, lse_col=None):
        super().__init__(tensor_names=["log_matrix"])
        self.log_matrix = log_matrix   # (..., N2, N1)
        self.lse_row = lse_row  # (..., N2)
        self.lse_col = lse_col  # (..., N1)

        self._nn_map = None
        self._inv_nn_map = None

        if lse_row is not None:
            self._add_tensor_name(["lse_row"])
        if lse_col is not None:
            self._add_tensor_name(["lse_col"])

    def _to_dense(self):
        if self.lse_row is None:
            self.lse_row = th.logsumexp(self.log_matrix, dim=-1)  # (..., N2)
        
        return th.exp(self.log_matrix - self.lse_row.unsqueeze(-1))
    
    def pull_back(self, f):
        if type(f) is KernelDenseDistMap:
            return self._to_dense() @ f._to_dense()

        return self._to_dense() @ f

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = self.log_matrix.argmax(-1)
            self._add_tensor_name(["_nn_map"])
        return self._nn_map

    @property
    def mT(self):
        obj = KernelDenseDistMap(self.log_matrix.transpose(-1,-2), lse_row=self.lse_col, lse_col=self.lse_row)
        obj._inv_nn_map, obj._nn_map = self._nn_map, self._inv_nn_map
        if obj._nn_map is not None:
            obj._add_tensor_name(["_nn_map"])
        if obj._inv_nn_map is not None:
            obj._add_tensor_name(["_inv_nn_map"])
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
            # self.emb1 = self.emb1 / th.linalg.norm(self.emb1, dim=-1, keepdim=True)  # (N1, p) or (B, N1, p)
            # self.emb2 = self.emb2 / th.linalg.norm(self.emb2, dim=-1, keepdim=True)  # (N2, p) or (B, N2, p)
            self.emb1 = nn.functional.normalize(self.emb1, p=2, dim=-1)  # (N1, p) or (B, N1, p)
            self.emb2 = nn.functional.normalize(self.emb2, p=2, dim=-1)  # (N2, p) or (B, N2, p)

        if dist_type == "sqdist":
            dist = compute_sqdistmat(self.emb2, self.emb1, normalized=normalize_emb)  # (N2, N1)  or (B, N2, N1)
        elif dist_type == "inner":
            dist = - self.emb2 @ self.emb1.transpose(-2, -1)  # (N2, N1)  or (B, N2, N1)

        self.dist_type = dist_type

        self.blur = th.ones(1, device=self.emb1.device)
        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            assert dist_type == "sqdist", "Normalization only supported for sqdist."
            with th.no_grad():
                self.blur = blur * th.sqrt(dist.max())

        log_matrix = - dist / (2 * th.square(self.blur))  # (N2, N1)  or (B, N2, N1)

        super().__init__(log_matrix)
        self._add_tensor_name(["emb1", "emb2", "blur"])

class KernelDistMap(PointWiseMap):
    """
    Map of the the shape exp(- ||X_i - Y_j||_2^2 / blur**2)). Normalized per row.
    """
    def __init__(self, emb1, emb2, normalize=False, blur=None, normalize_emb=False, dist_type="sqdist"):
        """
        
        Parameters
        -------------------
        emb1 : (N1, K)
        emb2 : (N2, K)
        normalize : bool
            Normalize the blur by the maximum distance.
        blur : float
            Standard deviation of the Gaussian kernel.
        """
        super().__init__(tensor_names=["emb1", "emb2", "blur"])
        assert dist_type in ["sqdist", "inner"], "Invalid distance type."
        self.dist_type = dist_type

        
        self.emb1 = emb1.contiguous()  # (N1, K)  or (B, N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)  or (B, N2, K)
        if normalize_emb:
            # self.emb1 = self.emb1 / th.linalg.norm(self.emb1, dim=-1, keepdim=True)  # (N1, p) or (B, N1, p)
            # self.emb2 = self.emb2 / th.linalg.norm(self.emb2, dim=-1, keepdim=True)  # (N2, p) or (B, N2, p)
            self.emb1 = nn.functional.normalize(self.emb1, p=2, dim=-1)  # (N1, p) or (B, N1, p)
            self.emb2 = nn.functional.normalize(self.emb2, p=2, dim=-1)  # (N2, p) or (B, N2, p)


        self.blur = th.ones(1, device=self.emb1.device)
        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            assert dist_type == "sqdist", "Normalization only supported for sqdist."
            with th.no_grad():
                self.blur = self.blur * th.sqrt(self.get_maxsqnorm())

        self.n1 = self.emb1.shape[-2]
        self.n2 = self.emb2.shape[-2]
        self._nn_map = None

    @property
    def shape(self):
        return th.Size([self.n2, self.n1])
    
    def get_maxsqnorm(self):
        formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({self.emb1.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({self.emb2.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='Max',
                    axis=0)

        max_dist = formula(self.emb1, self.emb2).max().squeeze()

        return max_dist
    
    def get_pull_back_formula(self, dim):
        """
        B, N1, 1 -> B, N2, 1
        """

        f = pykeops.torch.Vj(0, dim)  # (B, 1, N1, p)
        emb1_j = pykeops.torch.Vj(1, self.emb1.shape[-1])  # (1, 1, N1, K)
        emb2_i = pykeops.torch.Vi(2, self.emb1.shape[-1])  # (1, N2, 1, K)
        sqblur = pykeops.torch.Pm(3, 1)  # (B, 1)

        if self.dist_type == "sqdist":
            dist = - emb2_i.sqdist(emb1_j) / sqblur
        elif self.dist_type == "inner":
            dist = (emb2_i | emb1_j) / sqblur

        return dist.sumsoftmaxweight(f, axis=1)  # (B, N2, p)

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
        
        n_func = f.shape[-1] if f.ndim > 1 else 1
        pull_back_formula = self.get_pull_back_formula(n_func)

        sqblur = 2*th.square(self.blur)

        if f.ndim == 1:
            f_in = f.unsqueeze(-1).contiguous()  # (N1, 1)
            if self.emb1.ndim == 3:
                f_in = f_in.unsqueeze(0)  # (1, N1, 1)
            f_pb = pull_back_formula(f_in, self.emb1, self.emb2, sqblur).squeeze(-1)  # (N2, )
        
        elif f.ndim == 2:
            f_input = f.contiguous()  # (N1, p)
            if self.emb1.ndim == 3:
                f_input = f_input.unsqueeze(0)  # (1, N1, p)
            f_pb = pull_back_formula(f_input, self.emb1, self.emb2, sqblur) # (N2, p)
        
        elif f.ndim == 3:
            f_input = f.contiguous()
            if self.emb1.ndim == 2:
                f_pb = pull_back_formula(f, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur) # (B, N2, p)
            else:
                f_pb = pull_back_formula(f_input, self.emb1, self.emb2, sqblur)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
        
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = nn_query(self.emb1, self.emb2)
            self._add_tensor_name(["_nn_map"])
        
        return self._nn_map
    
    @property
    def mT(self):
        invmap = KernelDistMap(self.emb2, self.emb1, blur=self.blur, normalize=False, dist_type=self.dist_type)
        invmap.blur = self.blur
        return invmap
