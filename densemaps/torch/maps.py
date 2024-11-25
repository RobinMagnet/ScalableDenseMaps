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
    r"""
    Root class representing a pointwise map. Not supposed to be instanciated in itself.

    A pointwise, denoted $P : S_2 \to S_1$, is a map that associates to each point x in $S_2$ a point $P(x)$ in $S_1$.
    It can usually be represented as a $n_2 \times n_1$ matrix, where $n_2$ is the number of points in $S_2$ and $n_1$ the number of points in $S_1$.

    Given a pointwise map $P$, the pullback of a function $f : S_1 \to R$ is a function $f_{pb} : S_2 \to R$ defined by $f_{pb}(x) = f(P(x))$.
    In practice it can easily be computed by matrix multiplication: $f_{pb} = P f$.

    In practice, we usually don't need to use the exact values inside $P$, btu rather only care about multiplying with some functions,
    extracting maximal values per-row or per-column, or summing on rows or columns.

    In torch, all maps can be represented in a batched way, with an additional dimension at the beginning of the tensor.

    Attributes
    -------------------
    tensor_names : list of str
        Names of the tensors stored in the object. Used to transfer to GPU.

    """

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

    @property
    def shape(self):
        """
        Shape of the map.

        Returns
        -------------------
        shape : tuple
            Depends on the representation
        """
        raise NotImplementedError("Shape not implemented")

    @property
    def ndim(self):
        """
        Number of dimensions of the map.

        Returns
        -------------------
        ndim : int
            Number of dimensions
        """
        return len(self.shape)

    def pull_back(self, f):
        """Pull back a function $f$.
        Three possibilities:

        #. f is a function on S1, of shape (N1,), then the output is a function on S2, of shape (N2,)
        #. f represents multiple function on S1, of shape (N1, p), then the output is a function on S2, of shape (N2, p)
        #. f represents a batch multiple function on S1, of shape (B, N1, p), then the output is a function on S2, of shape (B, N2, p)

        Note tht the case where f is a batch of a single function (B, N1) is not supported, and one should then use `f[..., None]`

        Parameters
        -------------------
        f : torch.Tensor
            (N1,), (N1, p) or (B, N1, p)

        Returns
        -------------------
        f_pb : torch.Tensor
            (N2,), (N2, p) or (B, N2, p)
        """
        raise NotImplementedError

    def get_nn(self):
        """Ouptputs the nearest neighbor map.
        The nearest neighbor map is the map that associates to each point of S2 the index of the closest point in S1.

        Returns
        -------------------
        nn_map : torch.Tensor
            (N2,) or (B, N2) depending on the representation
        """
        raise NotImplementedError

    def _sparse_matmul(self, x, y):
        """Matrix multiplication of two (potentially batched) sparse matrices."""
        if x.ndim > 3:
            raise NotImplementedError(
                "Multi Batched multiplication not implemented yet."
            )
        elif x.ndim == 3:
            if y.ndim == 3:
                return th.stack([x[i] @ y[i] for i in range(x.shape[0])])
            else:
                return th.stack([x[i] @ y for i in range(x.shape[0])])
        elif y.ndim == 3:
            return th.stack([x @ y[i] for i in range(y.shape[0])])

    def __matmul__(self, other):
        """Matrix multiplication of two maps."""
        if issubclass(type(other), PointWiseMap):
            other_sparse = other._to_sparse()
            self_sparse = self._to_sparse()

            if (
                other_sparse.layout is th.sparse_coo
                and self_sparse.layout is th.sparse_coo
            ):
                prod = self._sparse_matmul(self_sparse, other_sparse)
                return SparseMap(prod)
            elif self_sparse.layout is th.strided and other_sparse.layout is th.strided:
                prod = self_sparse @ other_sparse
                return SparseMap(prod)
            elif other_sparse.layout is th.sparse_coo:
                if other_sparse.ndim == 3:
                    prod = th.stack(
                        [
                            self_sparse @ other_sparse[i]
                            for i in range(other_sparse.shape[0])
                        ]
                    )
                else:
                    prod = self_sparse @ other_sparse
            elif self_sparse.layout is th.sparse_coo:
                if self_sparse.ndim == 3:
                    prod = th.stack(
                        [
                            self_sparse[i] @ other_sparse
                            for i in range(self_sparse.shape[0])
                        ]
                    )
                else:
                    prod = self_sparse @ other_sparse
            else:
                raise ValueError("Not implemented yet.")

            return SparseMap(prod)

        return self.pull_back(other)

    @property
    def mT(self):
        """Returns the transpose of the map.

        Returns the transpose of the matrix representation of the map.

        Returns
        -------------------
        map_t : PointWiseMap
            Transpose map
        """
        raise NotImplementedError

    def _to_sparse(self):
        """Returns sparse (or dense) tensor representation of the map.

        Returns
        -------------------
        map_sparse : torch.sparse_coo_tensor or torch.Tensor
            Sparse or dense representation of the map
        """
        raise NotImplementedError


class SparseMap(PointWiseMap):
    """Map represented by a sparse matrix.

    The sparse matrix is of size `(N2, N1)`.

    Can actually represent a densemap too, but it's not recommended.

    Parameters
    -------------------
    map : scipy.sparse.csr_matrix
        (N2, N1) or (B, N2, N1)
    """

    def __init__(self, map):

        super().__init__(tensor_names=["map"])

        self.map = map

    @property
    def shape(self):
        """
        Shape of the map.

        Returns
        -------------------
        shape : tuple
            returns (N2, N1) or (B, N2, N1)
        """
        return self.map.shape

    @property
    def mT(self):
        """Returns the transpose of the map.

        Returns the transpose of the matrix representation of the map.

        Returns
        -------------------
        map_t : SparseMap
            Transpose map
        """
        return SparseMap(self.map.transpose(-1, -2))

    def pull_back(self, f):
        """Pull back a function $f$.
        Three possibilities:

        #. f is a function on S1, of shape (N1,), then the output is a function on S2, of shape (N2,)
        #. f represents multiple function on S1, of shape (N1, p), then the output is a function on S2, of shape (N2, p)
        #. f represents a batch multiple function on S1, of shape (B, N1, p), then the output is a function on S2, of shape (B, N2, p)

        Note tht the case where f is a batch of a single function (B, N1) is not supported, and one should then use `f[..., None]`

        Parameters
        -------------------
        f : torch.Tensor
            (N1,), (N1, p) or (B, N1, p)

        Returns
        -------------------
        f_pb : torch.Tensor
            (N2,), (N2, p) or (B, N2, p)
        """
        if self.map.ndim == 3:
            if f.ndim < 3:
                f_pb = th.stack([self.map[i] @ f for i in range(self.map.shape[0])])
            else:
                assert (
                    f.shape[0] == self.map.shape[0]
                ), "Batch size of f and map should match"
                f_pb = th.stack([self.map[i] @ f[i] for i in range(f.shape[0])])
        else:
            f_pb = self.map @ f
        return f_pb

    def _to_sparse(self):
        return self.map


class P2PMap(PointWiseMap):
    r"""
    Point to point map from a set S2 to a set S1.
    Defined by a map $P_{21} : S2 \to S1$ or a tensor `p2p_21` of size `(n_2)`, where `p2p_21[i]` is the index of the point in S1 closest to the point i in S2.
    Batched versions are accepted

    Parameters
    -------------------
    p2p_21 : th.Tensor
        (n2,) or (B, n2)
    n1 : int or None
        Number of points in S1. If None, n1 = p2p.max()+1
    """

    def __init__(self, p2p_21, n1=None):
        super().__init__(tensor_names=["p2p_21"])

        self.p2p_21 = p2p_21  # (n2, ) or (B, n2)

        self.n2 = self.p2p_21.shape[-1]
        self._n1 = n1

        self.max_ind = self.p2p_21.max().item() if n1 is None else n1 - 1

    @property
    def n1(self):
        """
        Number of vertices on the first shape.
        Estimated if not provided as input.

        Returns
        -------------------
        n1 : int
        """
        return self._n1 if self._n1 is not None else self.max_ind + 1

    @property
    def shape(self):
        """
        Shape of the map.

        Returns
        -------------------
        shape : tuple
         return (n2,) or (B, n2)
        """
        return self.p2p_21.shape

    def pull_back(self, f):
        """Pull back a function $f$.
        Four possibilities:

        #. f is a function on S1, of shape (N1,), then the output is a function on S2, of shape (N2,) (or (B, N2,) if the map is batched)
        #. f represents multiple function on S1, of shape (N1, p), then the output is a function on S2, of shape (N2, p) (or (B, N2, p) if the map is batched)
        #. f represents a batch multiple function on S1, of shape (B, N1, p), then the output is a function on S2, of shape (B, N2, p)

        Note tht the case where f is a batch of a single function (B, N1) is not supported, and one should then use `f[..., None]`

        Parameters
        -------------------
        f : torch.Tensor
            (N1,), (N1, p) or (B, N1, p)

        Returns
        -------------------
        f_pb : torch.Tensor
            (N2,), (N2, p) or (B, N2, p)
        """

        # Check dimensions consistency
        if f.ndim == 1 and f.shape[-1] <= self.max_ind:
            raise ValueError(
                f"Function f doesn't have enough entries, need at least {1+self.max_ind} but only has {f.shape[-1]}"
            )
        elif f.ndim > 1 and f.shape[-2] <= self.max_ind:
            raise ValueError(
                f"Function f doesn't have enough entries, need at least {1+self.max_ind} but only has {f.shape[-2]}"
            )

        if f.ndim == 3 and self.p2p_21.ndim == 2:
            assert (
                f.shape[0] == self.p2p_21.shape[0]
            ), "Batch size of f and p2p_21 should match"

        # Compute pull back on different shape scenarios
        if f.ndim == 1 or f.ndim == 2:  # (N1,) or (N1, p)
            f_pb = f[self.p2p_21]  # (n2, p) or (B, n2, p)

        elif f.ndim == 3:
            if self.p2p_21.ndim == 1:
                f_pb = f[:, self.p2p_21]  # (B, n2, k)
            else:
                # f_pb = f[th.arange(f.shape[0]).unsqueeze(1), self.p2p_21]
                f_pb = th.take_along_dim(
                    f, self.p2p_21.unsqueeze(-1), dim=1
                )  # (B, n2, k)
        else:
            raise ValueError("Function is only dim 1, 2 or 3")

        return f_pb

    def get_nn(self):
        """Ouptputs the nearest neighbor map.
        The nearest neighbor map is the same as the input.

        Returns
        -------------------
        nn_map : torch.Tensor
            (N2,) or (B, N2) depending on the representation
        """
        return self.p2p_21

    @property
    def mT(self):
        if self.p2p_21.ndim == 1:
            map_mt = self._single_p2p_to_sparse(self.p2p_21).mT
        else:
            map_mt = th.stack(
                [
                    self._single_p2p_to_sparse(self.p2p_21[i]).T
                    for i in range(self.p2p_21.shape[0])
                ]
            )
        return SparseMap(map_mt)

    def _single_p2p_to_sparse(self, p2p):
        assert p2p.ndim == 1, "Only for non-batched version"
        return th.sparse_coo_tensor(
            th.stack([th.arange(self.n2, device=p2p.device), p2p]),
            th.ones_like(p2p).float(),
            (self.n2, self.n1),
        ).coalesce()

    def _to_sparse(self):
        if self.p2p_21.ndim == 1:
            return self._single_p2p_to_sparse(self.p2p_21)
        else:
            return th.stack(
                [
                    self._single_p2p_to_sparse(self.p2p_21[i])
                    for i in range(self.p2p_21.shape[0])
                ]
            )

    def _to_np_sparse(self):
        assert self.p2p_21.ndim == 1, "Batched version not implemented yet."

        return sparse.csc_matrix(
            (
                np.ones(self.p2p_21.shape[0]),
                (np.arange(self.n2), self.p2p_21.cpu().numpy()),
            ),
            shape=(self.n2, self.n1),
        )


class PreciseMap(PointWiseMap):
    """
    Point to barycentric map from a set S2 to a surface S1.
    Batched Version is not supported yet.

    Is represented as a sparse matrix of size `(N2, N1)`, where there are at most 3 non-zero entries per row, which sum to 1.

    .. note::
        Batched version is not supported yet.

    Parameters
    -------------------
    v2face_21 : torch.Tensor
        (n2,) Indices of the faces of S1 closest to each point of S2.
    bary_coords : torch.Tensor
        (n2, 3) Barycentric coordinates of the points of S2 in the faces of S1.
    faces1 :  torch.Tensor
        (N1, 3) All the Faces of S1.
    """

    def __init__(self, v2face_21, bary_coords, faces1):
        """
        Point to barycentric map from a set S2 to a surface S1.

        """
        super().__init__(tensor_names=["v2face_21", "bary_coords", "faces1"])
        if v2face_21.ndim == 2:
            raise ValueError("Batched version not implemented yet.")

        self.v2face_21 = v2face_21  # (n2,) or (B, n2)
        self.bary_coords = bary_coords  # (N2, 3)  or (B, N2, 3)
        self.faces1 = faces1  # (N1, 3)

        self.n2 = self.v2face_21.shape[-1]
        self.n1 = self.faces1.max() + 1

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

        Returns
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
            raise NotImplementedError("Batched version not implemented yet.")

        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = th.take_along_dim(
                self.faces1[self.v2face_21],
                self.bary_coords.argmax(1, keepdims=True),
                1,
            ).squeeze(-1)
            self._add_tensor_name(["_nn_map"])

        return self._nn_map

    @property
    def mT(self):
        target_faces = self.faces1[self.v2face_21]  # (n2, 3)

        In = th.tile(th.arange(self.n2, device=self.v2face_21.device), (3,))  # (3*n2)
        Jn = th.concatenate(
            [target_faces[:, 0], target_faces[:, 1], target_faces[:, 2]]
        )  # (3*n2)
        Sn = th.concatenate(
            [self.bary_coords[:, 0], self.bary_coords[:, 1], self.bary_coords[:, 2]]
        )  # (3*n2)

        precise_map = th.sparse_coo_tensor(
            th.stack([In, Jn]), Sn, (self.n2, self.n1)
        ).coalesce()
        return SparseMap(precise_map).mT

    def _to_np_sparse(self):
        return barycentric_to_precise(
            self.faces1.cpu().numpy(),
            self.v2face_21.cpu().numpy(),
            self.bary_coords.cpu().numpy(),
        )


class EmbP2PMap(P2PMap):
    """
    Point to point map, computed from embeddings.

    Simple wrapper around P2PMap

    Parameters
    -------------------
    emb1 : torch.Tensor
        (N1, p) or (B, N1, p)
    emb2 : torch.Tensor
        (N2, p) or (B, N2, p)
    n_jobs : int
        Number of jobs to use for the NN query
    """

    def __init__(self, emb1, emb2):
        self.emb1 = emb1.contiguous()  # (N1, K) or (B, N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K) or (B, N2, K)

        p2p_21 = nn_query(self.emb1, self.emb2)

        super().__init__(p2p_21, n1=self.emb1.shape[-2])
        self._add_tensor_name(["emb1", "emb2"])


class EmbPreciseMap(PreciseMap):
    """
    Point to barycentric map, computed from embeddings.

    Simple wrapper around PreciseMap

    .. note::
        Batched version is not supported yet.

    Parameters
    -------------------
    emb1 : torch.Tensor
        (N1, K)
    emb2 : torch.Tensor
        (N2, K)
    faces1 : torch.Tensor
        (N1, 3)
    clear_cache : bool
        The projection somehow leaves lot of cache on the GPU, which should be cleared manually (slower than the projection itself...)
    """

    def __init__(self, emb1, emb2, faces1, clear_cache=True):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        v2face_21, bary_coords = nn_query_precise_torch(
            self.emb1,
            faces1,
            self.emb2,
            return_dist=False,
            batch_size=min(2000, emb2.shape[0]),
            clear_cache=clear_cache,
        )

        # th.cuda.empty_cache()
        super().__init__(v2face_21, bary_coords, faces1)
        self._add_tensor_name(["emb1", "emb2"])


class KernelDenseDistMap(PointWiseMap):
    r"""Map represented by a row-normalized dense matrix obtained from an element-wise exponential.

    The matrix is of size `(N2, N1)`, and has values
    $P_{ij} = \frac{1}{\sum_j \exp(D_{ij})} \exp(D_{ij})$ where $D$ is some matrix

    Only D has to be provided

    Parameters
    -------------------
    log_matrix : torch.Tensor
        (N2, N1) or (B, N2, N1), the matrix D
    lse_row : torch.Tensor, optional
        (N2,) or (B, N2). The logsumexp on rows
    lse_col : torch.Tensor
        (N1,) or (B, N1). The logsumexp on columns
    """

    def __init__(self, log_matrix, lse_row=None, lse_col=None):
        super().__init__(tensor_names=["log_matrix"])
        self.log_matrix = log_matrix  # (..., N2, N1)
        self.lse_row = lse_row  # (..., N2)
        self.lse_col = lse_col  # (..., N1)

        self._nn_map = None
        self._inv_nn_map = None

        if lse_row is not None:
            self._add_tensor_name(["lse_row"])
        if lse_col is not None:
            self._add_tensor_name(["lse_col"])

    def _to_sparse(self):
        return self._to_dense()

    def _to_dense(self):
        if self.lse_row is None:
            self.lse_row = th.logsumexp(self.log_matrix, dim=-1)  # (..., N2)
            self._add_tensor_name(["lse_row"])

        return th.exp(self.log_matrix - self.lse_row.unsqueeze(-1))

    def pull_back(self, f):
        """Pull back a function $f$.
        Four possibilities:

        #. f is a function on S1, of shape (N1,), then the output is a function on S2, of shape (N2,)
        #. f represents multiple function on S1, of shape (N1, p), then the output is a function on S2, of shape (N2, p)
        #. f represents a batch multiple function on S1, of shape (B, N1, p), then the output is a function on S2, of shape (B, N2, p)

        Note tht the case where f is a batch of a single function (B, N1) is not supported, and one should then use `f[..., None]`

        Parameters
        -------------------
        f : torch.Tensor
            (N1,), (N1, p) or (B, N1, p)

        Returns
        -------------------
        f_pb : torch.Tensor
            (N2,), (N2, p) or (B, N2, p)
        """
        return self._to_dense() @ f

    def get_nn(self):
        """Outputs the nearest neighbor map.
        The nearest neighbor map is the map that associates to each point of S2 the index of the closest point in S1.
        Simple argmax along the last dimension.

        Returns
        -------------------
        nn_map : torch.Tensor
            (N2,) on the representation
        """
        if self._nn_map is None:
            self._nn_map = self.log_matrix.argmax(-1)
            self._add_tensor_name(["_nn_map"])
        return self._nn_map

    @property
    def mT(self):
        """
        Transposes the map.

        Returns another KernelDenseDistMap object with the transposed matrix.

        Returns
        -------------------
        map_t : KernelDenseDistMap
            Transpose map
        """
        obj = KernelDenseDistMap(
            self.log_matrix.transpose(-1, -2),
            lse_row=self.lse_col,
            lse_col=self.lse_row,
        )
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
    r"""Kernel Map, computed from embeddings.

    Simple wrapper around KernelDenseDistMap.

    Kernel has the form $\exp\big(-\frac{s(x,y)}{2\sigma^2}\big)$, where $s(x,y)$ is either:
    - the negative squared distance between the embeddings of x and y.
    - the (positive) inner product between the embeddings of x and y (potentially normalized).

    Parameters
    -------------------
    emb1 : torch.Tensor
        (N1, p) or (B, N1, p) embedding on first shape
    emb2 : torch.Tensor
        (N2, p) or (B, N2, p) embedding on second shape
    blur : float
        Standard deviation of the Gaussian kernel.
    normalize : bool
        Normalize the blur by the maximum distance between embedding points
    normalize_emb : bool
        Normalize the embeddings.
    dist_type : string
        {"sqdist", "inner"} Type of score to use.

    """

    def __init__(
        self,
        emb1,
        emb2,
        blur=None,
        normalize=False,
        normalize_emb=False,
        dist_type="sqdist",
    ):

        assert dist_type in ["sqdist", "inner"], "Invalid distance type."

        self.emb1 = emb1  # (N1, p) or (B, N1, p)
        self.emb2 = emb2  # (N2, p) or (B, N2, p)
        if normalize_emb:
            # self.emb1 = self.emb1 / th.linalg.norm(self.emb1, dim=-1, keepdim=True)  # (N1, p) or (B, N1, p)
            # self.emb2 = self.emb2 / th.linalg.norm(self.emb2, dim=-1, keepdim=True)  # (N2, p) or (B, N2, p)
            self.emb1 = nn.functional.normalize(
                self.emb1, p=2, dim=-1
            )  # (N1, p) or (B, N1, p)
            self.emb2 = nn.functional.normalize(
                self.emb2, p=2, dim=-1
            )  # (N2, p) or (B, N2, p)

        if dist_type == "sqdist":
            dist = compute_sqdistmat(
                self.emb2, self.emb1, normalized=normalize_emb
            )  # (N2, N1)  or (B, N2, N1)
        elif dist_type == "inner":
            dist = -self.emb2 @ self.emb1.transpose(-2, -1)  # (N2, N1)  or (B, N2, N1)

        self.dist_type = dist_type

        self.blur = th.ones(1, device=self.emb1.device)
        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            assert dist_type == "sqdist", "Normalization only supported for sqdist."
            with th.no_grad():
                self.blur = blur * th.sqrt(dist.max())

        log_matrix = -dist / (2 * th.square(self.blur))  # (N2, N1)  or (B, N2, N1)

        super().__init__(log_matrix)
        self._add_tensor_name(["emb1", "emb2", "blur"])


class KernelDistMap(PointWiseMap):
    r"""Memory-Scalable Version of EmbKernelDenseDistMap

    Row normalized version fo the kernel map of the form $\exp\big(-\frac{s(x,y)}{2\sigma^2}\big)$, where $s(x,y)$ is either:
    - the negative squared distance between the embeddings of x and y.
    - the (positive) inner product between the embeddings of x and y (potentially normalized).

    Parameters
    -------------------
    emb1 : torch.Tensor
        (N1, p) or (B, N1, p) embedding on first shape
    emb2 : torch.Tensor
        (N2, p) or (B, N2, p) embedding on second shape
    blur : float
        Standard deviation of the Gaussian kernel.
    normalize : bool
        Normalize the blur by the maximum distance between embedding points
    normalize_emb : bool
        Normalize the embeddings.
    dist_type : string
        {"sqdist", "inner"} Type of score to use.

    """

    def __init__(
        self,
        emb1,
        emb2,
        normalize=False,
        blur=None,
        normalize_emb=False,
        dist_type="sqdist",
    ):

        super().__init__(tensor_names=["emb1", "emb2", "blur"])
        assert dist_type in ["sqdist", "inner"], "Invalid distance type."
        self.dist_type = dist_type

        self.emb1 = emb1.contiguous()  # (N1, K)  or (B, N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)  or (B, N2, K)
        if normalize_emb:
            # self.emb1 = self.emb1 / th.linalg.norm(self.emb1, dim=-1, keepdim=True)  # (N1, p) or (B, N1, p)
            # self.emb2 = self.emb2 / th.linalg.norm(self.emb2, dim=-1, keepdim=True)  # (N2, p) or (B, N2, p)
            self.emb1 = nn.functional.normalize(
                self.emb1, p=2, dim=-1
            )  # (N1, p) or (B, N1, p)
            self.emb2 = nn.functional.normalize(
                self.emb2, p=2, dim=-1
            )  # (N2, p) or (B, N2, p)

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
        formula = pykeops.torch.Genred(
            "SqDist(X,Y)",
            [
                f"X = Vi({self.emb1.shape[-1]})",  # First arg  is a parameter,    of dim 1
                f"Y = Vj({self.emb2.shape[-1]})",  # Second arg is indexed by "i", of dim
            ],
            reduction_op="Max",
            axis=0,
        )

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
            dist = -emb2_i.sqdist(emb1_j) / sqblur
        elif self.dist_type == "inner":
            dist = (emb2_i | emb1_j) / sqblur

        return dist.sumsoftmaxweight(f, axis=1)  # (B, N2, p)

    def _to_sparse(self):
        return self._to_dense()

    def _to_dense(self):
        densemap = EmbKernelDenseDistMap(
            self.emb1,
            self.emb2,
            blur=self.blur,
            normalize=False,
            normalize_emb=False,
            dist_type=self.dist_type,
        )

        return densemap._to_dense()

    def pull_back(self, f):
        """Pull back a function $f$.
        Four possibilities:

        #. f is a function on S1, of shape (N1,), then the output is a function on S2, of shape (N2,)
        #. f represents multiple function on S1, of shape (N1, p), then the output is a function on S2, of shape (N2, p)
        #. f represents a batch multiple function on S1, of shape (B, N1, p), then the output is a function on S2, of shape (B, N2, p)

        Note tht the case where f is a batch of a single function (B, N1) is not supported, and one should then use `f[..., None]`

        Parameters
        -------------------
        f : np.ndarray
            (N1,), (N1, p) or (B, N1, p)

        Returns
        -------------------
        f_pb : np.ndarray
            (N2,), (N2, p) or (B, N2, p)
        """

        n_func = f.shape[-1] if f.ndim > 1 else 1
        pull_back_formula = self.get_pull_back_formula(n_func)

        sqblur = 2 * th.square(self.blur)

        if f.ndim == 1:
            f_in = f.unsqueeze(-1).contiguous()  # (N1, 1)
            if self.emb1.ndim == 3:
                f_in = f_in.unsqueeze(0)  # (1, N1, 1)
            f_pb = pull_back_formula(f_in, self.emb1, self.emb2, sqblur).squeeze(
                -1
            )  # (N2, )

        elif f.ndim == 2:
            f_input = f.contiguous()  # (N1, p)
            if self.emb1.ndim == 3:
                f_input = f_input.unsqueeze(0)  # (1, N1, p)
            f_pb = pull_back_formula(f_input, self.emb1, self.emb2, sqblur)  # (N2, p)

        elif f.ndim == 3:
            f_input = f.contiguous()
            if self.emb1.ndim == 2:
                f_pb = pull_back_formula(
                    f, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur
                )  # (B, N2, p)
            else:
                f_pb = pull_back_formula(f_input, self.emb1, self.emb2, sqblur)
        else:
            raise ValueError("Function is only dim 1, 2 or 3")

        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = nn_query(self.emb1, self.emb2)
            self._add_tensor_name(["_nn_map"])

        return self._nn_map

    @property
    def mT(self):
        invmap = KernelDistMap(
            self.emb2,
            self.emb1,
            blur=self.blur,
            normalize=False,
            dist_type=self.dist_type,
        )
        invmap.blur = self.blur
        return invmap
