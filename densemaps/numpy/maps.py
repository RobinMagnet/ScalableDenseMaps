import numpy as np
import scipy.sparse as sparse
from scipy.special import logsumexp

# from .point_to_triangle import nn_query_precise_torch

from .nn_utils import knn_query, compute_sqdistmat
from .point_to_triangle import nn_query_precise_np, barycentric_to_precise


class PointWiseMap:
    r"""
    Root class representing a pointwise map. Not supposed to be instanciated in itself.

    A pointwise, denoted $P : S_2 \to S_1$, is a map that associates to each point x in $S_2$ a point $P(x)$ in $S_1$.
    It can usually be represented as a $n_2 \times n_1$ matrix, where $n_2$ is the number of points in $S_2$ and $n_1$ the number of points in $S_1$.

    Given a pointwise map $P$, the pullback of a function $f : S_1 \to R$ is a function $f_{pb} : S_2 \to R$ defined by $f_{pb}(x) = f(P(x))$.
    In practice it can easily be computed by matrix multiplication: $f_{pb} = P f$.

    In practice, we usually don't need to use the exact values inside $P$, btu rather only care about multiplying with some functions,
    extracting maximal values per-row or per-column, or summing on rows or columns.


    Attributes
    -------------------
    array_names : list of str
        Names of the arrays stored in the object. Used to transfer to torch and to GPU.

    """

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
        """Pull back a function $f$.
        Three possibilities:

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
        raise NotImplementedError

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

    def __matmul__(self, other):
        if issubclass(type(other), PointWiseMap):
            return SparseMap(self._to_sparse() @ other._to_sparse())

        return self.pull_back(other)

    def get_nn(self):
        """Ouptputs the nearest neighbor map.
        The nearest neighbor map is the map that associates to each point of S2 the index of the closest point in S1.

        Returns
        -------------------
        nn_map : np.ndarray
            (N2,) or (B, N2) depending on the representation
        """
        raise NotImplementedError

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
        """Returns sparse (or dense) matrix representation of the map.

        Returns
        -------------------
        map_sparse : scipy.sparse.csr_matrix or np.ndarray
            Sparse matrix representation of the map
        """
        raise NotImplementedError


class SparseMap(PointWiseMap):
    """Map represented by a sparse matrix.

    The sparse matrix is of size `(N2, N1)`.

    Parameters
    -------------------
    map : scipy.sparse.csr_matrix
        (N2, N1) or (N2, N1)
    """

    def __init__(self, map):
        super().__init__(array_names=["map"])

        self.map = map  # (N2, N1)

    @property
    def shape(self):
        """
        Shape of the map.

        Returns
        -------------------
        shape : tuple
            (N2, N1)
        """
        return self.map.shape

    @property
    def mT(self):
        """Returns the transpose of the map.

        Returns the transpose of the matrix representation of the map.

        Returns
        -------------------
        map_t : PointWiseMap
            Transpose map
        """
        return SparseMap(self.map.T)

    def __matmul__(self, other):
        if issubclass(type(other), PointWiseMap):
            return SparseMap(self.map @ other._to_sparse())
        return self.pull_back(other)

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
        return self.map @ f

    def get_nn(self):
        """Outputs the nearest neighbor map.
        The nearest neighbor map is the map that associates to each point of S2 the index of the closest point in S1.
        Simple argmax along the last dimension.

        Returns
        -------------------
        nn_map : np.ndarray
            (N2,) on the representation
        """
        return self.map.argmax(-1)  # (N2, )

    def _to_sparse(self):
        return self.map


class P2PMap(PointWiseMap):
    r"""
    Point to point map from a set S2 to a set S1.
    Defined by a map $P_{21} : S2 \to S1$ or an array `p2p_21` of size `(n_2)`, where `p2p_21[i]` is the index of the point in S1 closest to the point i in S2.
    Batched versions are accepted

    Parameters
    -------------------
    p2p_21 : (n2,) or (B, n2)
    n1 : int or None
        Number of points in S1. If None, n1 = p2p.max()+1
    """

    def __init__(self, p2p_21, n1=None):

        super().__init__(array_names=["p2p_21"])

        self.p2p_21 = p2p_21  # (n2, ) or (B, n2)

        self.n2 = self.p2p_21.shape[-1]
        self._n1 = n1

        self.max_ind = self.p2p_21.max().item() if n1 is None else n1 - 1

    @property
    def shape(self):
        """
        Shape of the map.

        Returns
        -------------------
        shape : tuple
            returns (N2,) or (B, N2) depending on the representation
        """
        return self.p2p_21.shape

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

    def pull_back(self, f):
        """Pull back a function $f$.
        Four possibilities:

        #. f is a function on S1, of shape (N1,), then the output is a function on S2, of shape (N2,) (or (B, N2,) if the map is batched)
        #. f represents multiple function on S1, of shape (N1, p), then the output is a function on S2, of shape (N2, p) (or (B, N2, p) if the map is batched)
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
        # Ensure f doesn't have too few entries
        if f.ndim == 1 and f.shape[-1] <= self.max_ind:
            raise ValueError(
                f"Function f doesn't have enough entries, need at least {1+self.max_ind} but only has {f.shape[-1]}"
            )
        elif f.ndim > 1 and f.shape[-2] <= self.max_ind:
            raise ValueError(
                f"Function f doesn't have enough entries, need at least {1+self.max_ind} but only has {f.shape[-2]}"
            )

        # Ensure potential batch dimensions match
        if f.ndim == 3 and self.p2p_21.ndim == 2:
            assert (
                f.shape[0] == self.p2p_21.shape[0]
            ), "Batch size of f and p2p_21 should match"

        if f.ndim == 1 or f.ndim == 2:  # (N1,) or (N1, p)
            f_pb = f[self.p2p_21]  # (n2, p) or (B, n2, p)
        elif f.ndim == 3:
            if self.p2p_21.ndim == 1:
                f_pb = f[:, self.p2p_21]  # (B, n2, k)
            else:
                f_pb = f[np.arange(f.shape[0])[:, None], self.p2p_21]
        else:
            raise ValueError("Function is only dim 1, 2 or 3")

        return f_pb

    def get_nn(self):
        """Ouptputs the nearest neighbor map.
        The nearest neighbor map is the same as the input.

        Returns
        -------------------
        nn_map : np.ndarray
            (N2,) or (B, N2) depending on the representation
        """
        return self.p2p_21

    @property
    def mT(self):
        assert self.p2p_21.ndim == 1, "Batched version not implemented yet."
        P21 = self._to_sparse()
        return SparseMap(P21.T)

    def _to_sparse(self):
        assert self.p2p_21.ndim == 1, "Batched version not implemented yet."
        return sparse.csc_matrix(
            (np.ones_like(self.p2p_21), (np.arange(self.n2), self.p2p_21)),
            shape=(self.n2, self.n1),
        )


class PreciseMap(SparseMap):
    """
    Point to barycentric map from a set S2 to a surface S1.
    Batched Version is not supported yet.

    Is represented as a sparse matrix of size `(N2, N1)`, where there are at most 3 non-zero entries per row, which sum to 1.

    Parameters
    -------------------
    v2face_21 : np.ndarray
        (n2,) Indices of the faces of S1 closest to each point of S2.
    bary_coords : np.ndarray
        (n2, 3) Barycentric coordinates of the points of S2 in the faces of S1.
    faces1 :  np.ndarray
        (N1, 3) All the Faces of S1.
    """

    def __init__(self, v2face_21, bary_coords, faces1):
        if v2face_21.ndim == 2:
            raise ValueError("Batched version not implemented yet.")

        sparse_map = barycentric_to_precise(
            faces1, v2face_21, bary_coords, n_vertices=None
        )  # (n2, n1)
        super().__init__(map=sparse_map)
        self._nn_map = None


class EmbP2PMap(P2PMap):
    """
    Point to point map, computed from embeddings.

    Simple wrapper around P2PMap

    Parameters
    -------------------
    emb1 : np.ndarray
        (N1, p) or (B, N1, p)
    emb2 : np.ndarray
        (N2, p) or (B, N2, p)
    n_jobs : int
        Number of jobs to use for the NN query
    """

    def __init__(self, emb1, emb2, n_jobs=1):
        assert (
            emb1.shape[-1] == emb2.shape[-1]
        ), "Embeddings should have the same dimension."
        self.emb1 = emb1  # (N1, p) or (B, N1, p)
        self.emb2 = emb2  # (N2, p) or (B, N2, p)
        self.n_jobs = n_jobs

        p2p_21 = knn_query(self.emb1, self.emb2, n_jobs=n_jobs)  # (N2,) or (B, N2)

        super().__init__(p2p_21=p2p_21, n1=self.emb1.shape[-2])
        self._add_array_name(["emb1", "emb2"])


class EmbPreciseMap(PreciseMap):
    """
    Point to barycentric map, computed from embeddings.

    Simple wrapper around PreciseMap

    Parameters
    -------------------
    emb1 : np.ndarray
        (N1, K)
    emb2 : np.ndarray
        (N2, K)
    faces1 : np.ndarray
        (N1, 3)
    n_jobs : int
        Number of jobs to use for the NN query in the point_to_precise computation
    """

    def __init__(self, emb1, emb2, faces1, n_jobs=1):
        assert (
            emb1.shape[-1] == emb2.shape[-1]
        ), "Embeddings should have the same dimension."
        self.emb1 = emb1  # (N1, p)
        self.emb2 = emb2  # (N2, p)

        v2face_21, bary_coords = nn_query_precise_np(
            self.emb1,
            faces1,
            self.emb2,
            return_dist=False,
            batch_size=min(2000, emb2.shape[0]),
            n_jobs=n_jobs,
        )

        # th.cuda.empty_cache()
        super().__init__(v2face_21=v2face_21, bary_coords=bary_coords, faces1=faces1)
        self._add_array_name(["emb1", "emb2"])


class KernelDenseDistMap(PointWiseMap):
    r"""Map represented by a row-normalized dense matrix obtained from an element-wise exponential.

    The matrix is of size `(N2, N1)`, and has values $P_{ij} = \frac{1}{\sum_j \exp(D_{ij})} \exp(D_{ij})$ where $D$ is some matrix

    Only D has to be provided

    Parameters
    -------------------
    log_matrix : np.ndarray
        (N2, N1) or (B, N2, N1), the matrix D
    lse_row : np.ndarray, optional
        (N2,) or (B, N2). The logsumexp on rows
    lse_col : np.ndarray
        (N1,) or (B, N1). The logsumexp on columns
    """

    def __init__(self, log_matrix, lse_row=None, lse_col=None):
        super().__init__(array_names=["log_matrix"])
        self.log_matrix = log_matrix  # (..., N2, N1)
        self.lse_row = lse_row  # (..., N2)
        self.lse_col = lse_col  # (..., N1)

        self._nn_map = None
        self._inv_nn_map = None

        if lse_row is not None:
            self._add_array_name(["lse_row"])
        if lse_col is not None:
            self._add_array_name(["lse_col"])

    def _to_sparse(self):
        return self._to_dense()

    def _to_dense(self):
        if self.lse_row is None:
            self.lse_row = logsumexp(self.log_matrix, axis=-1)  # (..., N2)
            self._add_array_name(["lse_row"])

        return np.exp(self.log_matrix - self.lse_row[..., None])

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
        return self._to_dense() @ f

    def get_nn(self):
        """Outputs the nearest neighbor map.
        The nearest neighbor map is the map that associates to each point of S2 the index of the closest point in S1.
        Simple argmax along the last dimension.

        Returns
        -------------------
        nn_map : np.ndarray
            (N2,) on the representation
        """
        if self._nn_map is None:
            self._nn_map = self.log_matrix.argmax(-1)
            self._add_array_name(["_nn_map"])
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
            self.log_matrix.T, lse_row=self.lse_col, lse_col=self.lse_row
        )
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
    r"""Kernel Map, computed from embeddings.

    Simple wrapper around KernelDenseDistMap.

    Kernel has the form $\exp\big(-\frac{s(x,y)}{2\sigma^2}\big)$, where $s(x,y)$ is either:
    - the negative squared distance between the embeddings of x and y.
    - the (positive) inner product between the embeddings of x and y (potentially normalized).

    Parameters
    -------------------
    emb1 : np.ndarray
        (N1, p) or (B, N1, p) embedding on first shape
    emb2 : np.ndarray
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

        # Normalize embeddings
        if normalize_emb:
            norm1 = np.linalg.norm(
                self.emb1, axis=-1, keepdims=True
            )  # (N1, 1) or (B, N1, 1)
            norm2 = np.linalg.norm(self.emb2, axis=-1, keepdims=True)
            self.emb1 = self.emb1 / np.clip(norm1, 1e-6, None)  # (N1, p) or (B, N1, p)
            self.emb2 = self.emb2 / np.clip(norm2, 1e-6, None)  # (N2, p) or (B, N2, p)

        if dist_type == "sqdist":
            dist = compute_sqdistmat(
                self.emb2, self.emb1, normalized=normalize_emb
            )  # (N2, N1)  or (B, N2, N1)
        elif dist_type == "inner":
            if self.emb1.ndim == 2:
                dist = -self.emb2 @ self.emb1.T
            else:
                dist = -self.emb2 @ self.emb1.transpose(
                    0, 2, 1
                )  # (N2, N1)  or (B, N2, N1)

        self.dist_type = dist_type

        self.blur = 1 if blur is None else blur

        if normalize:
            assert dist_type == "sqdist", "Normalization only supported for sqdist."
            self.blur = blur * np.sqrt(dist.max())

        log_matrix = -dist / (2 * self.blur**2)  # (N2, N1)  or (B, N2, N1)

        super().__init__(log_matrix=log_matrix)
        self._add_array_name(["emb1", "emb2", "blur"])
