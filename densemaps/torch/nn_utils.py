import torch as th


try:
    import pykeops
    KEOPS_AVAILABLE = True
except ImportError:
    KEOPS_AVAILABLE = False

def compute_sqdistmat(X, Y, normalized=False):
    """
    Computes the pairwise squared Euclidean distance matrix between two sets of points X and Y.

    Parameters
    ----------
    X : torch.Tensor
        The first set of points, of shape (N, D) or (B, N, D).
    Y : torch.Tensor
        The second set of points, of shape (M, D) or (B, M, D).

    Returns
    -------
    torch.Tensor
        The pairwise squared Euclidean distance matrix between X and Y, of shape (N, M) or (B, N, M).
    """
    if not normalized:
        # (..., N, 1) + (...,1, M)
        return th.square(X).sum(-1).unsqueeze(-1) + th.square(Y).sum(-1).unsqueeze(-2) - 2 * (X @ Y.mT)
    else:
        return 2 - 2 * (X @ Y.transpose(-2, -1)) / X.shape[-1]

def nn_query(X, Y, use_keops=None):
    """
    Computes the nearest neighbor query between two sets of points X and Y.
    Use KeOps for efficient computation if available and more than 25k points

    Parameters
    ----------
    X : torch.Tensor
        The first set of points, of shape (N, D) or (B, N, D).
    Y : torch.Tensor
        The second set of points, of shape (M, D) or (B, M, D).
    use_keops : bool
        Whether to use KeOps for efficient computation. If None, use KeOps if available.

    Returns
    -------
    torch.Tensor
        The indices of the nearest neighbors of each point of Y in X, of shape (M,) or (B, M).
    """
    if use_keops is None:
        if X.ndim == 2:
            size = X.shape[0] * Y.shape[0]
        else:
            size = X.shape[-2] * Y.shape[-2] * X.shape[0]

        use_keops = KEOPS_AVAILABLE and size >= 25000

    if use_keops:
        return nn_query_keops(X, Y)
    else:
        return nn_query_dense(X, Y)


def nn_query_dense(X, Y):
    """
    Computes the nearest neighbor query between two sets of points X and Y.
    Use dense computation with PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        The first set of points, of shape (N, D) or (B, N, D).
    Y : torch.Tensor
        The second set of points, of shape (M, D) or (B, M, D).

    Returns
    -------
    torch.Tensor
        The indices of the nearest neighbors of each point of Y in X, of shape (M,) or (B, M).
    """

    distmat = compute_sqdistmat(X, Y)  # (B, N, M)

    return distmat.argmin(-2)
    

def nn_query_keops(X, Y):
    """
    Computes the nearest neighbor query between two sets of points X and Y.
    Use KeOps for efficient computation.

    Parameters
    ----------
    X : torch.Tensor
        The first set of points, of shape (N, D) or (B, N, D).
    Y : torch.Tensor
        The second set of points, of shape (M, D) or (B, M, D).

    Returns
    -------
    torch.Tensor
        The indices of the nearest neighbors of each point of Y in X, of shape (M,) or (B, M).
    """
    formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({X.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({Y.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='ArgMin',
                    axis=0)
    
    return formula(X, Y).squeeze(-1)