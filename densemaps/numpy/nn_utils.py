from sklearn.neighbors import NearestNeighbors
import numpy as np

def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : (n1,p) first collection
    Y : (n2,p) second collection
    k : int - number of neighbors to look for
    return_distance : whether to return the nearest neighbor distance
    n_jobs          : number of parallel jobs. Set to -1 to use all processes

    Output
    -------------------------------
    dists   : (n2,k) or (n2,) if k=1 - ONLY if return_distance is False. Nearest neighbor distance.
    matches : (n2,k) or (n2,) if k=1 - nearest neighbor
    """
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches

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
        return np.square(X).sum(-1).unsqueeze(-1) + np.square(Y).sum(-1).unsqueeze(-2) - 2 * (X @ Y.mT)
    else:
        return 2 - 2 * X @ Y.mT