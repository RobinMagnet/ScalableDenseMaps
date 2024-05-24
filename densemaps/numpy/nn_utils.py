from sklearn.neighbors import NearestNeighbors
import numpy as np

def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : np.ndarray
        (N1,p) or (B, N1, p) first collection
    Y : np.ndarray
        (N2,p) or (B, N2, p) second collection
    k : int
        number of neighbors to look for
    return_distance : bool
        hether to return the nearest neighbor distance
    n_jobs          : int
        number of parallel jobs. Set to -1 to use all processes

    Returns
    -------------------------------
    dists   : np.ndarray, optional
        (n2,k) or (n2,) if k=1 (with optional first batch dimension)- ONLY if return_distance is False. Nearest neighbor distance.
    matches : np.ndarray
        (n2,k) or (n2,) if k=1 (with optional first batch dimension)- nearest neighbor
    """
    if X.shape == 3:
        assert Y.shape == 3
        if X.shape[0] != 1 and Y.shape[0] != 1:
            all_res = [knn_query(X[i], Y[i], k=k, return_distance=return_distance, n_jobs=n_jobs) for i in range(X.shape[0])]
        elif X.shape[0] == 1 and Y.shape[0] != 1:
            all_res = [knn_query(X.squeeze(), Y[i], k=k, return_distance=return_distance, n_jobs=n_jobs) for i in range(Y.shape[0])]
        else:
            all_res = [knn_query(X[i], Y.squeeze(), k=k, return_distance=return_distance, n_jobs=n_jobs) for i in range(X.shape[0])]

        if return_distance:
            dists = np.stack([res[0] for res in all_res], axis=0)  # (B, n2, k)
            matches = np.stack([res[1] for res in all_res], axis=0) # (B, n2, k)
            return dists, matches
        else:
            matches = np.stack(all_res, axis=0)  # (B, n2, k)
            return matches



    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)

    if X.ndim == 2:
        tree.fit(X)
        dists, matches = tree.kneighbors(Y)
    elif X.ndim == 3:
        n_batches = X.shape[0]
        dists = []
        matches = []
        for i in range(n_batches):
            tree.fit(X[i])
            d, m = tree.kneighbors(Y[i])
            dists.append(d)
            matches.append(m)
        dists = np.stack(dists)  # (B, N, k)
        matches = np.stack(matches)  # (B, N, k)
    else:
        raise ValueError("X must have 2 or 3 dimensions")

    if k == 1:
        dists = dists.squeeze(-1)
        matches = matches.squeeze(-1)

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
    normalized : bool
        Whether the points are normalized to have unit norm.

    Returns
    -------
    distmat : torch.Tensor
        The pairwise squared Euclidean distance matrix between X and Y, of shape (N, M) or (B, N, M).
    """
    if not normalized:
        # (..., N, 1) + (...,1, M)
        return np.square(X).sum(-1).unsqueeze(-1) + np.square(Y).sum(-1).unsqueeze(-2) - 2 * (X @ Y.mT)
    else:
        return 2 - 2 * X @ Y.mT