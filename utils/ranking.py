"""Global retrieval, SuperGlobal-style reranking, and bidirectional MaxSim."""
import numpy as np

from .encoder import normalize


def superglobal_scores(query, candidates, neighbors=5, beta=1.0):
    """Query-aware database aggregation plus max-pooled query expansion.

    Candidate neighborhoods are restricted to the coarse shortlist; the query is
    included in each database candidate's refinement neighborhood.
    """
    n = len(candidates)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if neighbors < 1 or beta < 0:
        raise ValueError("superglobal_neighbors must be positive and beta nonnegative")
    all_vectors = np.vstack([query[None], candidates]).astype(np.float32)
    similarities = all_vectors @ all_vectors.T
    refined = np.empty_like(candidates)
    k = min(neighbors, n)
    for i in range(n):
        row = i + 1
        neighbors_idx = np.argsort(-similarities[row], kind="stable")
        neighbors_idx = neighbors_idx[neighbors_idx != row][:k]
        weights = np.maximum(similarities[row, neighbors_idx], 0) * beta
        refined[i] = normalize((candidates[i] + (weights[:, None] * all_vectors[neighbors_idx]).sum(axis=0))
                                / (1 + weights.sum()))
    top = np.argsort(-(candidates @ query), kind="stable")[:k]
    expanded = normalize(np.max(refined[top], axis=0))
    return 0.5 * (refined @ query + candidates @ expanded)


def bidirectional_maxsim(query_patches, gallery_patches):
    matrix = query_patches @ gallery_patches.T
    return float((matrix.max(axis=1).mean() + matrix.max(axis=0).mean()) * 0.5)
