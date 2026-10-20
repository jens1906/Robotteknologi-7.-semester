"""Distance between part-based embeddings (upstream compute_distance_matrix_using_bp_features)."""
import numpy as np


def part_distance(query, query_vis, gallery, gallery_vis, min_parts=2):
    """Distance of one query to many gallery entries, over the parts both can see.

    query: (P, D) L2-normalised embeddings, row 0 = whole person, rows 1.. = body parts
    (P = 1 for a model without parts, e.g. OSNet). query_vis: (P,) bool.
    gallery: (G, P, D). gallery_vis: (G, P) bool.
    Returns (G,) mean euclidean distance (0..2) over parts visible in both. NaN means unknown:
    fewer than min_parts body parts (row 0 does not count) are visible in both.
    """
    shared = query_vis[None] & gallery_vis                                    # (G, P)
    dist = np.linalg.norm(query[None] - gallery, axis=-1)                     # (G, P)
    mean = (dist * shared).sum(axis=1) / np.maximum(shared.sum(axis=1), 1)
    body = shared[:, 1:] if shared.shape[1] > 1 else shared
    mean[body.sum(axis=1) < min(min_parts, body.shape[1])] = np.nan
    return mean
