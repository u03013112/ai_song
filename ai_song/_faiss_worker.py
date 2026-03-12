"""Subprocess worker for faiss index search.

faiss-cpu SEGFAULTS when called in the same process as PyTorch MPS on Apple
Silicon (faiss SWIG bindings conflict with Metal driver memory layout).
Running faiss in a separate subprocess avoids this completely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def search_index(
    index_path: str,
    query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load faiss index, search, and return results with reconstructed vectors.

    Args:
        index_path: Path to the .index file.
        query: Query vectors, shape (n, d), dtype float32.
        k: Number of nearest neighbors.

    Returns:
        Tuple of (scores, indices, big_npy) where big_npy is all index vectors.
    """
    import faiss

    index = faiss.read_index(index_path)
    big_npy = index.reconstruct_n(0, index.ntotal)
    score, ix = index.search(query, k=k)
    return score, ix, big_npy


if __name__ == "__main__":
    import pickle

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "rb") as f:
        args = pickle.load(f)

    result = search_index(**args)

    with open(output_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
