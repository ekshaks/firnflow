"""Compute the exact top-k neighbours of held-out query vectors.

This is the answer the index is scored against. For each query vector it
compares against every row of the loaded corpus and keeps the k closest.
No index, no approximation, no shortcuts. On a 1,000,000-row corpus of
1024-dimension vectors this takes a few minutes on a laptop.

The query vectors come from a shard that is NOT loaded into the
namespace. A query vector that is also a row in the table would find
itself at distance zero. That inflates the score and tells you nothing
about the index.

Shards are scanned one at a time and the running top-k is merged after
each, so peak memory stays at roughly one shard.

Pass the same shards in the same order that seed_namespace.py was given.
Ids here are assigned by position across the run, matching the ids the
server holds.

Usage:
    python ground_truth.py QUERY_SHARD NUM_QUERIES K OUTPUT.npz CORPUS_SHARD [CORPUS_SHARD ...]

Example:
    python ground_truth.py ./data/en0010.parquet 200 10 truth-1m.npz ./data/en000*.parquet

Output is a .npz holding three arrays:
    queries  (num_queries, dim) float32, the unit-length query vectors
    ids      (num_queries, k)   int64,   the true nearest row ids, closest first
    scores   (num_queries, k)   float32, their cosine similarities
"""

import sys

import numpy as np

from corpus import read_embeddings, unit_normalise


def merge_top_k(best_ids, best_scores, new_ids, new_scores, k):
    """Merge a new block of candidates into the running top-k.

    Args:
        best_ids: ``(queries, kept)`` int64 ids kept so far.
        best_scores: ``(queries, kept)`` float32 similarities kept so far.
        new_ids: ``(queries, take)`` int64 candidate ids.
        new_scores: ``(queries, take)`` float32 candidate similarities.
        k: how many to keep.

    Returns:
        A ``(ids, scores)`` pair, each ``(queries, min(k, total))``,
        sorted by descending similarity.
    """
    ids = np.hstack([best_ids, new_ids])
    scores = np.hstack([best_scores, new_scores])
    order = np.argsort(-scores, axis=1)[:, :k]
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(scores, order, axis=1)


def scan_shard(queries, path, first_id, k):
    """Exact top-k of every query against one shard.

    Args:
        queries: ``(num_queries, dim)`` float32 unit-length query matrix.
        path: parquet shard to scan.
        first_id: id of the shard's first row.
        k: how many neighbours to keep from this shard.

    Returns:
        A ``(ids, scores, rows_scanned)`` triple.
    """
    corpus = unit_normalise(read_embeddings(path))
    similarities = queries @ corpus.T
    take = min(k, similarities.shape[1])
    # argpartition is O(rows) per query and enough here: the exact order
    # of the top `take` is fixed by the sort inside merge_top_k.
    candidates = np.argpartition(-similarities, take - 1, axis=1)[:, :take]
    scores = np.take_along_axis(similarities, candidates, axis=1)
    return candidates.astype(np.int64) + first_id, scores, len(corpus)


def main():
    """Scan every corpus shard and write the merged top-k to disk."""
    query_shard, num_queries, k, output = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    corpus_shards = sys.argv[5:]
    if not corpus_shards:
        raise SystemExit(__doc__)

    queries = unit_normalise(read_embeddings(query_shard, num_queries))
    best_ids = np.zeros((len(queries), 0), dtype=np.int64)
    best_scores = np.zeros((len(queries), 0), dtype=np.float32)
    first_id = 0
    for path in corpus_shards:
        ids, scores, rows = scan_shard(queries, path, first_id, k)
        best_ids, best_scores = merge_top_k(best_ids, best_scores, ids, scores, k)
        first_id += rows
        print(f"scanned {path}, {first_id} rows total", flush=True)

    np.savez(output, queries=queries, ids=best_ids, scores=best_scores)
    print(f"wrote {output}: {len(queries)} queries x top-{k} over {first_id} rows")


if __name__ == "__main__":
    main()
