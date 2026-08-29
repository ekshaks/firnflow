"""Shared helpers: reading Cohere parquet shards and talking to the API.

Every script in this directory imports from here so the server URL, the
authentication header and the parquet reading rules are defined once.

Environment:
  FIRNFLOW_URL            base URL of the running server. Default
                          http://127.0.0.1:3000.
  FIRNFLOW_API_KEY        bearer token for the data routes. Only needed
                          if the server was started with a key. Omitted
                          from requests when unset.
  FIRNFLOW_METRICS_TOKEN  bearer token for /metrics, which the server
                          gates separately from the data routes. The
                          sweep needs it to read the result-cache
                          counter that validates its latency figures.
"""

import math
import os

import numpy as np
import pyarrow.parquet as pq
import requests

#: Base URL of the firnflow server under test.
BASE_URL = os.environ.get("FIRNFLOW_URL", "http://127.0.0.1:3000").rstrip("/")

#: Content type the /import route expects for an Arrow IPC stream body.
ARROW_STREAM = "application/vnd.apache.arrow.stream"

#: Column in the Cohere parquet files that holds the embedding.
EMBEDDING_COLUMN = "emb"


def auth_headers(extra=None):
    """Build request headers, adding the bearer token when one is configured.

    Args:
        extra: optional dict of additional headers to merge in.

    Returns:
        A dict suitable for passing as ``headers=`` to ``requests``.
    """
    headers = dict(extra or {})
    key = os.environ.get("FIRNFLOW_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def read_embeddings(path, limit=None):
    """Read a Cohere parquet shard's embedding column as a float32 matrix.

    Args:
        path: path to one ``enNNNN.parquet`` shard.
        limit: stop after this many rows. ``None`` reads the whole shard.

    Returns:
        A ``(rows, dim)`` float32 numpy array in file order. Row i of the
        returned matrix is row i of the shard, which is what lets the
        caller assign ids by position.
    """
    reader = pq.ParquetFile(path)
    blocks = []
    rows = 0
    for batch in reader.iter_batches(batch_size=10_000, columns=[EMBEDDING_COLUMN]):
        flat = np.asarray(batch.column(0).flatten(), dtype=np.float32)
        blocks.append(flat.reshape(len(batch), -1))
        rows += len(batch)
        if limit is not None and rows >= limit:
            break
    matrix = np.vstack(blocks)
    return matrix[:limit] if limit is not None else matrix


def unit_normalise(matrix):
    """Scale every row to length 1 so a dot product equals cosine similarity.

    A single-vector namespace ranks by squared Euclidean distance. On
    vectors of equal length, Euclidean order and cosine order are the
    same, so normalising both sides makes the brute-force ranking here
    directly comparable to the server's ranking. Cohere's published
    embeddings already have length 1 to within about 5e-5; this removes
    the remainder.

    Args:
        matrix: a ``(rows, dim)`` float32 array.

    Returns:
        A new ``(rows, dim)`` float32 array with unit-length rows. Rows
        that are all zero are left unchanged rather than divided by zero.
    """
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return matrix / lengths


def percentile(sorted_values, fraction):
    """Nearest-rank percentile of an already-sorted list.

    Nearest rank means the smallest value at or below which the given
    fraction of the sample falls. For 200 samples, p95 is the 190th
    value, which is index 189.

    Args:
        sorted_values: list of numbers in ascending order.
        fraction: the percentile as a fraction, e.g. 0.95 for p95.

    Returns:
        The value at that rank.
    """
    rank = math.ceil(fraction * len(sorted_values))
    return sorted_values[min(max(rank, 1), len(sorted_values)) - 1]


def cache_hit_count(namespace):
    """Read a namespace's cumulative result-cache hit counter from /metrics.

    The counter is exposed as
    ``firnflow_cache_hits_total{namespace="..."} <value>``.

    The sweep compares this before and after each measured pass. A
    measured query answered from the result cache reports the latency of
    a cache lookup rather than of a search, so any increase during a
    measured pass invalidates that pass.

    Args:
        namespace: the namespace whose counter to read.

    A namespace that has never had a cache hit has no line of its own,
    which counts as 0 rather than as unknown.

    Args:
        namespace: the namespace whose counter to read.

    Returns:
        The counter as an int.

    Raises:
        SystemExit: if /metrics cannot be read. The caller cannot tell a
            clean run from a cache-served one without this counter, so
            failing loudly beats writing results with the check quietly
            missing. The server gates /metrics with
            FIRNFLOW_METRICS_TOKEN, separately from FIRNFLOW_API_KEY.
    """
    headers = auth_headers()
    metrics_token = os.environ.get("FIRNFLOW_METRICS_TOKEN")
    if metrics_token:
        headers["Authorization"] = f"Bearer {metrics_token}"
    try:
        response = requests.get(f"{BASE_URL}/metrics", headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(
            f"cannot read {BASE_URL}/metrics ({error}). The sweep uses the "
            f"result-cache counter there to prove its latency figures are "
            f"search timings and not cache lookups. If the server sets "
            f"FIRNFLOW_METRICS_TOKEN, export the same value here."
        ) from error
    marker = f'firnflow_cache_hits_total{{namespace="{namespace}"}}'
    for line in response.text.splitlines():
        if line.startswith(marker):
            return int(float(line.split()[-1]))
    return 0
