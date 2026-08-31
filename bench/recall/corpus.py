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

import hashlib
import math
import os
import re
import time

import numpy as np
import pyarrow.parquet as pq
import requests

#: Base URL of the firnflow server under test.
BASE_URL = os.environ.get("FIRNFLOW_URL", "http://127.0.0.1:3000").rstrip("/")

#: Content type the /import route expects for an Arrow IPC stream body.
ARROW_STREAM = "application/vnd.apache.arrow.stream"

#: Column in the Cohere parquet files that holds the embedding.
EMBEDDING_COLUMN = "emb"

#: Hugging Face dataset the corpus comes from. The repository was
#: renamed from ``Cohere/wikipedia-2023-11-embed-multilingual-v3`` to
#: ``CohereLabs/...``. The old name still redirects. The new one is
#: canonical and is what the API reports.
DATASET_REPO = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"

#: Commit the harness downloads from. ``main`` is a branch pointer and
#: moves: this dataset was last modified on 2026-03-25, after the run
#: recorded under bench/results/. Pinning the commit means a run a year
#: from now reads the same bytes as the run in the report.
DATASET_REVISION = "ade45fb52bd549f5e8c065636fe4160a43c2af36"

#: sha256 digest and size in bytes of each ``en/NNNN.parquet`` at
#: DATASET_REVISION, keyed by shard number. Regenerate with:
#:
#:   curl -s -X POST \
#:     "https://huggingface.co/api/datasets/$REPO/paths-info/$REVISION" \
#:     -H 'Content-Type: application/json' \
#:     -d '{"paths":["en/0000.parquet"]}'
#:
#: and read ``lfs.oid`` and ``size`` from each entry. Hugging Face
#: stores these files in Git LFS, and an LFS object id is the sha256 of
#: the file's contents, so no download is needed to fill this in.
SHARD_MANIFEST = {
    0: ("2c6abfffa7dd336113251b3e6f3fe4ee16688ead7c67b99593cbadc5589e28b3", 216612385),
    1: ("1ce373d5730494429a64a0f788c62f71226ded2bc670aaf9399204e91e544c3b", 216746705),
    2: ("57ea548a95cc455997ee70ecdac98e19641e3c1db51cd45a4d52cc9017be87b8", 216645218),
    3: ("fe5a8e01a3dba351c000fa019340ee8162880c0927dda2fab59ce802d830ea48", 216311581),
    4: ("5240dfc163f58b540fa0647f064a2cdda02d9c981e9268ef4a3319fd9b042e73", 214758303),
    5: ("17f2357bbb91c068e5eaf15ade457ff6603c2cdbd757d21270912b48ad4d3062", 216790181),
    6: ("d2a27a94ecd315f0f82058979993bfabfd180ac0ffaaf6237b521df6fe621598", 217259568),
    7: ("4e0a32d44461c775fd42b01915a4ab8ad57c437854575f84e57cbccd796b9175", 217022068),
    8: ("0dafef2112c4b7c7427634dc44823c4afdccd7ca54544279f3d1d6cc0d002020", 217443476),
    9: ("1d02c056f05c658bc074ef30db1b9da22b16e47620a92cab90867b5a98fd3699", 216040478),
    10: ("e46f909cfc63ac7f7e62b6394f8026fd450ee5aff01a2fdbb9f8e187dbb8e39a", 216623894),
}


def shard_index(path):
    """Recover the shard number from a local shard's file name.

    ``download_shards.py`` writes shard 7 as ``en0007.parquet``. Every
    later script needs the number back, because the checksum manifest is
    keyed by it.

    Args:
        path: path to a downloaded shard.

    Returns:
        The shard number as an int.

    Raises:
        SystemExit: if the file name is not ``enNNNN.parquet``. Renaming
            a shard breaks the link between the file and the checksum
            that describes it, and silently skipping the check would
            defeat the point of having one.
    """
    name = os.path.basename(path)
    match = re.fullmatch(r"en(\d{4})\.parquet", name)
    if not match:
        raise SystemExit(
            f"{path}: expected a file named enNNNN.parquet as written by "
            f"download_shards.py. The shard number in the name selects the "
            f"checksum this file is checked against, so a renamed shard "
            f"cannot be verified."
        )
    return int(match.group(1))


def file_digest(path):
    """sha256 of a file's contents, read in 8 MiB blocks.

    Args:
        path: path to any file.

    Returns:
        The digest as a lowercase hex string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_shard(path, index=None):
    """Check a downloaded shard against the pinned manifest.

    The size is compared first because it is free and rules out a
    truncated download without reading 216 MB.

    Args:
        path: path to a downloaded shard.
        index: shard number. Recovered from the file name when omitted.

    Raises:
        SystemExit: if the shard is not in the manifest, or if its size
            or digest differs. A mismatch means the file is not the one
            the report measured, so every number produced from it would
            be attributed to the wrong data. Aborting is the only safe
            answer.
    """
    if index is None:
        index = shard_index(path)
    if index not in SHARD_MANIFEST:
        raise SystemExit(
            f"shard {index} has no entry in SHARD_MANIFEST in corpus.py. Add "
            f"its sha256 and size at DATASET_REVISION before using it, or the "
            f"run cannot say which bytes it measured."
        )
    expected_digest, expected_size = SHARD_MANIFEST[index]
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise SystemExit(
            f"{path}: {actual_size} bytes, expected {expected_size} at "
            f"revision {DATASET_REVISION}. Delete the file and download it "
            f"again."
        )
    actual_digest = file_digest(path)
    if actual_digest != expected_digest:
        raise SystemExit(
            f"{path}: sha256 {actual_digest}, expected {expected_digest} at "
            f"revision {DATASET_REVISION}. The file is the right length and "
            f"the wrong contents, so this is not a partial download. Delete "
            f"it and download it again."
        )


def wait_for_operation(operation_id, poll_seconds=1.0):
    """Poll GET /operations/{id} until the operation stops running.

    Import, index creation, warmup and compaction all return an
    operation id and finish in the background.

    Args:
        operation_id: the id the route returned.
        poll_seconds: delay between polls.

    Returns:
        The final operation record as a dict.

    Raises:
        SystemExit: if the operation finishes in any state but
            succeeded.
    """
    while True:
        response = requests.get(
            f"{BASE_URL}/operations/{operation_id}", headers=auth_headers(), timeout=60
        )
        response.raise_for_status()
        status = response.json()
        if status["status"] != "running":
            break
        time.sleep(poll_seconds)
    if status["status"] != "succeeded":
        raise SystemExit(f"operation {operation_id} {status['status']}: {status}")
    return status


def namespace_row_count(namespace):
    """Live row count of a namespace, or None if it does not exist.

    ``GET /ns/{namespace}`` reports ``row_count`` as Lance's live row
    count. The seeding script uses it to refuse to import into a
    namespace that already holds rows.

    Args:
        namespace: the namespace to ask about.

    Returns:
        The row count as an int, or None when the server answers 404.

    Raises:
        SystemExit: on any other transport or HTTP error, because the
            caller is about to decide whether it is safe to write.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/ns/{namespace}", headers=auth_headers(), timeout=60
        )
    except requests.RequestException as error:
        raise SystemExit(f"cannot reach {BASE_URL}/ns/{namespace} ({error})") from error
    if response.status_code == 404:
        return None
    if not response.ok:
        raise SystemExit(
            f"GET /ns/{namespace} returned {response.status_code}: {response.text[:200]}"
        )
    return int(response.json()["row_count"])


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
