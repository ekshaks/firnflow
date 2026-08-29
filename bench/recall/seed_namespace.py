"""Load Cohere parquet shards into a namespace through POST /ns/{ns}/import.

The namespace does not need to exist first. The first import creates it
and fixes its dimension at 1024, the width of these embeddings.

Ids are assigned by position across the whole run: the first row of the
first shard gets id 0, and numbering continues across shard boundaries.
`ground_truth.py` numbers rows the same way when it scans the shards, so
an id returned by the server refers to the same row the brute-force scan
saw. Pass the shards in the same order to both scripts.

Rows are normalised to unit length before import. Cohere's embeddings
already have length 1 to within about 5e-5, so this changes almost
nothing, but it makes the server's Euclidean ranking and the ground
truth's cosine ranking exactly equivalent rather than nearly so.

Usage:
    python seed_namespace.py NAMESPACE SHARD [SHARD ...]

Example:
    python seed_namespace.py wiki1m ./data/en00{00,01,02,03,04}.parquet

One shard of 100,000 rows is about 400 MB as an Arrow stream, well under
the 8 GiB default of FIRNFLOW_IMPORT_MAX_BYTES. The server spools the
body to FIRNFLOW_IMPORT_TMP_DIR, so that directory needs the same free
space.
"""

import io
import sys
import time

import pyarrow as pa
import requests

from corpus import ARROW_STREAM, BASE_URL, auth_headers, read_embeddings, unit_normalise

#: Rows per Arrow record batch inside one import request.
BATCH_ROWS = 10_000


def arrow_stream(matrix, first_id):
    """Encode a matrix as an Arrow IPC stream of (id, vector) batches.

    Args:
        matrix: a ``(rows, dim)`` float32 array of unit-length vectors.
        first_id: id to give the first row. Later rows count up from it.

    Returns:
        The encoded stream as bytes.
    """
    dim = matrix.shape[1]
    schema = pa.schema(
        [
            pa.field("id", pa.uint64(), nullable=False),
            pa.field(
                "vector",
                pa.list_(pa.field("item", pa.float32(), nullable=True), dim),
                nullable=True,
            ),
        ]
    )
    buffer = io.BytesIO()
    writer = pa.ipc.new_stream(buffer, schema)
    for start in range(0, len(matrix), BATCH_ROWS):
        block = matrix[start : start + BATCH_ROWS]
        values = pa.array(block.reshape(-1))
        vectors = pa.FixedSizeListArray.from_arrays(values, dim)
        ids = pa.array(range(first_id + start, first_id + start + len(block)), type=pa.uint64())
        writer.write_batch(pa.RecordBatch.from_arrays([ids, vectors], schema=schema))
    writer.close()
    return buffer.getvalue()


def wait_for_operation(operation_id):
    """Poll GET /operations/{id} until the import stops running.

    Args:
        operation_id: the id returned by the import request.

    Raises:
        SystemExit: if the operation finishes in any state but succeeded.
    """
    while True:
        response = requests.get(
            f"{BASE_URL}/operations/{operation_id}", headers=auth_headers(), timeout=60
        )
        response.raise_for_status()
        status = response.json()
        if status["status"] != "running":
            break
        time.sleep(1.0)
    if status["status"] != "succeeded":
        raise SystemExit(f"import failed: {status}")


def import_shard(namespace, path, first_id):
    """Import one shard into `namespace`.

    Args:
        namespace: target namespace name.
        path: path to the parquet shard.
        first_id: id to give the shard's first row.

    Returns:
        A ``(rows, read_seconds, import_seconds)`` triple. Reading and
        encoding the parquet is timed separately from the request,
        because the first is local work and only the second says
        anything about the server.
    """
    started = time.time()
    matrix = unit_normalise(read_embeddings(path))
    body = arrow_stream(matrix, first_id)
    read_seconds = time.time() - started
    started = time.time()
    response = requests.post(
        f"{BASE_URL}/ns/{namespace}/import",
        data=body,
        headers=auth_headers({"Content-Type": ARROW_STREAM}),
        timeout=3600,
    )
    response.raise_for_status()
    wait_for_operation(response.json()["operation_id"])
    return len(matrix), read_seconds, time.time() - started


def main():
    """Import every shard named on the command line, in order."""
    namespace = sys.argv[1]
    shards = sys.argv[2:]
    if not shards:
        raise SystemExit(__doc__)
    total = 0
    for path in shards:
        rows, read_seconds, import_seconds = import_shard(namespace, path, total)
        total += rows
        print(
            f"{path}: {rows} rows, read {read_seconds:.1f}s, "
            f"import {import_seconds:.1f}s, {total} rows in namespace",
            flush=True,
        )


if __name__ == "__main__":
    main()
