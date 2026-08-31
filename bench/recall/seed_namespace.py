"""Load Cohere parquet shards into a namespace through POST /ns/{ns}/import.

The namespace does not need to exist first. The first import creates it
and fixes its dimension at 1024, the width of these embeddings.

Ids are assigned by position across the whole run: the first row of the
first shard gets id 0, and numbering continues across shard boundaries.
`ground_truth.py` numbers rows the same way when it scans the shards, so
an id returned by the server refers to the same row the brute-force scan
saw. Pass the shards in the same order to both scripts.

The script refuses to run if the namespace already holds rows. Import
appends, so a second run would put two rows under every id and quietly
invalidate every recall figure measured afterwards. Delete the namespace
or pass a new name. Each shard is also checked against the sha256 pinned
in corpus.py before anything is sent, and the row count the server
reports afterwards is compared against the number of rows sent.

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

from corpus import (
    ARROW_STREAM,
    BASE_URL,
    auth_headers,
    namespace_row_count,
    read_embeddings,
    unit_normalise,
    verify_shard,
    wait_for_operation,
)

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


def refuse_if_not_empty(namespace):
    """Stop before writing if the namespace already holds rows.

    ``/import`` appends. It does not replace and it does not deduplicate,
    unlike ``/upsert``, which merges on `id`. Ids here are assigned by
    position starting at 0, so running this script twice against one
    namespace leaves two rows under every id. Every recall figure
    measured afterwards is then wrong, and nothing in the output says so.

    Args:
        namespace: the namespace about to be imported into.

    Raises:
        SystemExit: if the namespace exists and is not empty.
    """
    existing = namespace_row_count(namespace)
    if existing:
        raise SystemExit(
            f"namespace {namespace} already holds {existing} rows. Import "
            f"appends rather than replaces, and ids here are assigned by "
            f"position from 0, so importing again would put two rows under "
            f"every id and silently corrupt every recall number measured "
            f"afterwards. Either delete it first:\n"
            f"  curl -X DELETE {BASE_URL}/ns/{namespace}\n"
            f"or pass a namespace name that is not in use."
        )


def main():
    """Import every shard named on the command line, in order.

    Raises:
        SystemExit: if no shards are given, if the namespace is not
            empty, if a shard fails its checksum, or if the row count the
            server reports afterwards is not the number of rows sent.
    """
    namespace = sys.argv[1]
    shards = sys.argv[2:]
    if not shards:
        raise SystemExit(__doc__)
    refuse_if_not_empty(namespace)
    for path in shards:
        verify_shard(path)
    total = 0
    for path in shards:
        rows, read_seconds, import_seconds = import_shard(namespace, path, total)
        total += rows
        print(
            f"{path}: {rows} rows, read {read_seconds:.1f}s, "
            f"import {import_seconds:.1f}s, {total} rows in namespace",
            flush=True,
        )
    landed = namespace_row_count(namespace)
    if landed != total:
        raise SystemExit(
            f"sent {total} rows, but {namespace} reports {landed}. Ids are "
            f"assigned by position, so a count that does not match means the "
            f"ids the server holds are not the ids ground_truth.py will "
            f"compute. Delete the namespace and start again."
        )
    print(f"{namespace}: {landed} rows, count confirmed by GET /ns/{namespace}")


if __name__ == "__main__":
    main()
