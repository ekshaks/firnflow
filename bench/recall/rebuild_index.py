"""Build or rebuild a namespace's vector index through POST /ns/{ns}/index.

Two reasons this exists rather than the plain curl the README used to
give.

The first is repeatability. An IVF index divides the vectors into
partitions with k-means, and k-means starts from centroids chosen at
random. Two builds over identical rows produce different partitions, so
recall moves between builds. A single build cannot say whether a recall
figure is a property of the engine or of one lucky or unlucky set of
centroids. Running this script between sweeps gives a spread instead of
a point.

The second is that a rebuild replaces the existing index in place, so
the same namespace can be measured several times without loading the
corpus again. Firn keys its result cache on the Lance table version, and
committing a new index moves that version, so the earlier sweep's cached
answers are not reachable by the next one. The sweep checks this rather
than trusting it: it reads the cache-hit counter around every measured
pass and fails if the counter moves.

Usage:
    python rebuild_index.py NAMESPACE [OPTIONS_JSON]

OPTIONS_JSON is merged into the request body. Omit it to reproduce the
report, which passed no tuning options and took the defaults: one
partition per 8,192 rows, so 12 partitions over 100,000 rows and 122
over 1,000,000, and `num_sub_vectors` of dim / 16.

Example, three builds of the same corpus scored separately:
    for build in 1 2 3; do
      python rebuild_index.py wiki100k
      python recall_sweep.py wiki100k truth.npz 10 20 sweep-$build.json
    done

Example with an explicit partition count:
    python rebuild_index.py wiki100k '{"num_partitions": 128}'

Index creation is an admin route. Export FIRNFLOW_API_KEY if the server
was started with a key.
"""

import json
import sys
import time

import requests

from corpus import BASE_URL, auth_headers, namespace_row_count, wait_for_operation


def build_index(namespace, options):
    """Start an index build and wait for it to finish.

    Args:
        namespace: namespace to index.
        options: dict merged into the request body over `kind`.

    Returns:
        Seconds the build took, measured from the request to the
        operation leaving the running state.

    Raises:
        SystemExit: if the request is rejected or the build fails.
    """
    body = {"kind": "ivf_pq", **options}
    started = time.time()
    response = requests.post(
        f"{BASE_URL}/ns/{namespace}/index",
        json=body,
        headers=auth_headers(),
        timeout=120,
    )
    if not response.ok:
        raise SystemExit(
            f"POST /ns/{namespace}/index returned {response.status_code}: "
            f"{response.text[:300]}"
        )
    wait_for_operation(response.json()["operation_id"])
    return time.time() - started


def main():
    """Rebuild one namespace's index and print how long it took.

    Raises:
        SystemExit: if the namespace is missing or empty. Building an
            index over no rows succeeds and leaves queries on the
            full-scan path, which returns the exact answer and would
            report recall 1.0 while measuring nothing.
    """
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    namespace = sys.argv[1]
    options = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    rows = namespace_row_count(namespace)
    if not rows:
        raise SystemExit(
            f"namespace {namespace} holds no rows"
            f"{' and does not exist' if rows is None else ''}. Seed it first "
            f"with seed_namespace.py. An index over an empty namespace leaves "
            f"queries on the full-scan path, which is the exact answer, and "
            f"the sweep would report recall 1.0 while measuring nothing."
        )

    seconds = build_index(namespace, options)
    print(
        f"{namespace}: index built over {rows} rows in {seconds:.1f}s, "
        f"options {json.dumps(options)}"
    )


if __name__ == "__main__":
    main()
