"""Score the index against the exact answer at several nprobes settings.

For each query, this sends the same vector to POST /ns/{ns}/query and
counts how many of the returned ids appear in the exact top-k that
ground_truth.py computed. That fraction, averaged over all queries, is
recall@k.

    recall@10 = 1.0   the index returned exactly the true 10 nearest rows
    recall@10 = 0.5   half of what it returned is not among the true 10

This is not the recall in the BEIR reports under bench/results/. That one
asks whether a human labelled the returned document relevant. The two can
disagree: an index can return a different relevant document and score
well on relevance while missing the true nearest row.

Two things keep the measurement honest.

The measured pass sends the ground-truth vectors unchanged. Recall is
then a comparison between the index's answer and the exact answer for
the same vector, with nothing in between.

The warm-up pass before it sends perturbed copies instead. That pulls the
index and data files onto local disk without leaving result-cache entries
under the keys the measured pass will use. Each setting then reads
firnflow_cache_hits_total from /metrics before and after its measured
pass and records the difference as `cache_hits`.

A non-zero `cache_hits` at any setting fails the run. The script writes
nothing to the output path, saves the readings to OUTPUT.rejected for
inspection, and exits non-zero. Those latency figures are cache lookups
rather than search timings, and a warning on standard output is too easy
to scroll past on the way to a committed result. See the README for how
to clear the cache.

Every response is checked before it is scored. A query that asks for k
neighbours must come back with exactly k results, all with distinct ids.
A short response, or one padded with a repeated id, cannot reach recall
1.0 however good the index is, so scoring it would file a broken server
under low recall. The first query that fails stops the run before that
response is scored, and the run is rejected the same way a cache hit
rejects it.

Usage:
    python recall_sweep.py NAMESPACE TRUTH.npz K NPROBES_LIST OUTPUT.json

NPROBES_LIST is comma-separated, for example "1,2,5,10,20,50,100,1000".
The server default is 20 when the field is omitted.

Example:
    python recall_sweep.py wiki1m truth-1m.npz 10 10,20,50,100 sweep.json
"""

import json
import statistics
import sys
import time

import numpy as np
import requests

from corpus import BASE_URL, auth_headers, cache_hit_count, percentile, unit_normalise

#: Added to every component of a warm-up query so the warm-up does not
#: populate the result cache under the measured query's key. Far too
#: small to move a unit-length vector's neighbourhood, and it never
#: touches a measured query in any case.
WARMUP_OFFSET = np.float32(1e-3)


def run_query(session, namespace, vector, k, nprobes):
    """Send one vector query and time it.

    Args:
        session: a ``requests.Session`` so the connection is reused.
        namespace: namespace to query.
        vector: the query vector as a list of floats.
        k: how many neighbours to ask for.
        nprobes: how many index partitions to search.

    Returns:
        A ``(ids, milliseconds)`` pair.
    """
    body = {"vector": vector, "k": k, "nprobes": nprobes, "include_vector": False}
    started = time.perf_counter()
    response = session.post(
        f"{BASE_URL}/ns/{namespace}/query", json=body, headers=auth_headers(), timeout=600
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    return [hit["id"] for hit in response.json()["results"]], elapsed_ms


def check_ids(ids, k):
    """Say why one query's response cannot be scored, or return None.

    Args:
        ids: the row ids the server returned for one query.
        k: how many neighbours that query asked for.

    Returns:
        A short reason string when the response is unusable, otherwise
        None. Recall divides the overlap with the exact answer by k, so
        a response carrying fewer than k distinct ids cannot reach 1.0
        whatever the index does. Scoring it would turn a server fault
        into a low recall figure with nothing in the output to say so.
    """
    if len(ids) != k:
        return f"returned {len(ids)} results, asked for {k}"
    distinct = len(set(ids))
    if distinct != k:
        return f"returned {k} results holding only {distinct} distinct ids"
    return None


def score_setting(session, namespace, queries, truth_ids, k, nprobes):
    """Measure recall@k and latency for one nprobes setting.

    Args:
        session: a ``requests.Session``.
        namespace: namespace to query.
        queries: ``(num_queries, dim)`` float32 query matrix.
        truth_ids: ``(num_queries, k)`` int64 exact nearest ids.
        k: how many neighbours to ask for.
        nprobes: how many index partitions to search.

    Returns:
        A ``(row, fault)`` pair, one of which is always None. On a clean
        setting the row holds recall, latency percentiles and the
        result-cache hit count observed during the measured pass. On the
        first query `check_ids` rejects, the row is None and the fault
        names that query. Nothing is scored past that point, so an
        unscorable response never reaches the recall average.
    """
    warmup = unit_normalise(queries + WARMUP_OFFSET)
    for vector in warmup:
        run_query(session, namespace, vector.tolist(), k, nprobes)

    hits_before = cache_hit_count(namespace)
    overlaps, latencies = [], []
    for index, vector in enumerate(queries):
        ids, elapsed_ms = run_query(session, namespace, vector.tolist(), k, nprobes)
        reason = check_ids(ids, k)
        if reason is not None:
            return None, {"nprobes": nprobes, "query": index, "reason": reason}
        truth = set(truth_ids[index].tolist())
        overlaps.append(len(truth & set(ids)) / len(truth))
        latencies.append(elapsed_ms)
    hits_after = cache_hit_count(namespace)

    ordered = sorted(latencies)
    row = {
        "nprobes": nprobes,
        f"recall@{k}": round(float(np.mean(overlaps)), 4),
        "queries": len(queries),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(ordered, 0.95), 2),
        "p99_ms": round(percentile(ordered, 0.99), 2),
        "cache_hits": hits_after - hits_before,
    }
    return row, None


def load_truth(path, k):
    """Load a ground-truth file and check it can answer a top-k question.

    Args:
        path: the .npz written by ground_truth.py.
        k: the k the sweep is about to measure.

    Returns:
        A ``(queries, truth_ids)`` pair, with ``truth_ids`` trimmed to k.

    Raises:
        SystemExit: if the file holds fewer than k neighbours per query.
            Dividing by k when only j < k exist would score a perfect
            answer as j/k and understate recall.
    """
    truth = np.load(path)
    queries, truth_ids = truth["queries"], truth["ids"]
    if truth_ids.shape[1] < k:
        raise SystemExit(
            f"{path} holds only {truth_ids.shape[1]} neighbours per query, "
            f"which cannot score recall@{k}. Re-run ground_truth.py with "
            f"K of at least {k}, over a corpus of at least {k} rows."
        )
    return queries, truth_ids[:, :k]


def reject(output, report, faults, reason):
    """Save a run that must not be committed, then stop the script.

    Args:
        output: the path the caller asked for. Nothing is written there,
            so only a clean run can ever be committed by mistake.
        report: the settings measured before the run was rejected.
        faults: the per-query rejections from `check_ids`, if any.
        reason: what was wrong, printed as the exit message.

    Raises:
        SystemExit: always. This is the only way the script reports a
            measurement it does not stand behind.
    """
    rejected = output + ".rejected"
    with open(rejected, "w") as handle:
        json.dump({"settings": report, "bad_responses": faults}, handle, indent=2)
    raise SystemExit(
        f"{reason} Nothing was written to {output}. The readings are in "
        f"{rejected} for inspection."
    )


def main():
    """Sweep every nprobes setting and write the results as JSON.

    Raises:
        SystemExit: if any query came back unscorable, or if the result
            cache answered any measured query.
    """
    namespace, truth_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nprobes_list = [int(value) for value in sys.argv[4].split(",")]
    output = sys.argv[5]

    queries, truth_ids = load_truth(truth_path, k)
    session = requests.Session()

    report = []
    for nprobes in nprobes_list:
        row, fault = score_setting(session, namespace, queries, truth_ids, k, nprobes)
        if fault is not None:
            reject(
                output,
                report,
                [fault],
                f"at nprobes {nprobes}, query {fault['query']} "
                f"{fault['reason']}. Recall divides by {k}, so scoring that "
                f"response would record a server that answered wrongly as an "
                f"index that missed. This setting and the ones after it were "
                f"left unmeasured.",
            )
        report.append(row)
        print(json.dumps(row), flush=True)

    stale = [row["nprobes"] for row in report if row["cache_hits"]]
    if stale:
        reject(
            output,
            report,
            [],
            f"the result cache answered queries at nprobes {stale}, so the "
            f"latency columns for those settings are cache lookups and not "
            f"search timings. Clear the cache and run again. The README "
            f"section 'Keeping the result cache out of the latency figures' "
            f"says how.",
        )

    with open(output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
