"""Print the row ids one query returns at several nprobes settings.

The sweep reports recall averaged over hundreds of queries. When that
average stops moving as nprobes rises, this shows why in a form you can
read directly: the actual ids, side by side, one query at a time.

If the id lists are identical from some setting upward, the search has
stopped finding anything new and a larger nprobes cannot help. Comparing
them against the exact answer then shows how much the index is still
missing.

Usage:
    python probe_ids.py NAMESPACE TRUTH.npz QUERY_INDEX K NPROBES_LIST [OUTPUT.json]

Example:
    python probe_ids.py wiki1m truth-1m.npz 0 10 1,2,5,10,100,1000
"""

import json
import sys

import numpy as np
import requests

from corpus import BASE_URL, auth_headers


def query_ids(namespace, vector, k, nprobes):
    """Return the ids one query gives at one nprobes setting.

    Args:
        namespace: namespace to query.
        vector: the query vector as a list of floats.
        k: how many neighbours to ask for.
        nprobes: how many index partitions to search.

    Returns:
        The returned row ids, closest first.
    """
    response = requests.post(
        f"{BASE_URL}/ns/{namespace}/query",
        json={"vector": vector, "k": k, "nprobes": nprobes, "include_vector": False},
        headers=auth_headers(),
        timeout=600,
    )
    response.raise_for_status()
    return [hit["id"] for hit in response.json()["results"]]


def main():
    """Query one vector at each setting and print the ids against the truth."""
    namespace, truth_path = sys.argv[1], sys.argv[2]
    query_index, k = int(sys.argv[3]), int(sys.argv[4])
    nprobes_list = [int(value) for value in sys.argv[5].split(",")]
    output = sys.argv[6] if len(sys.argv) > 6 else None

    truth = np.load(truth_path)
    available = truth["ids"].shape[1]
    if available < k:
        raise SystemExit(
            f"{truth_path} holds only {available} neighbours per query, "
            f"which cannot answer a top-{k} comparison. Re-run "
            f"ground_truth.py with K of at least {k}."
        )
    vector = truth["queries"][query_index].tolist()
    exact = truth["ids"][query_index][:k].tolist()

    report = {"query_index": query_index, "k": k, "exact": exact, "returned": {}}
    for nprobes in nprobes_list:
        ids = query_ids(namespace, vector, k, nprobes)
        report["returned"][str(nprobes)] = ids
        found = len(set(exact) & set(ids))
        print(f"nprobes {nprobes:>5}  found {found}/{k}  {ids}", flush=True)
    print(f"{'exact':>13}  {'':>11}  {exact}")

    if output:
        with open(output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
