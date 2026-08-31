# Single-vector recall — raw results

The JSON behind the tables in
[`../single_vector_recall.md`](../single_vector_recall.md), so the
numbers can be checked rather than taken on trust. Each file is the
direct output of a script in [`bench/recall/`](../../recall/README.md):
the sweeps come from `recall_sweep.py`, and the id listing comes from
`probe_ids.py`.

| File | Run |
| ---- | --- |
| `nprobes_sweep_wiki1m.json` | 200 held-out queries against the 1,000,000-row namespace, `nprobes` 10, 20, 50 and 100. |
| `repeat_default_wiki1m.json` | An independent 100-query run at the default `nprobes` of 20, on the same namespace, to show how much the figure moves between runs. |
| `nprobes_exhaustive_100k.json` | 200 held-out queries against a 100,000-row namespace, `nprobes` from 1 up to 1000 in one run. Recall rises from 0.63 to 0.6945 between 1 and 10, then stops moving for the remaining hundred-fold increase. |
| `query0_ids_by_nprobes_100k.json` | The row ids the first query returns at each of the nine settings in the sweep above, next to its exact top ten. The same evidence, without averaging. |
| `build_repeat_100k.json` | Three consecutive builds of the index over one unchanged 100,000-row namespace, each scored at the default `nprobes` of 20 over the same 200 queries. The `build` field says which build the entry came from. Recall spans 0.587 to 0.595. |
| `nprobes_exhaustive_100k_build3.json` | The full `nprobes` curve, 1 to 1000, for the third of those builds. `num_partitions` defaulted to `sqrt(100000)` = 316, so the 316 row probes every partition and still returns 0.587. |

Every entry carries the number of queries it averaged, so a reader can
see how much weight the recall figure holds.

The files carry different latency percentiles. `nprobes_sweep_wiki1m.json`
predates the p99 column in the harness and `repeat_default_wiki1m.json`
predates the p95 column. Every file reports p50, which is the column the
report compares.

`cache_hits` is the change in `firnflow_cache_hits_total` for the
namespace across the measured pass. It must be 0, or the latency columns
are cache lookups rather than searches. The two `wiki1m` files predate
that check; every later file carries it and the harness now refuses to
write a result file when the count is not 0.
