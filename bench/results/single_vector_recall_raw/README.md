# Single-vector recall — raw results

The JSON behind the tables in
[`../single_vector_recall.md`](../single_vector_recall.md), so the
numbers can be checked rather than taken on trust. Each file is the
direct output of a script in [`bench/recall/`](../../recall/README.md):
the three sweeps come from `recall_sweep.py`, and the id listing comes
from `probe_ids.py`.

| File | Run |
| ---- | --- |
| `nprobes_sweep_wiki1m.json` | 200 held-out queries against the 1,000,000-row namespace, `nprobes` 10, 20, 50 and 100. |
| `repeat_default_wiki1m.json` | An independent 100-query run at the default `nprobes` of 20, on the same namespace, to show how much the figure moves between runs. |
| `nprobes_exhaustive_100k.json` | 200 held-out queries against a 100,000-row namespace, `nprobes` from 1 up to 1000 in one run. Recall rises from 0.63 to 0.6945 between 1 and 10, then stops moving for the remaining hundred-fold increase. |
| `query0_ids_by_nprobes_100k.json` | The row ids the first query returns at each of the nine settings in the sweep above, next to its exact top ten. The same evidence, without averaging. |

Every entry carries the number of queries it averaged, so a reader can
see how much weight the recall figure holds.

The files carry different latency percentiles. `nprobes_sweep_wiki1m.json`
predates the p99 column in the harness and `repeat_default_wiki1m.json`
predates the p95 column. All three report p50, which is the column the
report compares.
