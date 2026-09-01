# Single-vector recall: raw results

The JSON behind the tables in
[`../single_vector_recall.md`](../single_vector_recall.md), so the
numbers can be checked rather than taken on trust. Each file is the
direct output of a script in [`bench/recall/`](../../recall/README.md):
the sweeps come from `recall_sweep.py`, and the id listing comes from
`probe_ids.py`.

| File | Run |
| ---- | --- |
| `flat_scan_100k_validation.json` | Maintainer validation control: 20 held-out queries against the unindexed 100,000-row namespace. Recall@10 is 1.0 and measured result-cache hits are zero. |
| `build_repeat_100k_validation.json` | Maintainer validation: three fresh default IVF_PQ builds over the unchanged 100,000-row namespace, each scored at `nprobes` 20 over the same 200 queries. Recall spans 0.5880 to 0.5935. |
| `nprobes_exhaustive_100k_validation.json` | Maintainer validation: the third fresh build swept from `nprobes` 1 to 1000 over 200 queries. Recall reaches 0.5935 at 10 and remains there. Every measured cache-hit count is zero. |
| `query0_ids_by_nprobes_100k_validation.json` | Maintainer validation: the fresh per-setting ids for the first query, next to its exact top ten. |
| `validation_environment_100k.json` | Machine-readable record of the validated server and storage image digests, pinned dataset and truth checksums, namespace state, actual index type and coverage, source-derived index parameters, cache settings, host environment, package versions, harness checksums and result checksums. |
| `nprobes_sweep_wiki1m.json` | 200 held-out queries against the 1,000,000-row namespace, `nprobes` 10, 20, 50 and 100. |
| `repeat_default_wiki1m.json` | An independent 100-query run at the default `nprobes` of 20, on the same namespace, to show how much the figure moves between runs. |
| `nprobes_exhaustive_100k.json` | **Superseded.** An earlier 200-query sweep that reached 0.6945 Recall@10. The result was not reproduced and lacks the final validation safeguards. It is retained for auditability and supports no conclusion in the report. |
| `query0_ids_by_nprobes_100k.json` | **Superseded.** Per-setting ids associated with the earlier unreproduced build. Retained for auditability. |
| `build_repeat_100k.json` | **Superseded.** Three earlier builds that predate the final full-scan control and environment record. Retained for auditability. |
| `nprobes_exhaustive_100k_build3.json` | **Superseded.** The exhaustive sweep for the third earlier build. Retained for auditability. |

Every entry carries the number of queries it averaged, so a reader can
see how much weight the recall figure holds.

The files carry different latency percentiles. `nprobes_sweep_wiki1m.json`
predates the p99 column in the harness and `repeat_default_wiki1m.json`
predates the p95 column. Latency from different machines must not be
compared.

`cache_hits` is the change in `firnflow_cache_hits_total` for the
namespace across the measured pass. It must be 0, or the latency columns
are cache lookups rather than searches. The two `wiki1m` files predate
that check; every later file carries it and the harness now refuses to
write a result file when the count is not 0. It also refuses to reuse an
existing success or rejection path, so a failed rerun cannot leave an
older result looking current.
