# Single-vector recall against a full scan — 1M Cohere Wikipedia vectors

- **Date**: 2026-08-21 (1,000,000 rows), 2026-08-29 (100,000 rows)
- **Harness**: [`bench/recall/`](../recall/README.md)
- **Firn version**: 0.9.5, built from source at `a5f0a87`, unmodified
- **Backend**: MinIO on loopback
- **Corpus**: `Cohere/wikipedia-2023-11-embed-multilingual-v3` English split, dim=1024. Two namespaces: 1,000,000 rows and 100,000 rows
- **Index**: IVF_PQ, no tuning options passed, `num_sub_vectors` defaulted to `dim / 16` = 64
- **Queries**: held-out vectors from a shard that was never loaded, `k=10`, `include_vector: false`. 200 per run, except one repeat run of 100 noted below
- **Raw data**: [`single_vector_recall_raw/`](single_vector_recall_raw/)

## What recall@10 is here

There are two ways to answer "which rows are closest to this vector".

The first compares the query against every row in the table and keeps
the closest ten. That is the correct answer by definition.

The second is the index. It searches only some of the partitions, and it
stores each vector in a compressed form. Both shortcuts can change which
rows come back.

Recall@10 counts how many of the index's ten results also appear in the
exact top ten, divided by ten. A value of 1.0 means the index agreed
with the exact scan. A value of 0.5 means half the rows it returned are
not among the true ten nearest.

This is not the `recall@10` in the BEIR reports in this directory. That
one asks whether a human labelled the returned document relevant. An
index can return a different relevant document, score well on relevance,
and still return the wrong rows.

## Results

`nprobes` is the number of IVF partitions searched per query. The server
default is 20, from `DEFAULT_NPROBES` in `crates/firnflow-core/src/query.rs`.

| nprobes | recall@10 | p50 | p95 |
| ------- | --------: | --: | --: |
| 10 | 0.536 | 4.63 ms | 11.03 ms |
| **20** (default) | **0.559** | **4.72 ms** | **5.82 ms** |
| 50 | 0.5725 | 9.86 ms | 10.89 ms |
| 100 | 0.575 | 18.47 ms | 18.77 ms |

200 queries per row. A separate 100-query run at the default scored
0.528, so the range across every run is 0.53 to 0.58. Treat the second
decimal as noise at this sample size.

One caveat on this table only. It was measured before the harness
started recording result-cache hits, so its latency column has no
zero-hit check behind it and may be optimistic. The 100,000-row table
below does carry that check. Recall is unaffected either way, because a
cached result is the same result.

**The default configuration returns about five of the true ten nearest
rows.**

## Why more probing does not help

An obvious first guess is that the index is only looking at too small a
slice of the rows, and that a larger `nprobes` would recover the missing
neighbours. The table above already argues against it. Going from 20
partitions to 100 searches five times as much of the index, costs 3.9
times the latency, and buys 1.6 percentage points.

A second run settles it. On a 100,000-row namespace built the same way,
`nprobes` was pushed across three orders of magnitude in a single run:

| nprobes | recall@10 | p50 | p95 | cache hits |
| ------- | --------: | --: | --: | ---------: |
| 1 | 0.6300 | 1.14 ms | 1.26 ms | 0 |
| 2 | 0.6915 | 1.29 ms | 1.39 ms | 0 |
| 5 | 0.6940 | 1.78 ms | 1.92 ms | 0 |
| 10 | 0.6945 | 2.64 ms | 2.79 ms | 0 |
| 20 | 0.6945 | 2.91 ms | 3.07 ms | 0 |
| 50 | 0.6945 | 2.91 ms | 3.04 ms | 0 |
| 100 | 0.6945 | 2.91 ms | 3.11 ms | 0 |
| 316 | 0.6945 | 2.93 ms | 3.05 ms | 0 |
| 1000 | 0.6945 | 2.92 ms | 3.06 ms | 0 |

200 queries per row, one run, same corpus and same settings otherwise.
The last column is the number of measured queries the result cache
answered. It is zero everywhere, so these are search timings. Raw data
in
[`single_vector_recall_raw/nprobes_exhaustive_100k.json`](single_vector_recall_raw/nprobes_exhaustive_100k.json).

Read the first rows first, because they are the control. Going from 1
partition to 2 adds six points of recall, and latency climbs from
1.14 ms to 2.91 ms as the setting rises to 20. **The setting works.** It
reaches the index and it changes both the answers and the cost.

Then read the rest. Recall stops moving at `nprobes` 10 and latency
stops at 20. Beyond that a fifty-fold increase costs nothing and buys
nothing.

Per-query ids show the same thing without any averaging. For the first
query, at every one of the nine settings from 2 to 1000, the ten ids
returned are the same ten, and six of them match the exact answer:

```
nprobes     1  found 6/10  [62829, 67697, 59721, 47314, 20498, ...]
nprobes     2  found 6/10  [62829, 67697, 59721, 24449, 47314, ...]
nprobes  1000  found 6/10  [62829, 67697, 59721, 24449, 47314, ...]
exact                      [62829, 74773, 67697, 59719, 59721, ...]
```

Full lists for all nine settings in
[`single_vector_recall_raw/query0_ids_by_nprobes_100k.json`](single_vector_recall_raw/query0_ids_by_nprobes_100k.json).
Rows 74773, 24462, 67843 and 62944 are among the true ten nearest and
are returned at no setting at all.

## What that rules out, and what it does not

Two explanations are ruled out by these numbers.

The server is not falling back to a full scan at high `nprobes`. A full
scan gives the exact answer, and recall is 0.6945, not 1.0.

The missing rows are not being lost to a search that is too narrow in a
way `nprobes` can fix. Whatever the setting does, it stops doing it
above 20, and four of the true ten nearest rows are absent at every
setting including the largest.

The likely remaining cause is the compression. Each vector holds 4,096
bytes of original data and the index stores it as 64 bytes, a 64-fold
reduction. Distances computed from those 64 bytes are approximations,
close enough to gather a plausible set of candidates and too coarse to
order them correctly. `nprobes` controls which candidates are
considered, not how they are scored, so no setting of it corrects a
scoring error. Confirming this needs a re-scoring pass against the
stored full-precision vectors, which is not part of this measurement.

One thing here is unexplained and worth flagging rather than smoothing
over. `num_partitions` is documented in
`crates/firnflow-core/src/query.rs` as defaulting to `sqrt(row_count)`,
which is 316 for this namespace. If `nprobes` 316 really searched every
partition, the cost should follow the slope of the low end: about
0.09 ms per partition from 1 to 20 predicts roughly 30 ms at 316. The
measurement is 2.93 ms. Either the partition count is far below 316, or
`nprobes` stops taking effect above about 20. The same flat response
appears in this repository's own `nprobes_sweep_fiqa` results. This
harness cannot tell those apart from the outside.

## A cheaper way to see the same thing

The million-row run takes a few hours end to end. The 100,000-row run
above takes a few minutes: one shard to download, one shard to load, one
index build, one ground-truth scan.

That smaller run carries the same finding. Recall at 100,000 rows is
0.6945, higher than at a million rows, which is the direction expected
if compression error grows with the corpus. The flat response to
`nprobes` is already complete there, and so is the per-query evidence
that specific true neighbours are never returned at any setting.

## Limits

- MinIO on loopback, not a real object store. The latency columns are
  lower than they would be against S3. The recall column does not depend
  on the storage backend.
- One dataset, two corpus sizes, one index configuration. Nothing here
  varies `num_partitions`, `num_sub_vectors` or `num_bits`.
- Single-vector namespaces only. Nothing here touches the multivector
  path measured in `beir_multivector_objcache.md`.
- Single-client, sequential queries. This is not a throughput
  measurement.
- The hardware was a laptop: Apple M3 Pro, 11 cores, 18 GB RAM, with the
  server, MinIO and the harness all on the same machine.
- The two corpus sizes were measured a week apart on the same machine,
  so their latency columns are not directly comparable to each other.
  The recall figures within each run are.

## Reproducing

[`bench/recall/README.md`](../recall/README.md) has the full procedure:
download the shards, load ten of them, build the index, compute the
exact answers from the eleventh, and score the index against them.

The embeddings ship pre-computed with the dataset, so the run needs no
embedding model and no API key.
