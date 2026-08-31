# Single-vector recall against a full scan — 1M Cohere Wikipedia vectors

- **Date**: 2026-08-21 (1,000,000 rows), 2026-08-29 and 2026-08-31 (100,000 rows)
- **Harness**: [`bench/recall/`](../recall/README.md)
- **Firn version**: 0.9.5, built from source at `a5f0a87`. The only change was a `Cargo.lock` bump of `ethnum` 1.5.2 to 1.5.3, because 1.5.2 does not compile on `aarch64-apple-darwin` (`error[E0512]`). No source file was touched.
- **Backend**: MinIO on loopback
- **Corpus**: `CohereLabs/wikipedia-2023-11-embed-multilingual-v3` English split, dim=1024, at revision `ade45fb52bd549f5e8c065636fe4160a43c2af36`. The repository was previously named `Cohere/...` and the old name still redirects. Two namespaces: 1,000,000 rows and 100,000 rows. Shard checksums are pinned in `bench/recall/corpus.py` and verified before every import
- **Index**: IVF_PQ, no tuning options passed. The defaults that follow are 12 partitions over 100,000 rows and 122 over 1,000,000, and `num_sub_vectors` of `dim / 16` = 64. "How many partitions there are" below works through where those come from
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

## How many partitions there are

`nprobes` is the number of index partitions searched per query. The
server default is 20, from `DEFAULT_NPROBES` in
`crates/firnflow-core/src/query.rs`. What that setting can buy depends
on how many partitions the index holds, so that number comes first.

Firn passes no `num_partitions` when it creates an index. LanceDB 0.29
then takes the Lance 6 default, which aims for 8,192 rows in each
IVF_PQ partition and divides the row count by that, with a floor of 1
and a ceiling of 4,096.

**That is 12 partitions over 100,000 rows and 122 over 1,000,000.**

So the default `nprobes` of 20 searches every partition of the
100,000-row index, and 20 of the 122 in the million-row index. A value
above the partition count is capped by it. Asking for 1,000 partitions
of an index that holds 12 searches those 12 and stops.

The default is two steps of library code:

- LanceDB 0.29 falls through to `IvfBuildParams::default()` when
  `num_partitions` is unset:
  [`rust/lancedb/src/table.rs`](https://github.com/lancedb/lancedb/blob/v0.29.0/rust/lancedb/src/table.rs#L1820-L1837)
- Lance 6 gives IVF_PQ a target partition size of 8,192 rows:
  [`rust/lance-index/src/lib.rs`](https://github.com/lance-format/lance/blob/v6.0.0/rust/lance-index/src/lib.rs#L295-L313)
- and turns that into a count with
  `(num_rows / target).clamp(1, 4096)`:
  [`rust/lance-index/src/vector/ivf/builder.rs`](https://github.com/lance-format/lance/blob/v6.0.0/rust/lance-index/src/vector/ivf/builder.rs#L115-L120)

The sweeps below include a 316 setting. It is above the partition count
at both corpus sizes, so it searches the whole index.

## Results

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
neighbours. The table above already argues against it. The million-row
index holds 122 partitions. Going from `nprobes` 20 to 100 takes
coverage from 20 of those partitions to 100, costs 3.9 times the
latency, and buys 1.6 percentage points.

A second run settles it. On a 100,000-row namespace built the same way,
`nprobes` was pushed across three orders of magnitude in a single run:

| nprobes | recall@10 | p50 | p95 | cache hits |
| ------- | --------: | --: | --: | ---------: |
| 1 | 0.4625 | 1.34 ms | 1.97 ms | 0 |
| 2 | 0.5335 | 1.50 ms | 1.69 ms | 0 |
| 5 | 0.5715 | 2.06 ms | 2.29 ms | 0 |
| 10 | 0.5860 | 2.95 ms | 3.17 ms | 0 |
| 20 | 0.5870 | 3.30 ms | 3.57 ms | 0 |
| 50 | 0.5870 | 3.32 ms | 3.60 ms | 0 |
| 100 | 0.5870 | 3.31 ms | 3.61 ms | 0 |
| 316 | 0.5870 | 3.32 ms | 3.56 ms | 0 |
| 1000 | 0.5870 | 3.32 ms | 3.59 ms | 0 |

200 queries per row, one run, same corpus and same settings otherwise.
This is the third of the builds measured under "Between two builds of
the same index" below. The last column is the number of measured
queries the result cache answered. It is zero everywhere, so these are
search timings. Raw data in
[`single_vector_recall_raw/nprobes_exhaustive_100k_build3.json`](single_vector_recall_raw/nprobes_exhaustive_100k_build3.json).

Read the first rows first, because they are the control. Going from 1
partition to 10 adds 12 points of recall, and latency climbs from
1.34 ms to 2.95 ms across the same span. **The setting works.** It
reaches the index and it changes both the answers and the cost.

Then read the rest. This index holds 12 partitions. Every setting from
12 upward searches all of them, so the rows from 20 to 1000 are five
ways of asking for the same search. Recall stops moving between 10 and
20, and latency stops in the same place, which is where the index runs
out of partitions to search.

Per-query ids show the same thing without any averaging. They come from
the earlier build described under "An earlier build, superseded" below,
which is the only build whose per-query ids were recorded. For the first
query, at every one of the eight settings from 2 to 1000, the ten ids
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

At `nprobes` 20 the 100,000-row index searches all 12 of its partitions.
Every row in the table is a candidate and every row is scored. Nothing
is left out by the choice of partitions, and recall is still well short
of 1.0.

That is not the same as a full scan. A full scan compares the query
against the stored 4,096-byte vectors and returns the exact answer, by
definition. Searching every partition of this index compares the query
against 64-byte approximations of those vectors. The candidate set is
the whole table either way. Only the scoring differs, so only the
scoring can account for the gap.

So the cause is the compression. Distances computed from 64 bytes are
close enough to gather a plausible set of candidates and too coarse to
order them correctly. `nprobes` chooses which candidates are considered,
not how they are scored, and no setting of it corrects a scoring error.

Two things this does not settle.

It does not measure whether a re-scoring pass against the stored
full-precision vectors would recover the missing rows. That pass is not
part of this measurement.

It does not carry to the million-row index. That one holds 122
partitions and the default `nprobes` searches 20 of them, so a narrow
search and a coarse score are both in play there and this run does not
separate them.

## A cheaper way to see the same thing

The million-row run takes a few hours end to end. The 100,000-row run
above takes a few minutes: one shard to download, one shard to load, one
index build, one ground-truth scan.

That smaller run carries the same finding. Recall at 100,000 rows is
0.587 to 0.595 across three builds, against 0.559 over a million rows at
the same setting. The flat response to `nprobes` is already complete
there, and so is the per-query evidence that specific true neighbours
are never returned at any setting.

## Between two builds of the same index

An IVF index groups the vectors into partitions using k-means, and
k-means starts from centroids chosen at random. Two builds over
identical rows produce different partitions. Recall moves between
builds, so a figure from a single build cannot be separated from one
lucky or unlucky set of starting centroids.

Three consecutive builds of the same 100,000-row namespace, each scored
at the default `nprobes` of 20 over the same 200 queries
([`single_vector_recall_raw/build_repeat_100k.json`](single_vector_recall_raw/build_repeat_100k.json)):

```
build   recall@10   p50 ms   cache_hits
    1       0.592     3.41            0
    2       0.595     3.31            0
    3       0.587     3.33            0
```

Nothing was reloaded between builds. `POST /ns/{ns}/index` replaces the
index in place, and each build took about 60 seconds over 100,000 rows.

The spread across these three builds is 0.8 percentage points, 0.587 to
0.595. Treat the second decimal of a single recall figure as noise of
about that size. The gap this file reports is around 40 percentage
points, so it does not depend on which of these builds was measured.

That 0.8 figure covers these three builds and nothing else. It does not
cover the earlier build described next.

### An earlier build, superseded

An earlier build of a 100,000-row namespace was swept the same way and
scored 0.6945 from `nprobes` 10 upward
([`single_vector_recall_raw/nprobes_exhaustive_100k.json`](single_vector_recall_raw/nprobes_exhaustive_100k.json)):

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

That is 10.75 percentage points above build 3, where the three repeated
builds differ from each other by 0.8. The build no longer exists, so the
difference cannot be traced. It was not reproduced.

**This run is superseded.** Its recall level supports nothing in this
file and is not part of the build-variance figure above. One thing is
still drawn from this build: the per-query id listing under "Why more
probing does not help", because its ids are the only ones recorded. That
listing is about which rows came back at each setting, not about the
level, and it is marked there. The raw data stays in place as a record
of what was measured.

## Limits

- MinIO on loopback, not a real object store. The latency columns are
  lower than they would be against S3. The recall column does not depend
  on the storage backend.
- One dataset, two corpus sizes, one index configuration. Nothing here
  varies `num_partitions`, `num_sub_vectors` or `num_bits`. Three builds
  of that one configuration were measured; every other figure in this
  file comes from a single build.
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
