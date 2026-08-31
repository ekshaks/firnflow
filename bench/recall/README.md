# Single-vector recall harness

## What this measures

There are two ways to answer "which rows are closest to this vector".

The first compares the query against every row in the table and keeps
the closest ten. That is the correct answer, by definition. It is also
slow, because it reads everything.

The second is what an index does. An IVF_PQ index splits the rows into
partitions and searches only some of them, and it stores each vector in
a compressed form so more of the index fits in memory. Both shortcuts
can change which rows come back.

**Recall@10 is the overlap between the two.** Of the ten rows the exact
scan says are closest, how many did the index return, divided by ten.

- `1.0` means the index agreed with the exact scan on all ten.
- `0.5` means half the rows it returned are not among the true ten
  nearest.

A low value is not a slow query. It is a wrong answer returned quickly.

## What this is not

The BEIR reports under `bench/results/` also print a number called
`recall@10`. It answers a different question: of the documents a human
labelled relevant to a search, how many came back.

The two can disagree. A compressed index often returns a *different*
relevant document in place of the right one, which leaves the human-
relevance score untouched while the returned rows are wrong. Scoring
well on one says nothing about the other.

## The scripts

| Script | Does |
| ------ | ---- |
| `download_shards.py` | Fetches Cohere Wikipedia parquet shards at a pinned dataset revision and verifies each one's sha256. |
| `seed_namespace.py` | Loads shards into a namespace through `POST /ns/{ns}/import`. Refuses a namespace that already holds rows. |
| `rebuild_index.py` | Builds or rebuilds the vector index, so the same corpus can be measured across several builds. |
| `ground_truth.py` | Computes the exact top-k of held-out queries over every loaded row. Verifies every shard it is given and refuses a query shard that is also in the corpus. |
| `recall_sweep.py` | Queries the server and scores recall@k at several `nprobes` settings. Fails if the result cache answered anything, or if a query came back with anything other than k distinct ids. |
| `probe_ids.py` | Prints the ids one query returns at each setting, next to the exact answer. |
| `corpus.py` | Shared helpers, the pinned dataset revision and the shard checksums. Not run directly. |

Every script exits non-zero and explains itself rather than producing a
number that cannot be trusted. The six cases are a namespace that
already holds rows, a shard whose checksum does not match, a query shard
that is also part of the corpus, a measured pass that the result cache
answered, a query answered with anything other than k distinct ids, and
an index build over an empty namespace.

Every script reads two environment variables:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `FIRNFLOW_URL` | `http://127.0.0.1:3000` | Base URL of the server under test. |
| `FIRNFLOW_API_KEY` | unset | Bearer token. Only needed if the server was started with a key. |

Everything else is a command-line argument. No paths are baked in.

## The dataset

`CohereLabs/wikipedia-2023-11-embed-multilingual-v3`, English split. Each
shard is 100,000 rows, 1024 dimensions per row, about 216 MB on disk.
The embeddings ship pre-computed, so no embedding model and no API key
are involved, and two people running this get the same vectors.

The dataset was published as `Cohere/wikipedia-2023-11-embed-multilingual-v3`
and renamed. The old name still redirects.

Downloads are pinned to one commit, not to the `main` branch. A branch
pointer moves: this dataset was last modified on 2026-03-25, after the
million-row run recorded in `bench/results/`. The commit and the sha256
of every shard live in `corpus.py`:

```python
DATASET_REVISION = "ade45fb52bd549f5e8c065636fe4160a43c2af36"
SHARD_MANIFEST = {
    0: ("2c6abfffa7dd336113251b3e6f3fe4ee16688ead7c67b99593cbadc5589e28b3", 216612385),
    ...
}
```

The downloader checks both after fetching, `seed_namespace.py` checks
them again before importing, and either check failing stops the run. A
recall figure is only meaningful next to a statement of which bytes
produced it.

The manifest covers shards 0 to 10, which is what the committed runs
used. To use another shard, add its entry first. The comment above
`SHARD_MANIFEST` has the one `curl` command that reads the digest and
the size from the Hugging Face API, without downloading the file.

turbopuffer benchmarks itself on the same dataset at the same width and
the same `top_k`, in
[`benchmarks/vector-knn-1m-hot.toml`](https://github.com/turbopuffer/tpuf-benchmark/blob/main/benchmarks/vector-knn-1m-hot.toml)
and
[`benchmarks/website/vector-10m-hot.toml`](https://github.com/turbopuffer/tpuf-benchmark/blob/main/benchmarks/website/vector-10m-hot.toml).
One difference: their query vectors are generated pseudorandomly, while
the ones here are real embeddings from a held-out shard.

Eleven shards cover the million-row run in
`bench/results/single_vector_recall.md`: shards 0 through 9 are loaded,
shard 10 supplies the query vectors. That is about 2.4 GB of parquet to
download and roughly 20 GB of Lance data and index in the bucket.

**Two shards are enough to see the result.** Load one, query with the
other, and the whole run takes minutes instead of hours. The section
"Checking whether more probing helps" below explains why the small run
carries the same conclusion.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r bench/recall/requirements.txt
```

## Running it

The commands below assume the repository root as the working directory
and use `wiki1m` as the namespace name. Substitute your own.

### 1. Storage and server

The compose stack in the repository root brings up MinIO with the bucket
already created:

```bash
docker compose up -d minio minio-init
```

Then run the server from the checkout, rather than the compose
container, so the cache budgets can be set for a corpus this size:

```bash
cargo build --release -p firnflow-api

export FIRNFLOW_BIND=127.0.0.1:3000
export FIRNFLOW_STORAGE_URI=s3://firnflow
export FIRNFLOW_S3_ENDPOINT=http://127.0.0.1:9000
export FIRNFLOW_S3_ACCESS_KEY=minioadmin
export FIRNFLOW_S3_SECRET_KEY=minioadmin
export FIRNFLOW_S3_REGION=us-east-1

# Keep the Lance data and index files on local disk after the first
# read. Without this every query re-fetches them from the bucket and
# the latency column measures the object store, not the index.
export FIRNFLOW_OBJECT_CACHE_ENABLED=true
export FIRNFLOW_OBJECT_CACHE_DIR=./bench-cache/objects
export FIRNFLOW_OBJECT_CACHE_BYTES=21474836480   # 20 GiB

export FIRNFLOW_CACHE_NVME_PATH=./bench-cache/results
export FIRNFLOW_CACHE_MEMORY_BYTES=67108864      # 64 MiB
export FIRNFLOW_CACHE_NVME_BYTES=268435456       # 256 MiB

# The import route spools the request body here before committing it.
# One shard is about 400 MB as an Arrow stream.
export FIRNFLOW_IMPORT_TMP_DIR=./bench-cache/import

mkdir -p ./bench-cache/objects ./bench-cache/results ./bench-cache/import
./target/release/firnflow-api
```

`FIRNFLOW_IMPORT_MAX_BYTES` needs no change. Its 8 GiB default is well
above the per-shard body size.

### 2. Download the shards

```bash
python bench/recall/download_shards.py ./bench-data 0 10
```

Shards 0 to 9 become the corpus. Shard 10 supplies the query vectors and
is never loaded.

Each shard is checked against its pinned sha256 as it arrives. A file
already on disk is re-checked rather than assumed, so the command is
safe to re-run after an interrupted download. A mismatch stops the
script instead of leaving bytes a later run would accept.

### 3. Load the corpus

```bash
python bench/recall/seed_namespace.py wiki1m ./bench-data/en000{0,1,2,3,4,5,6,7,8,9}.parquet
```

The namespace is created by the first import and its dimension is fixed
at 1024. Ids run from 0 upward in the order the shards are given.

**The script refuses to run if the namespace already holds rows.**
`/import` appends. It does not replace, and unlike `/upsert` it does not
merge on `id`. Ids here are positional from 0, so a second run would put
two rows under every id and every recall figure measured afterwards
would be wrong with nothing in the output to say so. To start over,
delete the namespace first:

```bash
curl -X DELETE http://127.0.0.1:3000/ns/wiki1m
```

The shard checksums are verified a second time here, before the first
byte is sent. Afterwards the script compares the row count the server
reports against the number of rows it sent, and fails if they differ.

### 4. Build the index

```bash
python bench/recall/rebuild_index.py wiki1m
```

Index creation is an admin route. Export `FIRNFLOW_API_KEY` if the
server was started with a key.

No tuning options are passed, so the index takes the LanceDB defaults.
The partition count is the row count divided by 8,192, which is 12
partitions over 100,000 rows and 122 over 1,000,000. `num_sub_vectors`
is `dim / 16`, which is 64 here, and that stores each 4,096-byte vector
as 64 bytes. Pass a JSON object as a second argument to override either.

**This step is required.** Without an index the server falls back to a
full scan, which is the exact answer, and the harness would report
recall 1.0 while measuring nothing. The script refuses to build over an
empty namespace for the same reason.

The script polls the operation and returns when the build finishes. That
takes about a minute over 100,000 rows and several minutes over a
million.

### 5. Compute the exact answers

```bash
python bench/recall/ground_truth.py \
  ./bench-data/en0010.parquet 200 10 ./truth-1m.npz \
  ./bench-data/en000{0,1,2,3,4,5,6,7,8,9}.parquet
```

Pass the corpus shards in the same order as step 3, or the ids will not
line up with the ones the server holds.

The query vectors come from shard 10, which was not loaded. A query
vector that is also a row in the table finds itself at distance zero.
That inflates recall and measures nothing.

The script enforces that rather than trusting the command line. It
verifies the pinned checksum of the query shard and of every corpus
shard, and it refuses to run when the query shard is also one of the
corpus shards.

This scans a million rows per query in NumPy. Expect a few minutes.

### 6. Score the index

```bash
python bench/recall/recall_sweep.py wiki1m ./truth-1m.npz 10 10,20,50,100 ./sweep.json
```

The server default is `nprobes` 20 when the field is omitted, so the 20
row is the shipped behaviour.

### Keeping the result cache out of the latency figures

Firn caches query results. A repeat of an earlier query is answered from
that cache, and its latency is a cache lookup rather than a search. The
recall figure survives this, because a cached result is the same result.
The latency figures do not.

The harness handles it in two parts.

Each setting runs a warm-up pass first, using perturbed copies of the
query vectors. That pulls the index and data files onto local disk
without leaving cache entries under the keys the measured pass will use.
The measured pass then sends the ground-truth vectors unchanged, so
recall compares the index against the exact answer for the same vector.

Then it checks rather than assumes. Each setting reads
`firnflow_cache_hits_total` for the namespace from `/metrics` before and
after its measured pass, and records the difference:

```json
{ "nprobes": 20, "recall@10": 0.6945, "queries": 200,
  "p50_ms": 2.94, "p95_ms": 3.05, "p99_ms": 3.23, "cache_hits": 0 }
```

**`cache_hits` must be 0.** Anything else means that many of the
measured queries were answered from cache. The script then writes
nothing to the output path, saves the readings to `OUTPUT.rejected` so
they can be inspected, names the affected settings, and exits non-zero.
The output path only ever holds a clean run, so a poisoned result cannot
be committed by mistake.

The usual cause is running the sweep twice with the same query vectors
and no write in between. The second run finds the first run's answers
waiting. Building an index counts as a write: Firn derives its cache
generation from the Lance table version, and committing a new index
moves that version, so entries from before a rebuild are unreachable.
That is measured, not assumed. Three sweeps across three consecutive
builds each reported `cache_hits` 0, and an immediate fourth sweep with
no rebuild reported 200 out of 200 with p50 falling from 3.33 ms to
0.76 ms.

To clear the cache, stop the server, delete the directory at
`FIRNFLOW_CACHE_NVME_PATH`, and start it again:

```bash
rm -rf ./bench-cache/results && mkdir -p ./bench-cache/results
```

Leave `FIRNFLOW_OBJECT_CACHE_DIR` alone. That cache holds the index
files themselves, and clearing it only makes the first queries slower
without affecting correctness.

### Rejecting a response that cannot be scored

Recall divides the overlap with the exact answer by `k`. A response that
carries fewer than `k` ids, or that repeats one, cannot reach 1.0
however good the index is. Averaged in, it looks like an index that
missed rather than a server that answered wrongly.

So the sweep requires exactly `k` results with `k` distinct ids from
every query. The first query that fails names itself and the fault,
stops the run, and leaves the remaining settings unmeasured.

Both failures write the same file. `OUTPUT.rejected` holds
`{"settings": [...], "bad_responses": [...]}`: the settings that
finished before the run stopped, and the per-query rejections, which are
empty when the cache check is what failed. The output path itself stays
untouched either way, so only a clean run can be committed.

## Checking whether more probing helps

The first thing anyone asks on seeing a low recall figure is whether the
index is simply searching too little of itself. `nprobes` is the setting
that controls that, so push it past the point where it can matter.

Sweep it across three orders of magnitude in one run, starting at 1:

```bash
python bench/recall/recall_sweep.py wiki1m ./truth-1m.npz 10 \
  1,2,5,10,20,50,100,316,1000 ./exhaustive.json
```

Starting at 1 matters. It is the control. If recall at 1 partition is
no different from recall at 1000, the setting is not reaching the index
at all and nothing else in the run can be trusted.

The top of the sweep is capped by the index rather than by the setting.
`nprobes` cannot search more partitions than the index holds, and with
no tuning options that is 12 over 100,000 rows and 122 over 1,000,000.
Every setting at or above the partition count searches the whole index
and returns the same answer as every other.

Read the result this way:

- **Recall climbs and reaches 1.0.** The index was searching too
  narrowly. Raising `nprobes` is the fix.
- **Recall climbs at first, then stops below 1.0.** The low end proves
  the setting works. The flat part starts where `nprobes` reaches the
  partition count, and from there every row in the table is a candidate.
  Recall short of 1.0 with every row examined puts the loss in how
  candidates are scored rather than how many are examined. Product
  quantization stores each 4,096-byte vector as 64 bytes, and distances
  computed from those 64 bytes are too coarse to order the candidates
  correctly.
- **Recall is flat from `nprobes` 1 upward.** The setting is not
  reaching the index. Neither this nor any other reading holds.
- **Recall is exactly 1.0 everywhere.** The namespace has no index and
  the server is doing a full scan. Go back to step 4.

The committed run of this check is the second table in
[`../results/single_vector_recall.md`](../results/single_vector_recall.md).
It used a 100,000-row namespace rather than the million-row one, so
substitute that namespace and its ground-truth file to reproduce it
exactly. Recall climbs from 0.63 at 1 partition to 0.6945 at 10, then
does not move again through 1000.

A second run of the same sweep, on a later build of the same corpus size,
is in
[`../results/single_vector_recall_raw/nprobes_exhaustive_100k_build3.json`](../results/single_vector_recall_raw/nprobes_exhaustive_100k_build3.json).
It climbs from 0.4625 to 0.587 and then stops, the same shape at a lower
level. Between `nprobes` 10 and 1000, a hundred-fold increase in how
much of the index is searched, neither recall nor latency moves.

To see the same thing on a single query, without any averaging:

```bash
python bench/recall/probe_ids.py wiki1m ./truth-1m.npz 0 10 \
  1,2,5,10,20,50,100,316,1000
```

That prints the ids returned at each setting on their own line, with the
exact answer underneath. When the lists stop changing while rows from
the exact answer are still missing from all of them, the missing rows
were never going to be found by searching wider.

## Repeating the index build

An IVF index groups the vectors into partitions with k-means, and
k-means starts from centroids chosen at random. Two builds over
identical rows produce different partitions, so recall moves between
builds. One build cannot tell a property of the engine from one lucky or
unlucky set of starting centroids.

`rebuild_index.py` replaces the index in place, so the corpus does not
need loading again:

```bash
for build in 1 2 3; do
  python bench/recall/rebuild_index.py wiki100k
  python bench/recall/recall_sweep.py wiki100k ./truth.npz 10 20 ./sweep-b$build.json
done
```

Three builds of one unchanged 100,000-row namespace, scored at the
default `nprobes` of 20 over the same 200 queries, gave recall@10 0.592,
0.595 and 0.587. Each build took about 60 seconds. Read the second
decimal of any single recall figure as noise of roughly that size, and
quote a spread rather than a point.

## Reading the output

`recall_sweep.py` writes a JSON list, one entry per setting:

```json
{ "nprobes": 20, "recall@10": 0.587, "queries": 200,
  "p50_ms": 3.30, "p95_ms": 3.57, "p99_ms": 3.74, "cache_hits": 0 }
```

`queries` is the sample size the recall figure was averaged over. Read
it before the recall figure. `cache_hits` is 0 in every file the script
writes; see the section above for what happens when it is not.

Committed results and what they mean are in
[`../results/single_vector_recall.md`](../results/single_vector_recall.md).

## Limits of this setup

- Timings from MinIO on loopback are lower than they would be against a
  real object store. The recall figures do not depend on that. They are
  a property of the index and the data.
- One dataset, one index configuration. Nothing here varies
  `num_partitions`, `num_sub_vectors` or `num_bits`. Three builds of
  that one configuration were measured, which gives a spread for the
  build but says nothing about other configurations.
- The environment is not captured by any script. The committed report
  states it in prose. A run on other hardware will produce different
  latency columns.
- Single-vector namespaces only. Nothing here touches the multivector
  path.
- Recall is averaged over 200 queries. Separate runs differ by a couple
  of points at that sample size, and separate builds of the same index
  differ by about one, so treat the second decimal as noise.
- The query vectors are real embeddings from a held-out shard, which is
  a harder distribution than randomly generated vectors. Numbers from a
  harness that generates its queries are not comparable to these.
