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
| `download_shards.py` | Fetches Cohere Wikipedia parquet shards from Hugging Face. |
| `seed_namespace.py` | Loads shards into a namespace through `POST /ns/{ns}/import`. |
| `ground_truth.py` | Computes the exact top-k of held-out queries over every loaded row. |
| `recall_sweep.py` | Queries the server and scores recall@k at several `nprobes` settings. |
| `probe_ids.py` | Prints the ids one query returns at each setting, next to the exact answer. |
| `corpus.py` | Shared helpers. Not run directly. |

Every script reads two environment variables:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `FIRNFLOW_URL` | `http://127.0.0.1:3000` | Base URL of the server under test. |
| `FIRNFLOW_API_KEY` | unset | Bearer token. Only needed if the server was started with a key. |

Everything else is a command-line argument. No paths are baked in.

## The dataset

`Cohere/wikipedia-2023-11-embed-multilingual-v3`, English split. Each
shard is 100,000 rows, 1024 dimensions per row, about 216 MB on disk.
The embeddings ship pre-computed, so no embedding model and no API key
are involved, and two people running this get the same vectors.

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

### 3. Load the corpus

```bash
python bench/recall/seed_namespace.py wiki1m ./bench-data/en000{0,1,2,3,4,5,6,7,8,9}.parquet
```

The namespace is created by the first import and its dimension is fixed
at 1024. Ids run from 0 upward in the order the shards are given.

### 4. Build the index

```bash
curl -X POST http://127.0.0.1:3000/ns/wiki1m/index \
  -H 'Content-Type: application/json' -d '{"kind":"ivf_pq"}'
```

Index creation is an admin route. If the server was started with
`FIRNFLOW_ADMIN_API_KEY` or `FIRNFLOW_API_KEY`, add
`-H "Authorization: Bearer $KEY"`.

No tuning options are passed, so the server picks its own partition
count and its own `num_sub_vectors` of `dim / 16`, which is 64 here.
That stores each 4,096-byte vector as 64 bytes.

**This step is required.** Without an index the server falls back to a
full scan, which is the exact answer, and the harness will report
recall 1.0 while measuring nothing.

Poll `GET /operations/{id}` with the returned id until it succeeds.
The build takes several minutes at a million rows.

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
measured queries were answered from cache, and the script prints a
warning naming the affected settings.

The usual cause is running the sweep twice with the same query vectors.
The second run finds the first run's answers waiting. To clear it, stop
the server, delete the directory at `FIRNFLOW_CACHE_NVME_PATH`, and
start it again:

```bash
rm -rf ./bench-cache/results && mkdir -p ./bench-cache/results
```

Leave `FIRNFLOW_OBJECT_CACHE_DIR` alone. That cache holds the index
files themselves, and clearing it only makes the first queries slower
without affecting correctness.

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

Read the result this way:

- **Recall climbs and reaches 1.0.** The index was searching too
  narrowly. Raising `nprobes` is the fix.
- **Recall climbs at first, then stops below 1.0.** The low end proves
  the setting works. The flat part means the search has run out of
  things to find and the answer is still wrong, so the loss is in how
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

To see the same thing on a single query, without any averaging:

```bash
python bench/recall/probe_ids.py wiki1m ./truth-1m.npz 0 10 \
  1,2,5,10,20,50,100,316,1000
```

That prints the ids returned at each setting on their own line, with the
exact answer underneath. When the lists stop changing while rows from
the exact answer are still missing from all of them, the missing rows
were never going to be found by searching wider.

## Reading the output

`recall_sweep.py` writes a JSON list, one entry per setting:

```json
{ "nprobes": 20, "recall@10": 0.6945, "queries": 200,
  "p50_ms": 0.69, "p95_ms": 0.72, "p99_ms": 0.74 }
```

`queries` is the sample size the recall figure was averaged over. Read
it before the recall figure.

Committed results and what they mean are in
[`../results/single_vector_recall.md`](../results/single_vector_recall.md).

## Limits of this setup

- Timings from MinIO on loopback are lower than they would be against a
  real object store. The recall figures do not depend on that. They are
  a property of the index and the data.
- One dataset, one index configuration. Nothing here varies
  `num_partitions`, `num_sub_vectors` or `num_bits`.
- Single-vector namespaces only. Nothing here touches the multivector
  path.
- Recall is averaged over 200 queries. Separate runs differ by a couple
  of points at that sample size, so treat the second decimal as noise.
- The query vectors are real embeddings from a held-out shard, which is
  a harder distribution than randomly generated vectors. Numbers from a
  harness that generates its queries are not comparable to these.
