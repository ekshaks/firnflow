"""Download Cohere Wikipedia embedding shards from Hugging Face.

Each shard is one parquet file of 100,000 rows with a 1024-dimension
embedding per row, about 216 MB on disk. Eleven shards give the
1,000,000-row corpus the report uses, plus one held-out shard for the
query vectors.

Usage:
    python download_shards.py OUTPUT_DIR FIRST LAST [WORKERS]

Example, the eleven shards the report used (0 to 9 loaded, 10 held out):
    python download_shards.py ./data 0 10

The download needs no Hugging Face account. A shard already present at
its full size is skipped, so the command is safe to re-run after an
interrupted download.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

URL_TEMPLATE = (
    "https://huggingface.co/datasets/Cohere/"
    "wikipedia-2023-11-embed-multilingual-v3/resolve/main/en/{index:04d}.parquet"
)

#: A complete shard is around 216 MB. Anything much smaller is a partial
#: download from an interrupted run and gets fetched again.
MIN_COMPLETE_BYTES = 200_000_000


def shard_path(directory, index):
    """Local path for shard `index` inside `directory`."""
    return os.path.join(directory, f"en{index:04d}.parquet")


def fetch_shard(directory, index):
    """Download one shard unless a complete copy is already on disk.

    Args:
        directory: destination directory, created by the caller.
        index: shard number, matching the file name in the dataset.

    Returns:
        A one-line status string for printing.
    """
    path = shard_path(directory, index)
    if os.path.exists(path) and os.path.getsize(path) >= MIN_COMPLETE_BYTES:
        return f"skip  {path} (already complete)"
    partial = path + ".partial"
    with requests.get(URL_TEMPLATE.format(index=index), stream=True, timeout=1800) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    os.replace(partial, path)
    return f"fetch {path} ({os.path.getsize(path)} bytes)"


def main():
    """Download the requested shard range in parallel."""
    directory = sys.argv[1]
    first, last = int(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(directory, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(lambda i: fetch_shard(directory, i), range(first, last + 1)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
