"""Download Cohere Wikipedia embedding shards from Hugging Face.

Each shard is one parquet file of 100,000 rows with a 1024-dimension
embedding per row, about 216 MB on disk. Eleven shards give the
1,000,000-row corpus the report uses, plus one held-out shard for the
query vectors.

Usage:
    python download_shards.py OUTPUT_DIR FIRST LAST [WORKERS]

Example, the eleven shards the report used (0 to 9 loaded, 10 held out):
    python download_shards.py ./data 0 10

The download needs no Hugging Face account.

Every shard is fetched at the commit pinned as DATASET_REVISION in
corpus.py, not from the `main` branch. A branch pointer moves, and this
dataset has been modified since the run recorded under bench/results/.
After the download each file is checked against the sha256 and the byte
count recorded for that commit, and a mismatch stops the script.

A shard already on disk that passes both checks is skipped, so the
command is safe to re-run after an interrupted download.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

from corpus import DATASET_REPO, DATASET_REVISION, SHARD_MANIFEST, file_digest, verify_shard

URL_TEMPLATE = (
    "https://huggingface.co/datasets/{repo}/resolve/{revision}/en/{index:04d}.parquet"
)


def shard_path(directory, index):
    """Local path for shard `index` inside `directory`."""
    return os.path.join(directory, f"en{index:04d}.parquet")


def already_correct(path, index):
    """Whether a file on disk already matches the pinned manifest.

    A partial download from an interrupted run has the wrong length, so
    the cheap size check rejects it without hashing 216 MB.

    Args:
        path: local path to check.
        index: shard number, which selects the expected size and digest.

    Returns:
        True if the file exists and matches both size and sha256.
    """
    expected_digest, expected_size = SHARD_MANIFEST[index]
    if not os.path.exists(path) or os.path.getsize(path) != expected_size:
        return False
    return file_digest(path) == expected_digest


def fetch_shard(directory, index):
    """Download one shard unless a verified copy is already on disk.

    The download goes to a `.partial` name and is renamed only after the
    checksum passes, so an interrupted or corrupted fetch never leaves a
    file that a later run would accept.

    Args:
        directory: destination directory, created by the caller.
        index: shard number, matching the file name in the dataset.

    Returns:
        A one-line status string for printing.

    Raises:
        SystemExit: if the downloaded bytes do not match the manifest.
    """
    path = shard_path(directory, index)
    if already_correct(path, index):
        return f"skip  {path} (verified against {DATASET_REVISION[:12]})"
    url = URL_TEMPLATE.format(repo=DATASET_REPO, revision=DATASET_REVISION, index=index)
    partial = path + ".partial"
    with requests.get(url, stream=True, timeout=1800) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    try:
        verify_shard(partial, index)
    except SystemExit:
        os.remove(partial)
        raise
    os.replace(partial, path)
    return f"fetch {path} ({os.path.getsize(path)} bytes, sha256 verified)"


def main():
    """Download the requested shard range in parallel.

    Raises:
        SystemExit: if any requested shard has no manifest entry. The
            check happens before the first byte is fetched, so the run
            fails in a second rather than after an hour of downloading.
    """
    directory = sys.argv[1]
    first, last = int(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    missing = [i for i in range(first, last + 1) if i not in SHARD_MANIFEST]
    if missing:
        raise SystemExit(
            f"shards {missing} have no checksum in SHARD_MANIFEST in "
            f"corpus.py. Add their sha256 and size at revision "
            f"{DATASET_REVISION} first, or the run cannot record which "
            f"bytes it measured. The comment above SHARD_MANIFEST has the "
            f"command that reads them from the Hugging Face API."
        )
    print(f"{DATASET_REPO} at revision {DATASET_REVISION}", flush=True)
    os.makedirs(directory, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(lambda i: fetch_shard(directory, i), range(first, last + 1)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
