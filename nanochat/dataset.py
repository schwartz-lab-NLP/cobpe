"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

Training and validation use separate local shards; the downloader always includes
the designated validation shard.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
from functools import partial

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def resolve_parquet_data_dir(data_dir=None):
    """Resolve the parquet shard directory from an explicit arg, env, or repo default."""
    if data_dir:
        return os.path.abspath(os.path.expanduser(data_dir))
    env_data_dir = os.environ.get("LOCAL_PARQUET_DIR", "").strip()
    if env_data_dir:
        return os.path.abspath(os.path.expanduser(env_data_dir))
    return DATA_DIR


def list_parquet_files(data_dir=None):
    """Looks into a parquet data dir and returns full paths to all parquet files."""
    data_dir = resolve_parquet_data_dir(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Parquet data dir does not exist: {data_dir}. "
            "Pass --local-parquet-dir where available or set LOCAL_PARQUET_DIR."
        )

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}. Download or provide train and validation shards.")
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, data_dir=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    if split == "train" and len(parquet_paths) < 2:
        raise ValueError("Training requires at least one training shard and a separate validation shard.")
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index, data_dir=None):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(data_dir or resolve_parquet_data_dir(), filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            token = os.environ.get("HF_TOKEN", "").strip()
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, required=True, help="Number of training shards to download; -1 downloads all training shards")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("--data-dir", default=None, help="Destination (default: LOCAL_PARQUET_DIR or NANOCHAT_BASE_DIR/base_data_climbmix)")
    args = parser.parse_args()
    if args.num_files < -1 or args.num_workers < 1:
        parser.error("num-files must be -1 or nonnegative; num-workers must be positive")
    output_dir = resolve_parquet_data_dir(args.data_dir)

    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # always download the validation shard

    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {output_dir}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(partial(download_single_file, data_dir=output_dir), ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {output_dir}")

    if successful != len(ids_to_download):
        raise SystemExit(1)
