import argparse
import os
import sys
import time
from datetime import timedelta
import numpy as np
import zarr
from zarr.storage import LocalStore
from zarr.codecs import BloscCodec
from anemoi.datasets import open_dataset


# Base paths
INPUT_BASE = "/e/data1/slmet/ml_training/"
OUTPUT_BASE = "/e/scratch/hclimrep/kreshpa1/zarr3_datasets/"


def is_missing_date_error(e):
    return "MissingDateError" in type(e).__name__


def month_arg(value):
    """Parse "YYYY-MM" or "MM-YYYY" into the "YYYY-MM" form anemoi expects."""
    try:
        first, second = value.split("-")
        # The 4-digit part is the year
        if len(first) == 4:
            year, month = int(first), int(second)
        elif len(second) == 4:
            month, year = int(first), int(second)
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not in YYYY-MM or MM-YYYY format")
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"{value!r} has invalid month {month}")
    return f"{year:04d}-{month:02d}"


def format_duration(seconds):
    return f"{seconds:.1f} s ({timedelta(seconds=round(seconds))})"


def main():
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Convert a range of months of a Zarr v2 dataset to Zarr v3 with sharding and compression."
    )
    parser.add_argument(
        "input_filename",
        type=str,
        help="Name of the input Zarr v2 dataset file"
    )
    parser.add_argument(
        "output_filename",
        type=str,
        nargs="?",
        default=None,
        help="Name of the output Zarr v3 dataset file "
             "(default: <input name>_<start>_<end>_<chunks per shard>.zarr)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_BASE,
        help=f"Directory to write the output into (default: {OUTPUT_BASE})"
    )
    parser.add_argument(
        "--start", "-s",
        type=month_arg,
        required=True,
        metavar="YYYY-MM",
        help="First month to convert, inclusive, YYYY-MM or MM-YYYY (e.g. 2022-03)"
    )
    parser.add_argument(
        "--end", "-e",
        type=month_arg,
        required=True,
        metavar="YYYY-MM",
        help="Last month to convert, inclusive, YYYY-MM or MM-YYYY (e.g. 2023-05)"
    )
    parser.add_argument(
        "--chunks-per-shard", "-c",
        type=int,
        default=None,
        help="Number of chunks per shard along time (default: None, no sharding)"
    )
    parser.add_argument(
        "--compressor",
        type=str,
        choices=["lz4", "lz4hc", "zstd", "zlib", "snappy", "blosclz"],
        default="lz4",
        help="Compression algorithm to use (default: lz4)"
    )
    parser.add_argument(
        "--compressor-level", "-l",
        type=int,
        default=5,
        choices=range(1, 10),
        metavar="[1-9]",
        help="Compression level 1-9 (default: 5)"
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        choices=["shuffle", "bitshuffle", "noshuffle"],
        default="shuffle",
        help="Shuffle strategy (default: shuffle)"
    )

    args = parser.parse_args()

    # Default output name: input name without ".zarr", plus the date range
    # and the number of chunks per shard
    if args.output_filename is None:
        stem = os.path.basename(args.input_filename.rstrip("/")).removesuffix(".zarr")
        shard_tag = args.chunks_per_shard if args.chunks_per_shard is not None else "noshard"
        args.output_filename = f"{stem}_{args.start}_{args.end}_{shard_tag}.zarr"

    # Construct full paths
    input_path = os.path.join(INPUT_BASE, args.input_filename)
    output_path = os.path.join(args.output_dir, args.output_filename)

    # Validate input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # anemoi treats "YYYY-MM" as the first timestep of the month for start
    # and the last timestep of the month for end, so both months are included
    if args.start > args.end:
        print(f"Error: start {args.start} is after end {args.end}.")
        sys.exit(1)

    print("="*60)
    print("Zarr v2 to v3 Conversion (month range)")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Period: {args.start} -> {args.end}")
    print(f"Chunks per shard: {args.chunks_per_shard}")
    print(f"Compressor: {args.compressor} (level {args.compressor_level})")
    print(f"Shuffle: {args.shuffle}")
    print("="*60)

    # Open input dataset, restricted to the selected months
    print("\nOpening input dataset...")
    ds = open_dataset(input_path, start=args.start, end=args.end)

    print("Dataset opened.")
    print(f"Dates: {ds.dates[0]} -> {ds.dates[-1]}")
    print(f"Shape: {ds.shape}")
    print(f"Dtype: {ds.dtype}")

    time_dim = ds.shape[0]
    # Subset (date-selected) datasets don't expose .chunks, so read them
    # from the source Zarr array; the chunk layout is unchanged by subsetting
    chunks = zarr.open_group(input_path, mode="r")["data"].chunks
    print(f"Chunks: {chunks}")

    # Optional sharding
    if args.chunks_per_shard is not None:
        shards = (
            args.chunks_per_shard * chunks[0],
            chunks[1],
            chunks[2],
            chunks[3],
        )
        print(f"Shards: {shards}")
    else:
        shards = None
        print("Shards: None (no sharding)")

    # Create Zarr v3 store
    store = LocalStore(output_path)

    root = zarr.group(
        store=store,
        overwrite=True,
        zarr_format=3,
    )

    # Build compressor based on arguments
    compressor = BloscCodec(
        cname=args.compressor,
        clevel=args.compressor_level,
        shuffle=args.shuffle,
    )

    kwargs = dict(
        name="data",
        shape=ds.shape,
        dtype=ds.dtype,
        chunks=chunks,
        compressors=(compressor,),
        fill_value=np.nan,
        overwrite=True,
    )

    if shards is not None:
        kwargs["shards"] = shards

    print("\nCreating output array...")
    arr = root.create_array(**kwargs)

    # Copy data in whole shards (or chunks) along time, so each shard is written
    # once instead of being rewritten for every timestep it contains
    step = shards[0] if shards is not None else chunks[0]
    missing = []

    print(f"\nCopying data in blocks of {step} timesteps...")
    copy_start = time.perf_counter()
    for t0 in range(0, time_dim, step):
        t1 = min(t0 + step, time_dim)
        print(f"Writing timesteps {t0}-{t1 - 1}/{time_dim - 1}")
        try:
            arr[t0:t1] = ds[t0:t1]

        except Exception as e:
            if not is_missing_date_error(e):
                raise
            # Block contains missing dates: fall back to one timestep at a time
            for t in range(t0, t1):
                try:
                    arr[t] = ds[t]
                except Exception as e:
                    if not is_missing_date_error(e):
                        raise
                    print(f"Skipping timestep {t} ({ds.dates[t]}): {e}")
                    missing.append(t)
    copy_time = time.perf_counter() - copy_start

    # Record what was converted; missing timesteps are left as NaN
    arr.attrs["start_date"] = str(ds.dates[0])
    arr.attrs["end_date"] = str(ds.dates[-1])
    arr.attrs["frequency"] = str(ds.frequency)
    arr.attrs["missing_indices"] = missing

    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Timesteps written: {time_dim - len(missing)}/{time_dim}")
    print(f"Copy time:  {format_duration(copy_time)}")
    print(f"Per timestep: {copy_time / time_dim:.3f} s")
    print(f"Total time: {format_duration(time.perf_counter() - total_start)}")
    print("="*60)


if __name__ == "__main__":
    main()
