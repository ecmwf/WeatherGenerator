# ZRB-E002 — squashfs variant (dropped)

- Date: 2026-11-03 to 2026-11-04
- Status: done; squashfs dropped for datasets
- Question: does packing the Zarr 2 store into a read-only squashfs image, mounted per task
  with `squashfuse_ll`, read as fast as the plain Zarr 2 store? (The image would cut the
  dataset to one inode; the idea came from the squashfs venv images.)
- Part of [ZRB-E002](ZRB-E002-n320-v2-v3-march-2022.md): same data (the `v2` copy of March
  2022), same benchmark and code (WeatherGenerator `9e257596`). The squashfs variant was
  removed from that record on 2026-11-04; its build and check-run commands are blocks 3–4 of
  the record at [`fdc7cabc`](https://github.com/florianscheidl/WeatherGenerator/blob/fdc7cabc6/scripts/zarr_investigation/experiments/ZRB-E002-n320-v2-squashfs-v3-march-2022.md).
  The diagnosis commands (D1–D7, R0–R3) ran from local notes, not kept in the repo. The images
  were deleted on 2026-11-04.
- Mount wrapper: [`with_squashfs.sh`](../with_squashfs.sh) (one mount per task, `@MOUNT@`
  substitution, daemon CPU time, unmount on every way out).

## Image

`mksquashfs` of the `v2` copy (`stage/`, 792 chunk files, 91.8 GiB) with `-noD -noF`, i.e.
no squashfs compression of data and fragments, since the Zarr chunks are already compressed.
Tools on JUPITER: `squashfs-tools-4.4-10.git1.el9`, `squashfuse-0.1.104-1.el9` (FUSE 2),
`fuse-libs-2.9.9-17.el9`.

- `-b 1M`: built in 13 s from the just-written copy (cached); `-b 128K`: 153–164 s from
  uncached files (~0.6 GB/s with 8 processors on a login node, i.e. ~30 min per TB).
- 94016 MiB, 1 inode (`v2`: 94017 MiB, 1023 inodes). Mount 18–178 ms.
- `-noX` only leaves xattrs uncompressed; it doesn't drop them, so `mksquashfs` warns
  `Unrecognised xattr prefix system.posix_acl_*` for every directory with scratch's ACLs
  (harmless; `-no-xattrs` silences it).
- A closed shell kills `mksquashfs` before it writes the superblock (last): the first
  attempt left an image of full size that didn't mount.

## Results (login node `jpbl-s01-01`, shared: indicative)

First check run (10 windows of the `s6` order, 1 task; 2026-11-03): `v2` 2458 ms per window
(580 MiB/s decoded), `sq` 7584 ms (177 MiB/s), `v3` 3124 ms; all three returned identical
data (checksums 10/10).

Squashfuse 0.6.0 below is the copy inside Apptainer's `libexec` (FUSE 3, multithreaded),
run directly for the test; JUPITER provides only 0.1.104.

Raw reads with `dd` (6 chunk files, ~710 MiB):

| Path | Cache | One after another | 6 at once |
| --- | --- | --- | --- |
| GPFS files | cold | 1.05–1.20 s | 0.21–0.22 s |
| sq 128 KiB, 0.1.104 | cold | 0.47 s | 0.28 s* |
| sq 1 MiB, 0.1.104 | cold | 0.42 s | 0.45 s* |
| sq 128 KiB, 0.6.0 | cold | 0.55 s | 0.21 s* |
| sq 1 MiB, 0.6.0 | cold | 0.48 s | 0.09 s* |
| sq 1 MiB, 0.1.104 | warm | 0.24 s | 0.74 s |
| sq 128 KiB, 0.1.104 | warm | 0.25 s | 0.25 s |

\* optimistic: in the image the "at once" files follow the "one after another" files, so
GPFS had already prefetched part of them.

Check runs (10 windows of the `s6` order, 1 task; medians):

| Runs (seed) | Zarr concurrency | `v2` | `sq` |
| --- | --- | --- | --- |
| 0.1.104, 1 MiB (0, 1) | 10 | 2458, 2646 ms | 7584, 7248 ms |
| 0.1.104, 128 KiB (1) | 10 | 2646 ms | 6645 ms |
| 0.6.0, 1 MiB (4) | 10 | 2486 ms | 3553 ms (1.43×) |
| 0.6.0, 128 KiB (4) | 10 | 2486 ms | 4215 ms (1.70×) |
| 0.1.104, 128 KiB (5) | 1 | 3867 ms | 3790 ms (p90 7254) |

Daemon CPU stayed small (0.4–8.4 s per run of ~40–70 s).

- The image itself reads fast: cold, one file after another, faster than separate GPFS
  files (one large file streams better).
- The gap comes from concurrent reads. A window's 6 chunks are read concurrently (the reader
  takes `ds[start:end]`); `v2` gains ~1.55× from that (3867 → 2486 ms), squashfuse 0.1.104
  serialises them (and 1 MiB blocks thrash under concurrency when warm), 0.6.0 recovers only
  part. With Zarr reading one chunk at a time, `sq` and `v2` are equal.
- Likely reason for the remaining gap (not measured): one large read of a GPFS file is
  spread over many storage servers by the GPFS client; through FUSE it arrives as a stream
  of small requests served by a few daemon threads.

## Conclusion

Dropped for these datasets (decided 2026-11-04 against an agreed bar of ~1.2× of `v2`):
with JUPITER's squashfuse 0.1.104 reads are 2.5–3× slower than `v2`; even squashfuse 0.6.0
(not provided on JUPITER, only inside Apptainer) is 1.4× slower. With chunks of ~119 MiB
there are no small-file costs to save, so squashfs could at best tie `v2`, and its only gain
over Zarr 3 sharding is the inode count (1 vs ~17k for N320 with one-day shards), at the
price of mount handling in every job and a squashfuse we would have to supply. The venv
images are unaffected: there the many small files are what squashfs saves.
