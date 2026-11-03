#!/bin/bash
# Mount a squashfs image with squashfuse_ll for one command, then unmount.
#
#   with_squashfs.sh IMAGE COMMAND [ARGS...]
#
# Every @MOUNT@ in the arguments is replaced by the mount point,
# /tmp/zrb-<job>-<step>-<rank>: one mount per task, as in the venv-image design
# (WeatherGenerator-private, branch fs/squashfs-venv-design,
# doc/inode-issues/squashfs-venv/mounting.md). Prints the mount time and, at the end, the
# CPU time of the squashfuse_ll process to stderr.
#
# Slurm kills squashfuse_ll when a step ends and leaves a dead mount in /tmp, which is
# shared between jobs. So the wrapper unmounts on every way out (the command runs in the
# background, TERM/INT are forwarded) and first clears this user's dead /tmp/zrb-* mounts.

set -u
img=${1:?usage: with_squashfs.sh IMAGE COMMAND [ARGS...]}
shift
if [ $# -eq 0 ]; then
    echo "with_squashfs: no command" >&2
    exit 2
fi
if [ ! -r "$img" ]; then
    echo "with_squashfs: cannot read $img" >&2
    exit 1
fi
tag="with_squashfs[${SLURM_PROCID:-0}]"

# Dead mounts fail stat; fusermount refuses other users' mounts.
for d in $(findmnt -rn -t fuse.squashfuse_ll -o TARGET); do
    case $d in /tmp/zrb-*) ;; *) continue ;; esac
    stat "$d" >/dev/null 2>&1 && continue
    fusermount -u "$d" 2>/dev/null && rmdir "$d" && echo "$tag: cleared dead mount $d" >&2
done

m=/tmp/zrb-${SLURM_JOB_ID:-local$$}-${SLURM_STEP_ID:-0}-${SLURM_PROCID:-0}
mkdir -p "$m" || exit 1
t0=$(date +%s%N)
# -f keeps squashfuse_ll in the foreground, so its PID is known for the CPU time.
squashfuse_ll -f "$img" "$m" &
dpid=$!
for _ in $(seq 200); do
    mountpoint -q "$m" && break
    kill -0 "$dpid" 2>/dev/null || break
    sleep 0.05
done
if ! mountpoint -q "$m"; then
    echo "$tag: mounting $img at $m failed" >&2
    kill "$dpid" 2>/dev/null
    rmdir "$m"
    exit 1
fi
echo "$tag: mounted $img at $m on $(hostname) in $((($(date +%s%N) - t0) / 1000000)) ms" >&2

unmount() {
    local cpu
    cpu=$(awk -v tck="$(getconf CLK_TCK)" '{printf "%.1f", ($14 + $15) / tck}' \
        "/proc/$dpid/stat" 2>/dev/null)
    echo "$tag: squashfuse_ll cpu_seconds=${cpu:-unknown}" >&2
    fusermount -u "$m" 2>/dev/null || fusermount -u -z "$m"
    wait "$dpid" 2>/dev/null
    rmdir "$m"
}

args=()
for a in "$@"; do
    args+=("${a//@MOUNT@/$m}")
done
"${args[@]}" &
pid=$!
trap 'kill -TERM $pid 2>/dev/null' TERM INT
wait "$pid"
rc=$?
while kill -0 "$pid" 2>/dev/null; do
    wait "$pid"
    rc=$?
done
unmount
exit $rc
