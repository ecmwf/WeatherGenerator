#!/usr/bin/env -S uv run

# ruff: noqa: T201
"""
Summarize the memory samples written during a run with ``launch-slurm.py --io-timing-memory``
(see src/weathergen/utils/memory_sampler.py): the peak memory of every process (ranks and data
loader workers) and the node-wide total over time. Totals sum PSS (shared pages split between
the processes sharing them), so fork-inherited and shared-memory pages are not counted once per
worker. With ``--window`` only the samples inside an nsys capture window are used.

USAGE EXAMPLES (from the root of the repo):
  uv run scripts/analyze_io_memory.py logs/<run-id>/profiling/io_timing/
  uv run scripts/analyze_io_memory.py logs/<run-id>/profiling/io_timing/ --window startup
  uv run scripts/analyze_io_memory.py logs/<run-id>/profiling/io_timing/ --memory-csv memory.csv
  uv run scripts/analyze_io_memory.py logs/<run-id>/profiling/io_timing/ --plot memory.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

# keep in sync with weathergen.utils.nsys_windows.IO_EVENT_STREAM (not imported: standalone)
EVENT_STREAM = "nsys-window"
# keep in sync with weathergen.utils.memory_sampler.STREAM
MEMORY_STREAM = "memory"
MEMORY_COLS = ["rss_mib", "pss_mib", "pss_anon_mib", "pss_file_mib", "pss_shmem_mib"]
GPU_COLS = ["gpu_alloc_mib", "gpu_reserved_mib", "gpu_max_alloc_mib"]


def load(sources: list[Path]) -> pd.DataFrame:
    """Load timing records from directories (all io_timing_*.jsonl inside) and/or files."""
    files: list[Path] = []
    for src in sources:
        files += sorted(src.glob("io_timing_*.jsonl")) if src.is_dir() else [src]
    if not files:
        sys.exit(f"No io_timing_*.jsonl files found in {[str(s) for s in sources]}.")

    records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # a process killed mid-write can leave a truncated last line
                    continue
    df = pd.DataFrame.from_records(records)
    # first-call time relative to the run start, used to order the tree
    df["t_rel"] = df["t_start"] - df["t_start"].min()
    return df


def select_window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep the calls that started inside the nsys capture window ``name``.

    The window bounds are the ``<name> open`` / ``<name> close`` events written by
    weathergen.utils.nsys_windows (earliest open and latest close over all processes).
    """
    events = df[df["stream"] == EVENT_STREAM]
    t_open = events.loc[events["op"] == f"{name} open", "t_start"]
    t_close = events.loc[events["op"] == f"{name} close", "t_start"]
    if t_open.empty:
        known = sorted({op.rsplit(" ", 1)[0] for op in events["op"]})
        sys.exit(f"No window '{name}' in the records (found: {known}).")
    # a window that was never closed (run ended early) extends to the last record
    t_end = t_close.max() if not t_close.empty else (df["t_start"] + df["dt"]).max()
    return df[(df["t_start"] >= t_open.min()) & (df["t_start"] <= t_end)]


def _fmt(value: float, spec: str) -> str:
    return "" if pd.isna(value) else format(value, spec)


def print_table(
    header: list[tuple[str, int]], rows: list[list[str]], footer: list[str] | None = None
) -> None:
    width = max([len(r[0]) for r in rows + [footer or [""]]] + [len(header[0][0])])

    def fmt_row(r: list[str]) -> str:
        cells = zip(r[1:], header[1:], strict=False)
        return r[0].ljust(width) + "".join(c.rjust(w) for c, (_, w) in cells)

    line = header[0][0].ljust(width) + "".join(h.rjust(w) for h, w in header[1:])
    print(line)
    print("-" * len(line))
    for r in rows:
        print(fmt_row(r))
    if footer is not None:
        print("-" * len(line))
        print(fmt_row(footer))


def prepare_memory(mem: pd.DataFrame) -> pd.DataFrame:
    """Add ``mem_mib`` (PSS, RSS where PSS is unavailable) and ``group`` (rank main pid)."""
    mem = mem.copy()
    for col in MEMORY_COLS + GPU_COLS:
        if col not in mem:
            mem[col] = float("nan")
    mem["mem_mib"] = mem["pss_mib"].fillna(mem["rss_mib"])
    # a worker belongs to the rank whose main process is its parent
    mem["group"] = mem["pid"].where(mem["role"] == "main", mem["ppid"])
    return mem


def process_peaks(mem: pd.DataFrame) -> pd.DataFrame:
    """Peak memory (MiB) per process, and when the peak PSS happened (s since run start)."""
    g = mem.groupby(["host", "group", "role", "pid"])
    peaks = g[MEMORY_COLS + ["mem_mib"]].max()
    peaks["t_peak_s"] = mem.loc[g["mem_mib"].idxmax(), "t_rel"].to_numpy()
    return peaks.reset_index().sort_values(["host", "group", "role"])


def memory_timeline(mem: pd.DataFrame, bin_s: float = 1.0) -> pd.DataFrame:
    """Node-wide memory per time bin: PSS summed over processes, split main vs. workers.

    A sample is carried forward for 1.5 sampling intervals (the median spacing of the samples),
    so a worker that exited stops counting about one interval after its last sample, while a
    process whose next sample is delayed or lies outside a ``--window`` is still counted.
    GPU columns are the maximum over the main processes.
    """
    mem = mem.assign(bin=(mem["t_rel"] // bin_s).astype(int))
    per_proc = mem.groupby(["bin", "pid"]).agg(mib=("mem_mib", "last"), role=("role", "last"))
    wide = per_proc["mib"].unstack("pid")
    spacing = mem.sort_values("t_rel").groupby("pid")["t_rel"].diff().dropna()
    spacing = spacing[spacing > 0]
    interval = float(spacing.median()) if not spacing.empty else bin_s
    carry = max(1, math.ceil(1.5 * interval / bin_s))
    wide = wide.reindex(range(int(wide.index.min()), int(wide.index.max()) + 1)).ffill(limit=carry)
    is_main = per_proc["role"].unstack("pid").ffill().iloc[0].eq("main")
    main_cols = [c for c in wide.columns if is_main.get(c, False)]
    out = pd.DataFrame(
        {
            "t_s": wide.index * bin_s,
            "total_mib": wide.sum(axis=1).to_numpy(),
            "main_mib": wide[main_cols].sum(axis=1).to_numpy(),
        }
    )
    out["workers_mib"] = out["total_mib"] - out["main_mib"]
    gpu = mem.groupby("bin")[GPU_COLS].max().reindex(wide.index)
    for col in GPU_COLS:
        out[col] = gpu[col].to_numpy()
    return out


def report_memory(mem: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """Print the memory report and return the node-wide timeline."""
    mem = prepare_memory(mem)
    print("\nMemory (MiB), peak per process")
    peaks = process_peaks(mem)
    header = [("host / rank group / role", 0)] + [
        (h, 10) for h in ["rss", "pss", "anon", "file", "shmem", "t_peak_s"]
    ]
    rows = [
        [f"{r.host} / {int(r.group)} / {r.role}"]
        + [_fmt(getattr(r, c), ".0f") for c in MEMORY_COLS]
        + [_fmt(r.t_peak_s, ".1f")]
        for r in peaks.itertuples()
    ]
    print_table(header, rows)
    if mem["pss_mib"].isna().any():
        print("(some samples have no PSS; the totals below use RSS for them and overcount shared)")

    timeline = memory_timeline(mem, bin_s)
    peak = timeline.loc[timeline["total_mib"].idxmax()]
    print(
        f"\nNode-wide total over all processes (PSS): peak {peak.total_mib:.0f} MiB at "
        f"t={peak.t_s:.0f}s (main processes {peak.main_mib:.0f}, workers {peak.workers_mib:.0f})"
    )
    if timeline["gpu_max_alloc_mib"].notna().any():
        print(
            f"GPU (PyTorch allocator, max over main processes): allocated peak "
            f"{timeline['gpu_alloc_mib'].max():.0f} MiB, reserved peak "
            f"{timeline['gpu_reserved_mib'].max():.0f} MiB, max-allocated "
            f"{timeline['gpu_max_alloc_mib'].max():.0f} MiB"
        )
    return timeline


# Chart colors: categorical slots 1-4 of the reference palette, validated for adjacent-pair CVD
# and normal-vision separation (aqua and yellow are low-contrast on the surface, hence the
# direct labels on the GPU lines). Ink and grid are neutral.
SURFACE, INK, INK_2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#eceae4"
C_MAIN, C_WORKERS, C_GPU_ALLOC, C_GPU_RESERVED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GIB = 1024.0


def plot_memory(
    timeline: pd.DataFrame,
    spans: dict[str, tuple[float, float]],
    path: Path,
    title: str = "Memory footprint",
) -> None:
    """Write the memory chart: host PSS stacked (main vs. workers) and, if sampled, GPU lines.

    ``spans`` are the nsys capture windows in seconds since the run start (shaded bands).
    """
    import matplotlib  # noqa: PLC0415 (only needed for --plot)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    has_gpu = timeline["gpu_reserved_mib"].notna().any()
    fig, axes = plt.subplots(
        2 if has_gpu else 1,
        1,
        figsize=(10, 6.6 if has_gpu else 4.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]} if has_gpu else None,
        facecolor=SURFACE,
        squeeze=False,
    )
    axes = axes[:, 0]
    t = timeline["t_s"].to_numpy()

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)
        for t_open, t_close in spans.values():
            ax.axvspan(t_open, t_close, color=BAND, zorder=0, linewidth=0)
    for name, (t_open, _) in spans.items():
        axes[0].text(
            t_open,
            1.02,
            name,
            transform=axes[0].get_xaxis_transform(),
            color=INK_2,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax = axes[0]
    ax.stackplot(
        t,
        timeline["main_mib"] / GIB,
        timeline["workers_mib"] / GIB,
        colors=[C_MAIN, C_WORKERS],
        edgecolor=SURFACE,
        linewidth=1.0,
        labels=["rank processes (main)", "data loader workers"],
        zorder=2,
    )
    ax.set_ylim(0, timeline["total_mib"].max() / GIB * 1.2)
    ax.set_ylabel("Host memory, PSS (GiB)", color=INK, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    peak = timeline.loc[timeline["total_mib"].idxmax()]
    ax.plot(
        [peak.t_s],
        [peak.total_mib / GIB],
        "o",
        color=INK,
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
    )
    ax.annotate(
        f"peak {peak.total_mib / GIB:.1f} GiB at {peak.t_s:.0f} s",
        (peak.t_s, peak.total_mib / GIB),
        xytext=(8, 8),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )

    if has_gpu:
        ax = axes[1]
        for col, color, label in (
            ("gpu_reserved_mib", C_GPU_RESERVED, "reserved"),
            ("gpu_alloc_mib", C_GPU_ALLOC, "allocated"),
        ):
            series = timeline[["t_s", col]].dropna()
            ax.plot(series["t_s"], series[col] / GIB, color=color, linewidth=2, zorder=3)
            ax.annotate(
                f"{label} {series[col].max() / GIB:.1f} GiB (peak)",
                (series["t_s"].iloc[-1], series[col].iloc[-1] / GIB),
                xytext=(0, 7),
                textcoords="offset points",
                color=INK_2,
                fontsize=9,
                ha="right",
                va="bottom",
            )
        ax.set_ylim(0, timeline["gpu_reserved_mib"].max() / GIB * 1.25)
        ax.set_ylabel("GPU memory, PyTorch allocator (GiB)", color=INK, fontsize=10)
        ax.margins(x=0)
    axes[-1].set_xlabel("time since first record (s)", color=INK_2, fontsize=10)
    axes[0].set_xlim(t.min(), t.max())
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def window_bounds(df: pd.DataFrame, t0: float) -> dict[str, tuple[float, float]]:
    """nsys capture windows as (open, close) in seconds since ``t0``, over all processes."""
    events = df[df["stream"] == EVENT_STREAM]
    spans: dict[str, tuple[float, float]] = {}
    for name in sorted({op.rsplit(" ", 1)[0] for op in events["op"]}):
        opens = events.loc[events["op"] == f"{name} open", "t_start"]
        closes = events.loc[events["op"] == f"{name} close", "t_start"]
        if not opens.empty:
            end = closes.max() if not closes.empty else df["t_start"].max()
            spans[name] = (opens.min() - t0, end - t0)
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", nargs="+", type=Path, help="timing dir(s) or .jsonl file(s)")
    parser.add_argument(
        "--window", help="only use samples inside this nsys capture window (e.g. startup, steady)"
    )
    parser.add_argument(
        "--memory-bin", type=float, default=1.0, help="time bin (s) of the node-wide memory total"
    )
    parser.add_argument(
        "--memory-csv", type=Path, help="also write the node-wide memory timeline to this CSV"
    )
    parser.add_argument(
        "--plot", type=Path, help="also draw the memory chart to this image (.png, .svg, .pdf)"
    )
    args = parser.parse_args()

    df = load(args.run)
    spans = window_bounds(df, df["t_start"].min())
    if args.window:
        df = select_window(df, args.window)
    mem = df[df["stream"] == MEMORY_STREAM]
    if mem.empty:
        sys.exit(
            "No memory samples" + (f" in window {args.window}" if args.window else "") + ". "
            "Launch with launch-slurm.py --io-timing-memory."
        )
    timeline = report_memory(mem, args.memory_bin)
    if args.memory_csv:
        timeline.to_csv(args.memory_csv, index=False)
        print(f"Wrote {args.memory_csv}")
    if args.plot:
        title = "Memory footprint" + (f", window {args.window}" if args.window else "")
        run_ids = [p for p in args.run[0].resolve().parts if p == "io_timing"]
        if run_ids and len(args.run[0].resolve().parts) >= 4:
            title += f" — run {args.run[0].resolve().parts[-3]}"
        shown = {args.window: spans[args.window]} if args.window else spans
        plot_memory(timeline, shown, args.plot, title)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
