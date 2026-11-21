# ruff: noqa: T201
"""
Growth table for lead-time balancing (forecast.lead_balance, see
src/weathergen/train/loss_modules/lead_balance.py) from the per-step training losses of a run.

For each channel and output step s, the loss is averaged over the training log entries (the
first --skip-fraction of the entries, e.g. warm-up at a high learning rate, are left out) and
divided by the average at the first step. The table is written as a config overlay.

Usage:
    python scripts/compute_lead_balance.py \\
        /e/scratch/weatherai/shared_work/results/xxlm80yw/xxlm80yw_train_metrics.json \\
        config/config_lead_balance_xxlm80yw.yml [--stream ERA5] [--loss mse] [--power 1.0]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def per_step_losses(metrics_file: Path, stream: str, loss: str, skip_fraction: float):
    """Mean training loss per channel and output step: {channel: {step: (mean, count)}}."""
    rows = []
    with open(metrics_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("stage") == "train":
                rows.append(row)
    assert len(rows) > 0, f"No training entries in {metrics_file}."
    rows = rows[int(len(rows) * skip_fraction) :]

    prefix = f"LossPhysical.{stream}.{loss}."
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, v in row.items():
            if not key.startswith(prefix):
                continue
            ch, _, step = key[len(prefix) :].rpartition(".")
            if not step.isdigit() or v is None or not math.isfinite(v):
                continue
            values[ch][int(step)].append(v)
    assert len(values) > 0, f"No per-step losses {prefix}<channel>.<step> in {metrics_file}."
    means = {
        ch: {s: (sum(v) / len(v), len(v)) for s, v in sorted(steps.items())}
        for ch, steps in values.items()
    }
    return means, len(rows)


def growth_table(means: dict, min_count: int):
    """Loss relative to the first step, per channel; all channels need the same steps."""
    steps = sorted(next(iter(means.values())))
    assert steps == list(range(steps[0], steps[-1] + 1)), f"Steps are not contiguous: {steps}."
    table = {}
    for ch, by_step in means.items():
        assert sorted(by_step) == steps, f"Channel {ch} has steps {sorted(by_step)}, not {steps}."
        counts = [by_step[s][1] for s in steps]
        assert min(counts) >= min_count, f"Channel {ch}: only {min(counts)} entries at some step."
        first = by_step[steps[0]][0]
        assert first > 0.0, f"Channel {ch}: loss at the first step is {first}."
        table[ch] = [by_step[s][0] / first for s in steps]
    return steps[0], table


def write_overlay(out, stream, first_step, table, power, source):
    lines = [
        "# Lead-time balancing of the training loss (forecast.lead_balance), see",
        "# src/weathergen/train/loss_modules/lead_balance.py.",
        f"# Growth: {source}.",
        "# Written by scripts/compute_lead_balance.py.",
        "training_config:",
        "  forecast:",
        "    lead_balance:",
        f"      power: {power}",
        f"      first_step: {first_step}",
        "      growth:",
        f"        {stream}:",
    ]
    for ch, values in table.items():
        lines.append(f"          {ch}: [{', '.join(f'{v:.4g}' for v in values)}]")
    Path(out).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("metrics", type=Path, help="<run>_train_metrics.json of a rollout run")
    ap.add_argument("out", type=Path, help="config overlay to write")
    ap.add_argument("--stream", default="ERA5")
    ap.add_argument("--loss", default="mse", help="loss function name in the logged keys")
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--skip-fraction", type=float, default=1 / 3)
    ap.add_argument("--min-count", type=int, default=10)
    a = ap.parse_args()

    means, num_rows = per_step_losses(a.metrics, a.stream, a.loss, a.skip_fraction)
    first_step, table = growth_table(means, a.min_count)
    source = (
        f"{a.loss} of stream {a.stream} per output step relative to step {first_step}, mean over"
        f" the last {num_rows} training log entries of {a.metrics.name}"
    )
    write_overlay(a.out, a.stream, first_step, table, a.power, source)

    # summary: largest growth and steps where the loss decreases (should be rare and small)
    last = max(len(v) for v in table.values())
    top = sorted(table, key=lambda ch: -table[ch][-1])[:5]
    print(f"{len(table)} channels, steps {first_step}-{first_step + last - 1}, {num_rows} entries")
    print("largest growth at the last step:", {ch: round(table[ch][-1], 1) for ch in top})
    for ch, v in table.items():
        drops = [(first_step + i + 1, round(v[i + 1] / v[i], 3)) for i in range(len(v) - 1)]
        drops = [d for d in drops if d[1] < 0.98]
        if drops:
            print(f"{ch}: loss decreases by more than 2% at steps {drops}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
