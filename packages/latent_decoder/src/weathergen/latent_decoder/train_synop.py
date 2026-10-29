"""Train a simple decoder head on stored latents and SYNOP reports, then test it.

usage: python -m weathergen.latent_decoder.train_synop LATENTS OBS OUT
           --train FIRST LAST --test FIRST LAST [options]

Writes OUT/head.pt, OUT/predictions.npz (one row per test report: base, step, time, lat, lon,
prediction and observation per channel) and OUT/scores.json (RMSE per step and channel).
"""

import argparse
import json
import logging
import math
import pathlib
import time

import numpy as np
import torch

from weathergen.latent_decoder.dataset import StateDataset, pairs
from weathergen.latent_decoder.head import Head
from weathergen.latent_decoder.latents import STEP, open_latents
from weathergen.latent_decoder.obs import ObsStore

_logger = logging.getLogger(__name__)


def channel_stats(obs, train_pairs, n=48, seed=0):
    """Per-channel mean and std of the reports in n random training windows."""
    rng = np.random.default_rng(seed)
    values = []
    for i in rng.choice(len(train_pairs), min(n, len(train_pairs)), replace=False):
        base, k = train_pairs[i]
        values.append(obs.window(base + STEP * k, base + STEP * (k + 1))["values"])
    values = np.concatenate(values)
    return np.nanmean(values, axis=0).astype(np.float32), np.nanstd(values, axis=0).astype(
        np.float32
    )


def loader(ds, shuffle, workers):
    return torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=shuffle, num_workers=workers, persistent_workers=workers > 0
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("latents")
    ap.add_argument("obs")
    ap.add_argument("out")
    ap.add_argument("--train", nargs=2, required=True, metavar=("FIRST", "LAST"), help="init range")
    ap.add_argument("--test", nargs=2, required=True, metavar=("FIRST", "LAST"), help="init range")
    ap.add_argument("--iters", type=int, default=20000, help="training states")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-obs", type=int, default=32768, help="reports per training state")
    ap.add_argument("--test-max-obs", type=int, default=4096, help="reports per test state")
    ap.add_argument("--test-stride", type=int, default=4, help="test every n-th init")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    _logger.info("device %s %s", dev, torch.cuda.get_device_name(dev) if dev.type == "cuda" else "")
    t0 = time.time()

    latents, obs = open_latents(args.latents), ObsStore(args.obs)
    train_pairs = pairs(latents.bases, *args.train)
    test_bases = latents.bases[
        (latents.bases >= np.datetime64(args.test[0], "s"))
        & (latents.bases <= np.datetime64(args.test[1], "s"))
    ][:: args.test_stride]
    test_pairs = pairs(test_bases, *args.test)
    # an init forecasts 41 windows ahead: test inits must not share verification times with training
    last_train = max(b for b, _ in train_pairs)
    if min(b for b, _ in test_pairs) < last_train + STEP * latents.steps:
        raise ValueError("test inits start within 10 days of the last training init")
    _logger.info("%d training states, %d test states", len(train_pairs), len(test_pairs))

    mean, std = channel_stats(obs, train_pairs)
    mean_t, std_t = torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev)
    _logger.info("channels %s mean %s std %s", obs.channels, mean.round(2), std.round(2))

    train = StateDataset(latents, obs, train_pairs, max_obs=args.max_obs)
    n_features = train[0]["features"].shape[1]
    head = Head(n_features, len(obs.channels)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda i: min(1, (i + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(i / args.iters, 1))),
    )

    it, losses = 0, []
    while it < args.iters:
        for item in loader(train, True, args.workers):
            if len(item["values"]) == 0:
                continue
            y = (item["values"].to(dev) - mean_t) / std_t
            pred = head(
                item["tokens"].to(dev).float(), item["nbours"].to(dev), item["features"].to(dev)
            )
            mask = torch.isfinite(y)
            loss = ((pred - torch.nan_to_num(y)) ** 2)[mask].mean()
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at {item['base']} step {item['step']}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            it += 1
            if it % 100 == 0:
                _logger.info(
                    "iter %d loss %.4f %.0fs", it, np.mean(losses[-100:]), time.time() - t0
                )
            if it == args.iters:
                break
    torch.save(
        {
            "head": head.state_dict(),
            "n_features": n_features,
            "channels": obs.channels,
            "geoinfos": obs.geoinfos,
            "mean": mean,
            "std": std,
            "args": vars(args),
        },
        out / "head.pt",
    )

    head.eval()
    test = StateDataset(latents, obs, test_pairs, max_obs=args.test_max_obs, seed=1)
    rows = {name: [] for name in ["base", "step", "time", "lat", "lon", "pred", "obs"]}
    with torch.no_grad():
        for item in loader(test, False, args.workers):
            n = len(item["values"])
            if n == 0:
                continue
            pred = head(
                item["tokens"].to(dev).float(), item["nbours"].to(dev), item["features"].to(dev)
            )
            rows["pred"].append((pred * std_t + mean_t).cpu().numpy())
            rows["obs"].append(item["values"].numpy())
            rows["base"].append(np.full(n, np.datetime64(item["base"], "s").astype(np.int64)))
            rows["step"].append(np.full(n, item["step"]))
            rows["time"].append(np.asarray(item["times"]))
            rows["lat"].append(np.asarray(item["lat"]))
            rows["lon"].append(np.asarray(item["lon"]))
    rows = {name: np.concatenate(v) for name, v in rows.items()}
    np.savez(out / "predictions.npz", channels=np.array(obs.channels), **rows)

    scores = {}
    for c, name in enumerate(obs.channels):
        err = rows["pred"][:, c] - rows["obs"][:, c]
        ok = np.isfinite(err)
        scores[name] = {
            int(k): round(float(np.sqrt(np.mean(err[ok & (rows["step"] == k)] ** 2))), 3)
            for k in np.unique(rows["step"])
        }
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    for name, s in scores.items():
        leads = " ".join(f"k{k}:{v}" for k, v in s.items() if k in (1, 2, 4, 10, 20, 30, 40))
        _logger.info("%s %s", name, leads)
    _logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
