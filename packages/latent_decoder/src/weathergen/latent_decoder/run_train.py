# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Train a latent decoder on stored latents and the reports of one obs stream, then validate it.

usage: train_latent_decoder [--config FILE ...] [--options key=value ...]

Writes to results_path: config.yml, latent_decoder.pt, predictions.npz (one row per validation
report: init_time, fstep, datetime, lat, lon, prediction and observation per channel) and
scores.json (RMSE per forecast step and channel).
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.latent_decoder import geometry
from weathergen.latent_decoder.data_reader_obs import DataReaderObs, ReaderData
from weathergen.latent_decoder.latents import NUM_FSTEPS, TIME_STEP, LatentReader
from weathergen.latent_decoder.model import LatentDecoder

_logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "default_config.yml"
# step 0 is the encoded source window, whose reports were model inputs
FSTEPS = range(1, NUM_FSTEPS)
# geoinfos are used as they are, except station height, which is in metres
GEOINFO_SCALE = {"stalt": 1e-3}


def get_samples(init_times, periods, stride=1):
    """(init_time, fstep) for every fstep of every init in the periods."""
    selected = []
    for start_date, end_date in periods:
        t = init_times[
            (init_times >= np.datetime64(start_date, "s"))
            & (init_times <= np.datetime64(end_date, "s"))
        ]
        selected.extend(t[::stride])
    return [(t, fstep) for t in selected for fstep in FSTEPS]


def target_coords(rdata, geoinfo_channels, fstep, dtr, cells):
    """[n, f]: position (6), time within the window (1), forecast step (1), geoinfos."""
    dt = (rdata.datetimes - np.datetime64(dtr.start, "ns")) / TIME_STEP.astype("timedelta64[ns]")
    scale = np.array([GEOINFO_SCALE.get(g, 1.0) for g in geoinfo_channels], dtype=np.float32)
    return np.concatenate(
        [
            geometry.position_features(rdata.coords[:, 0], rdata.coords[:, 1], cells),
            dt[:, None].astype(np.float32),
            np.full((len(cells), 1), fstep / (NUM_FSTEPS - 1), np.float32),
            np.nan_to_num(rdata.geoinfos * scale),
        ],
        axis=1,
    ).astype(np.float32)


class LatentDataset(torch.utils.data.Dataset):
    """One sample per (init_time, fstep): the state's tokens and every report in its window."""

    def __init__(self, latents, reader, samples, max_num_targets=-1, rng_seed=0):
        self.latents, self.reader, self.samples = latents, reader, samples
        self.max_num_targets, self.rng_seed = max_num_targets, rng_seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        init_time, fstep = self.samples[i]
        state = self.latents.get_state(init_time, fstep)
        rdata = self.reader.get_target(state.dtr)
        n = len(rdata.data)
        if 0 < self.max_num_targets < n:
            pick = np.random.default_rng((self.rng_seed, i)).choice(n, self.max_num_targets, False)
            rdata = ReaderData(**{name: v[pick] for name, v in vars(rdata).items()})
        cells = geometry.cell_of(rdata.coords[:, 0], rdata.coords[:, 1])
        coords = target_coords(rdata, self.reader.geoinfo_channels, fstep, state.dtr, cells)
        return {
            "init_time": str(init_time),
            "fstep": fstep,
            # float16 halves what workers pass around; it is well below the run-to-run noise
            "tokens": torch.from_numpy(state.tokens.astype(np.float16)),
            "nbours": torch.from_numpy(geometry.hp_nbours()[cells]),
            "target_coords": torch.from_numpy(coords),
            "target": torch.from_numpy(self.reader.normalize_target_channels(rdata.data)),
            "datetimes": rdata.datetimes.astype("datetime64[s]").astype(np.int64),
            "coords": rdata.coords,
        }


def load_config(config_files, options):
    cf = OmegaConf.merge(
        OmegaConf.load(_DEFAULT_CONFIG),
        *(OmegaConf.load(f) for f in config_files),
        OmegaConf.from_cli(options),
    )
    val_cf = OmegaConf.merge(cf.training_config, cf.validation_config)
    return cf, val_cf


def check_periods(train_samples, val_samples):
    """An init forecasts 41 windows ahead: validation inits must not share valid times with
    training inits."""
    train_times = np.unique([t for t, _ in train_samples])
    for t in np.unique([t for t, _ in val_samples]):
        if np.abs(train_times - t).min() < TIME_STEP * NUM_FSTEPS:
            raise ValueError(f"validation init {t} is within 41 steps of a training init")


def lr_lambda(step, cf):
    lr_cf = cf.learning_rate_scheduling
    num_steps = cf.num_mini_epochs * cf.samples_per_mini_epoch
    warmup = min(1, (step + 1) / lr_cf.num_steps_warmup)
    return warmup * 0.5 * (1 + math.cos(math.pi * min(step / num_steps, 1)))


def train() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, nargs="*", default=[], help="config files")
    parser.add_argument("--options", nargs="+", default=[], help="overrides: key=value")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cf, val_cf = load_config(args.config, args.options)
    results_path = Path(cf.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cf, results_path / "config.yml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info(f"device {device}")
    t0 = time.time()

    latents, reader = LatentReader(cf.latents_path), DataReaderObs(cf.stream)
    train_samples = get_samples(latents.init_times, cf.training_config.periods)
    val_samples = get_samples(latents.init_times, val_cf.periods, val_cf.init_time_stride)
    check_periods(train_samples, val_samples)
    _logger.info(f"{len(train_samples)} training samples, {len(val_samples)} validation samples")

    num_workers = cf.data_loading.num_workers
    train_ds = LatentDataset(
        latents, reader, train_samples, cf.training_config.max_num_targets, cf.data_loading.rng_seed
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=None,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    dim_coords = train_ds[0]["target_coords"].shape[1]
    model = LatentDecoder(dim_coords, len(reader.target_channels), **cf.decoder).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cf.training_config.learning_rate_scheduling.lr_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, cf.training_config)
    )

    def predict(sample):
        return model(
            sample["tokens"].to(device).float(),
            sample["nbours"].to(device),
            sample["target_coords"].to(device),
        )

    step, losses = 0, []
    num_steps = cf.training_config.num_mini_epochs * cf.training_config.samples_per_mini_epoch
    while step < num_steps:
        for sample in train_loader:
            if len(sample["target"]) == 0:
                continue
            target = sample["target"].to(device)
            mask = torch.isfinite(target)
            loss = ((predict(sample) - torch.nan_to_num(target)) ** 2)[mask].mean()
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at {sample['init_time']} step {sample['fstep']}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % cf.training_config.samples_per_mini_epoch == 0:
                _logger.info(
                    f"mini-epoch {step // cf.training_config.samples_per_mini_epoch} "
                    f"loss {np.mean(losses):.4f} {time.time() - t0:.0f}s"
                )
                losses = []
            if step == num_steps:
                break
    torch.save(
        {
            "model": model.state_dict(),
            "dim_coords": dim_coords,
            "config": OmegaConf.to_container(cf),
        },
        results_path / "latent_decoder.pt",
    )

    model.eval()
    val_ds = LatentDataset(latents, reader, val_samples, val_cf.max_num_targets, rng_seed=1)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=None, num_workers=num_workers)
    rows = {name: [] for name in ["init_time", "fstep", "datetime", "lat", "lon", "pred", "obs"]}
    with torch.no_grad():
        for sample in val_loader:
            n = len(sample["target"])
            if n == 0:
                continue
            pred = reader.denormalize_target_channels(predict(sample).cpu().numpy())
            rows["pred"].append(pred)
            rows["obs"].append(reader.denormalize_target_channels(sample["target"].numpy()))
            init_time = np.datetime64(sample["init_time"], "s").astype(np.int64)
            rows["init_time"].append(np.full(n, init_time))
            rows["fstep"].append(np.full(n, sample["fstep"]))
            rows["datetime"].append(np.asarray(sample["datetimes"]))
            rows["lat"].append(np.asarray(sample["coords"])[:, 0])
            rows["lon"].append(np.asarray(sample["coords"])[:, 1])
    rows = {name: np.concatenate(v) for name, v in rows.items()}
    np.savez(results_path / "predictions.npz", channels=np.array(reader.target_channels), **rows)

    scores = {}
    for c, name in enumerate(reader.target_channels):
        err = rows["pred"][:, c] - rows["obs"][:, c]
        ok = np.isfinite(err)
        scores[name] = {
            int(k): round(float(np.sqrt(np.mean(err[ok & (rows["fstep"] == k)] ** 2))), 3)
            for k in np.unique(rows["fstep"])
        }
    (results_path / "scores.json").write_text(json.dumps(scores, indent=1))
    for name, s in scores.items():
        leads = " ".join(f"k{k}:{v}" for k, v in s.items() if k in (1, 2, 4, 10, 20, 30, 40))
        _logger.info(f"{name} {leads}")
    _logger.info(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train()
