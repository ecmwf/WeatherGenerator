# weathergen-latent-decoder

Train a decoder from stored WeatherGenerator latent states to observations, without installing
WeatherGenerator. The example decoder predicts SYNOP station reports (2 m temperature and
dewpoint, 10 m wind, pressure) from the latent tokens around each station.

## Install

Python 3.12. From the root of the WeatherGenerator repository:

```
uv venv --python 3.12 .venv-ld
source .venv-ld/bin/activate
uv pip install packages/latent_decoder
```

or, without uv, `python3.12 -m venv .venv-ld` and `pip install packages/latent_decoder`. This
installs numpy, torch, zarr, astropy-healpix and omegaconf, and the command
`train_latent_decoder`. On Linux the torch wheel from PyPI includes CUDA, on x86 and on GH200
(aarch64). Check with `python -c "import torch; print(torch.cuda.is_available())"` on a GPU node.

- torch is pinned to 2.9.1, as in WeatherGenerator; its PyPI wheels are built for CUDA 12. If
  your machine needs another torch, change the pin in `pyproject.toml` before installing.
- The environment is about 5 GB, mostly torch. On a parallel filesystem the install can take
  10 to 15 minutes, nearly all of it copying files.
- An older uv prints a warning that it cannot parse `[tool.uv.extra-build-dependencies]` in the
  repository's `pyproject.toml`. It does not affect this package.

## Data

- Latents: one directory per run of 8 initialisations, each with a `MANIFEST.json`, kept in one
  directory per year with a `README.md` that describes them. `latents_path` can point at one run,
  a year, or the directory of years.
- Observations: a WeatherGenerator obs zarr store, given as a stream (see below). The default is
  the combined surface store used to train the model.

The paths in the default config are those at ECMWF. Elsewhere, set `latents_path` and
`stream.filenames` to where the data is on your machine.

Each initialisation has 41 states. State `fstep` of an initialisation at time t covers the window
[t + 6h·fstep, t + 6h·(fstep + 1)); state 0 is the encoded input window, states 1 to 40 are the
forecast. A sample is one state with the reports in its window.

## Run

Copy the default config, `src/weathergen/latent_decoder/config/default_config.yml`, change what you
need, and pass it with `--config`. Single values can be overridden with `--options key=value`.

A quick test, about 2 minutes on one GPU (add `latents_path=...` and `stream.filenames=[...]` if
the data is not at the default paths):

```
train_latent_decoder --options results_path=./runs/test \
    training_config.periods="[['2023-01-01T00:00','2023-01-10T18:00']]" \
    training_config.num_mini_epochs=2 training_config.samples_per_mini_epoch=100 \
    validation_config.periods="[['2023-02-01T00:00','2023-02-01T18:00']]" \
    validation_config.init_time_stride=1
```

It trains on only 200 samples, so the scores are poor (2 m temperature RMSE about 5 K)
but show that everything works: `runs/test` gets the four files listed under Results.

The default config trains on January to September 2023 and validates on every 4th initialisation
from mid-October to December 2023: 20,000 samples, about 15 minutes on one GH200 with
`data_loading.num_workers=16`.

Run on a GPU node, not a login node: the data loaders read a lot and login nodes usually limit
memory. With Slurm, an interactive session is for example

```
srun --gpus=1 --cpus-per-task=32 --mem=240G --time=02:00:00 --pty bash
source .venv-ld/bin/activate
train_latent_decoder --config my_config.yml --options results_path=./runs/try1
```

with the account, partition or QoS your machine needs. As a batch job, put the last two lines in
a script with the matching `#SBATCH` options. Each sample reads a 100 MB state, so data loading
sets the pace: use `data_loading.num_workers` of 8 to 16, with a CPU core for each. The
reference run used 16 workers and 240 GB of memory.

## Config

| key | meaning |
|---|---|
| `latents_path`, `results_path` | where the latents are and where the results go |
| `stream` | the obs stream: `filenames`, `target` (channels to predict) and `geoinfo_channels`, written as in WeatherGenerator's `config/streams/`, with full column names |
| `decoder` | `dim_embed`, `num_heads` of the decoder |
| `training_config.periods` | initialisations to train on, one or more `[start_date, end_date]` |
| `training_config.num_mini_epochs`, `samples_per_mini_epoch` | training length; the loss is logged once per mini-epoch |
| `training_config.max_num_targets` | at most this many reports per sample (-1: all) |
| `training_config.learning_rate_scheduling` | `lr_max`, `num_steps_warmup`; cosine decay after the warm-up |
| `validation_config` | overrides `training_config` for validation: `periods`, `init_time_stride`, `max_num_targets` |
| `data_loading.num_workers` | data-loader processes |

An initialisation forecasts 41 windows ahead, so validation initialisations must be at least 41
states (10 days 6 h) from every training initialisation; the run stops otherwise. Do not train or
validate on state 0: its reports were inputs of the model.

## Results

`results_path` gets `config.yml` (the full config used), `latent_decoder.pt` (weights, config and
the number of input features), `predictions.npz` (one row per validation report: `init_time`,
`fstep`, `datetime`, `lat`, `lon`, `pred` and `obs` per channel, `channels`) and `scores.json`
(RMSE per channel and forecast step, also printed at the end).

## Your own data

- Another WeatherGenerator obs store: change `stream`, e.g. copy an entry from
  `config/streams/`. `target` and `geoinfo_channels` must be full column names of the store.
- Other observations: write a reader like `data_reader_obs.py`. It needs `target_channels`,
  `geoinfo_channels`, `get_target(dtr)` returning a `ReaderData` (`coords` [n, 2] as lat, lon in
  degrees, `geoinfos` [n, g], `data` [n, c] with NaN for missing values, `datetimes` [n]) for the
  reports with `dtr.start <= time < dtr.end`, and `normalize_target_channels` and
  `denormalize_target_channels`. Then construct it instead of `DataReaderObs` in `run_train.py`.
- Another decoder: change `model.py`. Its input is the tokens of one state [12288, 2048], the
  indices [n, 9] of each report's cell and its 8 neighbours, and per-report features [n, f]
  (position, time in the window, forecast step, geoinfos).
- The latents in your own code:

  ```python
  import numpy as np

  from weathergen.latent_decoder import geometry
  from weathergen.latent_decoder.latents import LatentReader

  reader = LatentReader("/path/to/sdnbfpve-v1-raw/2023")
  state = reader.get_state(reader.init_times[0], 1)  # tokens [12288, 2048], register_tokens, dtr
  lat, lon = np.array([51.5, 48.2]), np.array([-0.1, 16.4])
  cells = geometry.cell_of(lat, lon)  # the token of each point
  nbours = geometry.hp_nbours()[cells]  # [n, 9]: that token and its 8 neighbours
  ```

  `latents.py` is the only module that knows the storage format.

Token c is HEALPix level-5 nested cell c of WeatherGenerator's tokenizer, which uses longitude +
180° (see `geometry.py`). Use `geometry.cell_of` and `geometry.cell_centres` for positions. The
`coords` stored with the 2023 latents are 180° off; this package does not read them.

Tests: `python -m pytest packages/latent_decoder/tests`.
