# weathergen-latent-decoder

Train a simple decoder on stored WeatherGenerator latent states and observations, without
installing WeatherGenerator. Needs numpy, torch, zarr, astropy-healpix and omegaconf.

```
pip install packages/latent_decoder
```

| module | what it does |
|---|---|
| `latents.py` | reads the stored states; the only code that knows the storage format |
| `data_reader_obs.py` | reads a WeatherGenerator obs stream; replace it for other data |
| `geometry.py` | HEALPix level-5 cells, neighbours and position features |
| `model.py` | the decoder: target coordinates attend to a cell's token and its 8 neighbours |
| `run_train.py` | trains and validates a decoder |
| `config/default_config.yml` | the default config |

```python
from weathergen.latent_decoder.latents import LatentReader

reader = LatentReader("/ec/weathergen/latents/sdnbfpve-v1-raw/2023")
state = reader.get_state(reader.init_times[0], 1)  # tokens [12288, 2048], register_tokens, dtr
```

```
train_latent_decoder --config my_config.yml --options results_path=./runs/synop
```

The config follows WeatherGenerator's: `training_config` and `validation_config` with their
periods of init times, the obs `stream` as in `config/streams/`, and `--options key=value`
overrides.

State `fstep` of an init at time t covers [t + 6h·fstep, t + 6h·(fstep + 1)). Step 0 is the
encoded source window, whose observations were model inputs, so training and validation use
steps 1 to 40. Validation inits must be at least 41 steps (10 days 6 h) from every training init.

Token c is the HEALPix nested cell c of WeatherGenerator's tokenizer, which uses longitude + 180°
(see `geometry.py`). The `coords` stored by the latent writer before this was fixed are 180° off;
this package does not read them.

Tests: `python -m pytest packages/latent_decoder/tests`.
