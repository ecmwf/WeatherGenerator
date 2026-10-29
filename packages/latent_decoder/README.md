# weathergen-latent-decoder

Train a simple decoder on stored WeatherGenerator latent states and observations, without
installing WeatherGenerator. Needs numpy, torch, zarr and astropy-healpix.

```
pip install packages/latent_decoder
```

| module | what it does |
|---|---|
| `latents.py` | reads the stored states; the only code that knows the storage format |
| `obs.py` | reads WeatherGenerator observation zarr stores; replace it for other data |
| `geometry.py` | HEALPix level-5 cells, neighbours and position features |
| `head.py` | the decoder head: attention over a cell and its 8 neighbours, then an MLP |
| `dataset.py` | pairs each state with the reports in its 6-h window |
| `train_synop.py` | trains and tests a head on SYNOP reports |

```python
from weathergen.latent_decoder.latents import open_latents

lat = open_latents("/ec/weathergen/latents/sdnbfpve-v1-raw/2023")
s = lat.state(lat.bases[0], 1)  # tokens [12288, 2048], registers [64, 2048], window
```

```
python -m weathergen.latent_decoder.train_synop LATENTS OBS OUT \
    --train 2023-01-01T00 2023-09-30T18 --test 2023-10-15T00 2023-12-31T18
```

State k of an init with base time b covers [b + 6h·k, b + 6h·(k+1)). Step 0 is the encoded
source window, whose observations were model inputs, so training and testing use steps 1 to 40.
Test inits must start at least 41 steps (10 days 6 h) after the last training init.

Token c is the HEALPix nested cell c of WeatherGenerator's tokenizer, which uses longitude + 180°
(see `geometry.py`). The `coords` stored by the latent writer before this was fixed are 180° off;
this package does not read them.

Tests: `python -m pytest packages/latent_decoder/tests`.
