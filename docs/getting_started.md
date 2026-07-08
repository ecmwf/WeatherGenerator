# Getting started

This page explains how to set up and run the WeatherGenerator project.
This tutorial assumes you the following:
- You have installed the `uv` tool version 0.10+
- You have a modern NVidia GPU

# Train a model

This tutorial goes through the following steps:
1. Fetch a sample from the ERA5 dataset
2. Set up the dependencies and the python environment
3. Train a model using simple parameters

*NOTE: the trained model will not be competitive. This configuration is for demonstration
purposes only.*

## Set up directories

```bash
mkdir -p logs models output results
```

## Set up pyton environment

```sh
./scripts/actions.sh sync
```

## Fetch data

The file `era5_o96_2020_1pct.yaml` contains an Anemoi dataset template for loading 1% of the ERA5 dataset.

```sh
mkdir -p datasets
cd datasets
uv run --with " anemoi-datasets[remote]" anemoi-datasets create --overwrite ../docs/era5_o96_2020_1pct.yaml era5-o96-2020-1pct-6h-v1.zarr
cd ..
```

## Train a model

```sh
WEATHERGEN_PRIVATE_CONF=./config/toy/toy_era5_private.yml uv run train --base-config ./config/toy/toy_era5.yml 
```

Running WeatherGenerator requires three pieces of configuration:
- a private configuration that points to the general environment (location of the datasets)
- a base configuration that describes the model architecture
- a description of the data streams (loaded as part of the base configuration)

# Run inference with a trained model

TODO
