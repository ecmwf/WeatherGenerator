# The `FastEvaluation` package

A modular evaluation and visualization framework for verifying forecast data and model outputs in the WeatherGenerator project.

---

## Overview

The **FastEvaluation** module of the WeatherGenerator has the following features:

- compute performance metrics and diagnostics for forecast or model outputs  
- produce maps, time‑series, and other visualizations for qualitative & quantitative evaluation  
- handle gridded and non gridded data (observations / reanalysis) 
- export the WeatherGenerator output into grib/netCDf files suitable to be used by the project partners.  


---

## Quick Start — Running the Evaluation Workflow

After the inference step you can run evaluation (on CPUs) as:
```
uv run evaluate --config <path to config file>
```

The default config file is at: `WeatherGenerator/configs/evaluate/eval_config.yml`

More instructions can be found here: https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/wikis/home/Common-workflows/fast-evaluation

---

## Licence
This package is licensed under the Apache‑2.0 License. 



