# Example extra: tropical cyclone case studies
A simple implementation of tropical cyclone detection and tracking, used here to analyze a single prediction of a single storm of interest in terms of the track error, wind speed and pressure. The underlying algorithm is designed also for future uses, for example computing statistics of simulated tropical storms. 

### Usage
Workflow:
1. Export some inference data to netcdf, regridded to 1°x1° like so:
```
uv run export --run-id <INFERENCE_ID> --stream ERA5 \
--output-dir <OUTDIR> --format netcdf --regrid-degree 1 \
--regrid-type regular_ll \
--channel 10u 10v msl
```
2. Export the target with the same command, adding `--type target`
3. Run the casestudy: `uv run TC_casestuy_main.py`

Notes:
* all parameters are set in `TC_config.yml`, make sure to set `inpath` to the location of your exported netcdf data
* the parameters set there are for a regular 1°x1° grid. The code may also work for other grids but you probably have to adapt `laplace_size` and maybe `peak_separation`.
* Make sure that the timesteps specificed in TC_config.yml are within the simulation and the storm is actually present in the data at the specified time.

### Algorithm
* Detection:
    1. apply gaussian laplace filter (`scipy.ndimage.
gaussian_laplace`) to each msl field. 
    2. select local minima of the laplacian, separated by some minimum distance
    3. drop points where 
        * laplacian value is too low (`< tracking_params.laplace_threshold`)
        * wind speed is too low  (`< tracking_params.wind_threshold`)
        * pressure is too high  (`> tracking_params.pressure_threshold`)
* Tracking: for each storm, search the previous timestep for all storms within a maximum distance (`tracking_params.merge_distance`), then merge storms into tracks starting with the closest match. 
* Matching with the selected storm: at the specified time (`selected_storm.time`) search target and prediction for the storm closest to the specified location (`selected_strom.lon,lat`). Then compare the target and prediction track to which those storms belong. 

### Parameters

| **Parameter** | **Description** |
| --- | --- |
| ``runid`` | id of the inference run |
| ``init_time`` | Initialization time to use |
| ``inpath`` | Directory containing the netcdf files |
| ``outpath`` | Directory where plots will be saved |
| ``latmin`` | Minimum latitude considered for tropical cyclone detection |
| ``latmax`` | Maximum latitude considered for tropical cyclone detection |
| ``selected_storm.lon`` | Longitude of the storm selected for analysis |
| ``selected_storm.lat`` | Latitude of the storm selected for analysis |
| ``selected_storm.time`` | Timestamp at which to find the storm in the run |
| ``tracking_params.laplace_size`` | Size of the Laplacian filter kernel in units gridboxes |
| ``tracking_params.laplace_threshold`` | Threshold applied to the Laplacian field to identify low‑pressure regions |
| ``tracking_params.pressure_threshold`` | Minimum pressure value used to filter candidate cyclone centers |
| ``tracking_params.wind_threshold`` | Minimum wind speed required for a detection to be considered |
| ``tracking_params.peak_separation`` | Minimum separation (in gridboxes) between distinct pressure minima |
| ``tracking_params.merge_distance`` | Maximum distance (in km) for merging nearby detections into a single track |