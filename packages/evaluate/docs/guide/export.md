# Exporting WeatherGenerator outputs

If the scores/evaluation metrics you require are more custom than those available in the [config](config.md), you can use the `export` functionality of the package to convert the native WeatherGenerator output to NetCDF or GRIB (for quaver).
This is a command line utility. Use:

```
uv run export --run-id <INFERENCE_ID> --stream ERA5 --output-dir ../output_nc --format netcdf --regrid-degree 1 --regrid-type regular_ll
--samples 0 --rank 0 --fsteps 1 2 3
```

## Required Arguments

|Argument | Choices | Description |
|-|-|-|
| `--run-id` | <str> | id produced by inference run; the code will use this to find the correct results folder and is used to create default output filepaths. |
| `--stream`|  |users can add their own streams this input is flexible, commonly used streams are `ERA5`, `IMERG` etc|
|`--output-dir`||where to save the converted outputs|
| `--format`|`netcdf`, `quaver`, `verif`|`quaver`  produces GRIB files to use with ECMWF quaver tool, `netcdf` for CF-compliant NetCDF files, `verif` for NetCDF files compatible with MetNor verif tool|


## Optional Arguments
|Argument | Choices | Description |
|-|-|-|
| `--type` | `prediction`, `target` | Choose from either type of data or list both to convert  |
| `--samples` | |list of samples of the inference run to export e.g. 0 1 2; defaults to all if not specified |
|`--fsteps`| |list of forecast steps to retrieve e.g. 1 2 3; defaults to all if not provided |
|`--channels`|| list of channels to retrieve e.g. 'q_500 t_2m'; defaults to all if not specified|
|`--n-processes`|| number of parallel processes to use for data retrieval. Default is 8|
|`--fstep-hours`|| Time difference between forecast steps in hours. Default is 6|
|`--epoch`| | Epoch used to specify zarr store. Default 0 |
| `--rank`| | Rank for specific zarr store. Default all |

## Quaver arguments

if `--format` set to `quaver` the additional parameters are needed:

- `--quaver-template-folder`: Path to the GRIB template file
- `--quaver-template-grid-type`: Grid type to include in the output file name e.g. O96 or N320. Defaults to O96
- `--expver`: Expver to include in output file name i.e. "iuoo"

A more comprehensive look at running Quaver can be found under [Quaver Setting](export/quaver.md).

## Verif arguments

if `--format` is set to `verif` the additional parameters are required:

|Argument | Choices | Description |
|-|-|-|
|`--verif-template`||Default is "verif/%S/%V/%R_%S_%V_%M_%D.nc", which creates files in a `output/verif/%S/%V` folder where %S, %V, %M, %D, %R are replaced by the streams, variable, method, date, and run ID"|
|`--obs`, `-b` | | netcdf file containing observation stations for which WeatherGenerator data will be compared for using the verif tool|
|`--method`| `2d`, `lat_lon`, `nearest`| This is the method used by the `verif_parser` to interpolate from the data on the WeatherGenerator output grid to the observation locations |

## Regridding arguments

- `--regrid-type`: `regular_ll` for a regular latitude- longitude grid , `O` for octahedral reduced Gaussian or `N` for reduced Gaussian grid
- `--regrid-degree`: Float number that defines the degree to regrid the data to for a regular lat/lon grid (e.g. 0.25 for 0.25x0.25 degree grid) or O/N Gaussian grid (e.g., 320 for N320 grid), depending on the input of `regrid_type` These are subject to the regridding matrices available from [earthkit-regrid](https://earthkit-regrid.readthedocs.io/en/latest/guide/precomputed/inventory/index.html)