Exporting WeatherGenerator outputs
==================================

If the scores/evaluation metrics you require are not found above, you can use the ``export`` functionality of the package to convert the native WeatherGenerator output to NetCDF or GRIB (for quaver).
This is a command line utility. Use:

.. code-block:: console

    uv run export --run-id <INFERENCE_ID> --stream ERA5 --output-dir ../output_nc --format netcdf --regrid-degree 1 --regrid-type regular_ll

Required Arguments
++++++++++++++++++

- ``--stream``: currently restricted to ERA5, will take the ERA5 data source from an individual inference run. In the future will add functionality for other data sources e.g. IMERG etc
- ``--output-dir``: where to save the converted outputs
- ``--format``: ``quaver`` which produces GRIB files to use with ECMWF quaver tool or ``netcdf`` for CF-compliant NetCDF files

Optional Arguments
++++++++++++++++++

- ``--type``: list of data to convert, choose from ``prediction``, ``target`` or both; defaults to only ``prediction`` if nothing specified
- ``--samples``: list of samples of the inference run to export e.g. 0 1 2; defaults to all if not specified
- ``--fsteps``: list of forecast steps to retrieve e.g. 1 2 3; defaults to all if not provided 
- ``--channels``: list of channels to retrieve e.g. 'q_500 t_2m'; defaults to all if not specified
- ``--n-processes``: number of parallel processes to use for data retrieval. Default is 8
- ``--fstep-hours``: Time difference between forecast steps in hours. Default is 6
- ``--epoch``: Epoch used to specify zarr store. Default 0
- ``--rank``: Rank for specific zarr store. Default 0

Quaver arguments
++++++++++++++++

if ``--format`` set to ``quaver`` the additional parameters are needed:

- ``--quaver-template-folder``: Path to the GRIB template file
- ``--quaver-template-grid-type``: Grid type to include in the output file name e.g. O96 or N320. Defaults to O96
- ``--expver``: Expver to include in output file name i.e. "iuoo"

A more comprehensive look at running Quaver can be found under :ref:`quaver`.

Regridding arguments
++++++++++++++++++++

- ``--regrid-type``: ``regular_ll`` for a regular latitude- longitude grid , ``O`` for octahedral reduced Gaussian or ``N`` for reduced Gaussian grid
- ``--regrid-degree``: Float number that defines the degree to regrid the data to for a regular lat/lon grid (e.g. 0.25 for 0.25x0.25 degree grid) or O/N Gaussian grid (e.g., 63 for N63 grid), depending on the input of ``regrid_type``