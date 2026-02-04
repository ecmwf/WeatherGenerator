.. _quaver:

Quaver Instructions
===================

This file summarises the instructions to compute and plot scores with QUAVER. 

Templates
+++++++++
The first step is to create the grib files out of the WG output. For that we use EarthKit, which needs to some grib templates with the right format.  
The templates for the o96 grid are here: `/ec/weathergen/quaver_templates/`

Download templates for different grids
++++++++++++++++++++++++++++++++++++++
If you need a template for a different grid you can download it from mars. 
Examples of mars requests for the o96 grid are in the `/ec/weathergen/quaver_templates/` folder under `req_aifs_pl` or `req_aifs_sl`.

Change the grid type into the request and run it like this:

.. code-block::

    mars /ec/weathergen/quaver_templates/req_aifs_pl -> pressure levels
    mars /ec/weathergen/quaver_templates/req_aifs_sl -> surface variables


No need to request the whole time sequence. Just one timestep is enough to get the template (and it is actually faster for eathkit).


Create grib files
+++++++++++++++++
Once you have the grib templates with the correct grid type for your data, you need to convert your data into grib. You can do it with the `export` command: 

.. code-block::
    uv run export --run-id buydgjm5 --stream ERA5 --output-dir /ec/weathergen/quaver_checkpoints/ --format quaver --type prediction --fsteps 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 --quaver-template-folder "/ec/weathergen/quaver_templates/" --quaver-template-grid-type o96 --expver iuoo --n-processes 12


this can take a while for long runs, use a screen or a tmux session and you can leave `{level_type}` empty, i.e., using `aifs_{}_o96_data.grib`, to process both pressure levels (pl) and surface fields (sfc) simultaneously.

.. note::
    Quaver scores are computed at valid times: 00:00 and 12:00, so you just need to convert the steps with those valid times. The others will not have Pangu/AIFS/GraphCast counterparts. 

Note on exp_ver
+++++++++++++++

The ``exp_ver`` label is an internal id that quaver uses to store the scores on the database. We can't arbitrarily choose it but we should generate one through ``prep ml``. 
.. code-block::
    ml prepml
    prepml expver --create

until we clarify how to properly do it, you can use the existing exp_ver (`iuoo`) **no need to change it** in the code. 

Compute scores
++++++++++++++
Use the following script ``compute_quaver_scores.py`` in the private repo:

.. code-block:: console
    ml quaver
    quaver ../WeatherGenerator-private/data/preprocessing/quaver/compute_quaver_scores.py <name of my grib file> <exp_ver> <start_date> <end_date> <first_forecast_step[hours]> <last_forecast_step[hours]> <frequency> <grid> <comment> <"ea"/"od">

example:

.. code-block::
    quaver ../WeatherGenerator-private/data/preprocessing/quaver/compute_quaver_scores.py /ec/weathergen/quaver_checkpoints/prediction_pl_buydgjm5_iuoo.grib iuoo 2022100100 2022100418 12 120 12 O96 buy ea

the <comment> (here `buy`) should be a unique id that you give to your run. Imagine expver as a big bucket for the WeatherGen runs into the quaver database and **the comment here as an identifier for your runs**. There's a max length for the comment, so keep it as short as possible. **Tip**: use the first three letters of the `run_id`.

See Section "Debugging tips" to find the correct datetimes and steps. (TODO: improve this in the future.)

"ea"/"od": Compute the scores against ERA5 ("ea") or against IFS analysis ("od"). Soon we will add support for the observations. See "ERA5 vs operational analysis" paragraph for more details.

To search for all computed scores that are available in the quaver database, visit the following page, set ``expver`` to ``iouu`` and hit the search button.
https://sites.ecmwf.int/ecverify/quaverdb-browse/dblookup.py

Choose the right variables:
+++++++++++++++++++++++++++

The code automatically detects the variables to use by the file name. These are the default variables:

Pressure levels
++++++++++++++++

.. code-block:: python
    specifics=specifics(
            levtype="pl",
            parameter=["v", "u", "t", "z"],
            level=[850, 500],
            grid=grid_resol,
            intgrid = "off", 
            truncation="off",
            score=['rmsef', 'sdef', 'mef', 'maef'],
            domain=domains)

Surface levels
+++++++++++++++

.. code-block:: python
    specifics=specifics(
            levtype="sfc",
            parameter=["msl", "2t", "10u", "10v"],
            grid=grid_resol,
            intgrid = "off", 
            truncation="off",
            score=['rmsef', 'sdef', 'mef', 'maef'],
            domain=domains)



ERA5 vs operational analysis
++++++++++++++++++++++++++++
- To compare against ERA5 you should use:

.. code-block:: python
    reference=analysis(
            Class="ea", #ea = era5 /od = operational analysis
            expver="0001",
            
        ),

- To compare against the IFS Operational Analysis you should use: 

.. code-block:: python
    reference=analysis(
            Class="od", #ea = era5 /od = operational analysis
            expver="0001",
            
        ),

We expect a few percent difference between the two. All other models (Pangu etc.) are compared against the **Operational Analysis**. To obtain perfect closure with Target you should compare against ERA5. 

A list of the other operational verification scores is <here https://confluence.ecmwf.int/display/VER/Overview+of+the+operational+verification+scores>

Plot scores
++++++++++++

Use the following script: ``plot_scores.py``. 
The existing scores for AIFS/GraphCast etc are only computed every 12h. 

It contains the curves for PanguWeather, GraphCast, IFS (which are available for 2022) and AIFS (only available > 2023). Add the WeatherGen curve you want to plot using the expver and the comment above and run it with:   

.. code-block:: console

    ml quaver
    quaver ../WeatherGenerator-private/data/preprocessing/quaver/plot_scores.py <expver> <start_date> <end_date> <time_freq> <first_forecast_step[hours]> <last_forecast_step[hours]> <frequency>

example

.. code-block:: console

    quaver ../WeatherGenerator-private/data/preprocessing/quaver/plot_scores.py iuoo 2022100100 2022100300 6 12 240 6

Debugging tips
++++++++++++++

If quaver fails it might be that you are setting the wrong dates in `compute_scores.py`. The idea here is to check which `time` and `step` values you have in the grib file as follows:


.. code-block:: python

    >>> import xarray as xr
    >>> import cfgrib
    >>> ds = xr.open_dataset("/ec/perm/ecm9336/test_weathergen_sfc_ciga1p9c_iuoo_target.grib", engine = "cfgrib")
    >>> ds
    <xarray.Dataset> Size: 30MB
    Dimensions:            (time: 5, step: 18, values: 40320)
    Coordinates:
    * time               (time) datetime64[ns] 40B 2022-10-01 ... 2022-10-31
    * step               (step) timedelta64[ns] 144B 0 days 12:00:00 ... 9 days...
        meanSea            float64 8B ...
        latitude           (values) float64 323kB ...
        longitude          (values) float64 323kB ...
        valid_time         (time, step) datetime64[ns] 720B ...
        heightAboveGround  float64 8B ...
    Dimensions without coordinates: values
    Data variables:
        msl                (time, step, values) float32 15MB ...
        t2m                (time, step, values) float32 15MB ...
    Attributes:
        GRIB_edition:            2
        GRIB_centre:             ecmf
        GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
        GRIB_subCentre:          0
        Conventions:             CF-1.7
        institution:             European Centre for Medium-Range Weather Forecasts
        history:                 2025-10-10T13:42 GRIB to CDM+CF via cfgrib-0.9.1...
    >>> ds.time.values
    array(['2022-10-01T00:00:00.000000000', '2022-10-08T12:00:00.000000000',
        '2022-10-16T00:00:00.000000000', '2022-10-23T12:00:00.000000000',
        '2022-10-31T00:00:00.000000000'], dtype='datetime64[ns]')
    >>> ds.step.values
    array([ 43200000000000,  86400000000000, 129600000000000, 172800000000000,
        216000000000000, 259200000000000, 302400000000000, 345600000000000,
        388800000000000, 432000000000000, 475200000000000, 518400000000000,
        561600000000000, 604800000000000, 648000000000000, 691200000000000,
        734400000000000, 777600000000000], dtype='timedelta64[ns]')
    >>> ds.step.values.astype('timedelta64[h]')
    array([ 12,  24,  36,  48,  60,  72,  84,  96, 108, 120, 132, 144, 156,
        168, 180, 192, 204, 216], dtype='timedelta64[h]')

- ``time[0]`` corresponds to your <start_date> 
- ``time[-1]`` corresponds to your <end_date>  
- ``step[0]`` (in hours not ns!) corresponds to your <first_forecast_step[hours]>
- ``step[-1]`` (in hours not ns!) corresponds to your <last_forecast_step[hours]>
- The frequency is the difference between 2 forecast steps here. It should always be `12` in case you want to compare with AIFS/IFS etc.. for WG run intercomparisons it can also be 6. 



