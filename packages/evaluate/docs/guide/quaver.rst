.. _quaver:

Quaver Instructions
===================

This file summarises the instructions to compute and plot scores with QUAVER. 

Templates
+++++++++
The first step is to create the grib files out of the WG output. For that we use EarthKit, which needs to some grib templates with the right format.  

Download templates for different grids
++++++++++++++++++++++++++++++++++++++
If you need a template for a different grid you can download it from mars. 
Examples of mars requests for the o96 grid are in the ``$WEGEN_DATA_FOLDER/quaver_templates/`` folder under ``req_aifs_pl`` or ``req_aifs_sl``.

Change the grid type into the request and run it like this:

.. code-block::

    mars $WEGEN_DATA_FOLDER/req_aifs_pl -> pressure levels
    mars $WEGEN_DATA_FOLDER/req_aifs_sl -> surface variables


No need to request the whole time sequence. Just one timestep is enough to get the template (and it is actually faster for eathkit).


Create grib files
+++++++++++++++++
Once you have the grib templates with the correct grid type for your data, you need to convert your data into grib. You can do it with the ``export`` command: 

.. code-block::

    uv run export --run-id buydgjm5 --stream ERA5 --output-dir $WEGEN_DATA_FOLDER/quaver_checkpoints/ --format quaver --type prediction --fsteps 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 --quaver-template-folder "$WEGEN_DATA_FOLDER/quaver_templates/" --quaver-template-grid-type o96 --expver iuoo --n-processes 12


this can take a while for long runs, use a screen or a tmux session and you can leave ``{level_type}`` empty, i.e., using ``aifs_{}_o96_data.grib``, to process both pressure levels (pl) and surface fields (sfc) simultaneously.

.. note::

    Quaver scores are computed at valid times: 00:00 and 12:00, so you just need to convert the steps with those valid times. The others will not have Pangu/AIFS/GraphCast counterparts. 

.. note::
    
    ``exp_ver`` label is an internal id that quaver uses to store the scores on the database. We can't arbitrarily choose it but we should generate one: 

    .. code-block::
        ml pifsenv
        getNewId --class rd

If you are on HPC2020 the code can automatically generate a new one by using: `--expver NEW`

