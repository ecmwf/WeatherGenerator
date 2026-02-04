External models
===============

The FastEvaluation package handles comparisons from different models with the WeatherGenerator outputs but first requires **precomputed scores** for other models must be provided in the following format, saved under the ``metrics_dir`` specified under the model with ``type: csv`` in the config (see the config for an example with PanguWeather)

.. code-block::
    ,parameter,level,number,score,step,date,domain_name,value
    0,t,925,0,mef,0 days 12:00:00,2022-10-01 00:00:00,n.hem,0.031371469251538386
    1,t,925,0,mef,0 days 12:00:00,2022-10-01 12:00:00,n.hem,-0.010387031341104752
    2,t,925,0,mef,0 days 12:00:00,2022-10-02 00:00:00,n.hem,0.030255780718550083
    3,t,925,0,mef,0 days 12:00:00,2022-10-02 12:00:00,n.hem,-0.028894746338016246



