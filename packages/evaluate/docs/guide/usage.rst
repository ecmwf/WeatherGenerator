Getting Started
===============

Prerequisites
------------

To use the FastEvaluation package you must first run inference. e.g.

.. code-block::

    uv run inference --from-run-id <model id> --samples 2 --options forecast_steps=5

Command Line Usage
------------------

This package is used after the uv run inference step. The user interface takes care of both scoring and plotting:

.. code-block::

    uv run evaluate --config <your plotting config.yml>

A template of the config file can be found here `WeatherGenerator/config/evaluate/eval_config.yml`_

.. _WeatherGenerator/config/evaluate/eval_config.yml: https://github.com/ecmwf/WeatherGenerator/blob/develop/config/evaluate/eval_config.yml
