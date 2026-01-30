Supported Metrics
~~~~~~~~~~~~~~~~~

Check
`score.py <https://github.com/ecmwf/WeatherGenerator/blob/main/packages/evaluate/src/weathergen/evaluate/scores/score.py>`__
for the stable list of scores available. Experimental scores can be
found on the develop branch
`here <https://github.com/ecmwf/WeatherGenerator/blob/develop/packages/evaluate/src/weathergen/evaluate/scores/score.py>`__.

**Requires target and prediction:**

- ``eta`` — Equitable Threat Score of forecast data w.r.t.reference data
- ``pss`` — Peirce Skill Score of forecast data w.r.t.reference data
- ``fbi`` — Frequency Bias Index of forecast data w.r.t.reference data
- ``mae`` — Mean Absolute Error of forecast data w.r.t.reference data
- ``l1`` — L1 error norm of forecast data w.r.t.reference data
- ``l2`` — L2 error norm of forecast data w.r.t.reference data
- ``mse`` — Mean Squared Error of forecast data w.r.t.reference data
- ``rmse`` — Root Mean Squared Error of forecast data w.r.t.reference
  data
- ``vrmse`` - Variance-normalized Root Mean Squared Error (VRMSE) of
  forecast data w.r.t.reference data
- ``grad_amplitude`` — Ratio between the spatial variability of
  differental operator with order 1 (higher values unsupported yet)
  forecast and ground truth data using the calc_geo_spatial-method.
  (requires regular lat-lon grid)
- ``psnr`` — Peak Signal-to-Noise Ratio
- ``seeps`` — Stable Equitable Error in Probability Space see `Rodwell
  et al.,
  2011 <https://journals.ametsoc.org/view/journals/mwre/140/8/mwr-d-11-00301.1.pdf>`__
  > Please note many of these scores use default hard coded threshold
  values; see the code
  `here <https://github.com/ecmwf/WeatherGenerator/blob/develop/packages/evaluate/src/weathergen/evaluate/scores/score.py>`__.

**Requires next step and alignment** \* ``froct`` — Forecast Rate of
Change over Time \* ``troct`` — Target Rate of Change over Time

**Requires climatology** \* ``acc`` — Anomaly Correlation Coefficient \*
``fact`` — Forecast Activity \* ``tact`` — Target Activity as standard
deviation of target anomaly

**Probability metrics** \* ``ssr`` - Spread-Skill Ratio of the forecast
ensemble data w.r.t. reference data \* ``crps`` - Wrapper around
CRPS-methods (Continuous ranked probability score) provided by
xskillscore-package. See
`here <https://xskillscore.readthedocs.io/en/stable/search.html?q=crps&check_keywords=yes&area=default>`__
\* ``rank_histogram`` - Rank Histogram of the forecast data w.r.t
reference data \* ``spread`` - Ensemble Spread of the forecast

   One needs to be cautious when computing scores that need alignment
   between steps or climatology. One can not: \* compute ``froct`` or
   ``troct`` for datasets where coordinates change between forecast
   steps (shuffled is fine) \* compute ``acc``, ``fact``, ``tact`` for
   datasets that do not have a precomputed climatology available
