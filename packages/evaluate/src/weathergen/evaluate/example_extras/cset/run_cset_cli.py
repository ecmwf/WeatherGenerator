#!/usr/bin/env -S uv run --script
# ruff: noqa: E501
# /// script
# dependencies = [
#   "numpy==1.26.4",
#   "cf-units",
#   "scitools-iris>=3.11",
#   "mo-pack@git+https://github.com/SciTools/mo_pack.git@8fb088227f4ffd1b45823309feb65cc6495fb19a",
#   "pygraphviz",
#   "CSET==26.2.0",
#   "weathergen-common",
#   "omegaconf"
# ]
#
# [tool.uv.sources]
# weathergen-common = { path = "../../../../../../common" }
# ///
"""
Uses the CSET library to make plots specified using the config
"""
from pathlib import Path
from CSET.operators import execute_recipe
import yaml
from yaml.loader import SafeLoader
from importlib import resources as impresources
from CSET import recipes
import subprocess
import os
import shlex

#parsecommand line string
inp_file = impresources.files(recipes) / 'surface_fields/surface_structural_similarity_spatial_plot.yaml'
exp_name = "ion2tz7w_structuralsimilarity"
output_dir = f"/users/sowens/WeatherGenerator/plots/cset/{exp_name}"
os.makedirs(output_dir, exist_ok=True)
command_string =f"""cset -v bake -r {inp_file} -o {output_dir}
    --INPUT_PATHS='["/users/sowens/ion2tz7w/prediction_2022-10-02T06_ion2tz7w.nc","/users/sowens/ion2tz7w/target_2022-10-02T06_ion2tz7w.nc"]' \
    --VARNAME='air_pressure_at_mean_sea_level' \
    --METHOD='MEAN' \
    --BASE_MODEL='target', \
    --OTHER_MODEL='prediction'
    --SUBAREA_TYPE='None' \
    --SUBAREA_EXTENT='None' """
# lower latitude, upper latitude, lower longitude, upper longitude.
try:
    print("running:", command_string)
    result = subprocess.run(shlex.split(command_string), capture_output=True, text = True, check = True)
except subprocess.CalledProcessError as e:
    # Command failed — capture error output
    print("Command failed with exit code:", e.returncode)
    print(e.stdout)
    print("STDERR:\n", e.stderr)
