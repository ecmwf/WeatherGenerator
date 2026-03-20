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
from weathergen.common.config import _REPO_ROOT

inp_file = impresources.files(recipes) / 'surface_fields/surface_structural_similarity_mean.yaml'

replacements = {
    "$INPUT_PATHS": '["/users/sowens/a8ux1pk2_1/prediction_2022-10-01T00_a8ux1pk2.nc", "/users/sowens/a8ux1pk2_1/prediction_2022-10-01T06_a8ux1pk2.nc"]',
    "$VARNAME": "air_pressure_at_mean_sea_level",
    "$LEVEL_TYPE": "pressure",
    "$BASE_MODEL": "target",
    "$OTHER_MODEL": "prediction",
    "$METHOD": "mean",
    "$SUBAREA_TYPE": "gridcells",
    "$SUBAREA_EXTENT": "[55, -45, 70, 0]",
}

with open(inp_file, "r", encoding="utf-8-sig") as f:
    string = f.read()

for old, new in replacements.items():
    string = string.replace(old, new)

recipe = yaml.load(string, Loader=SafeLoader)
print(recipe)
execute_recipe(
    recipe,
    Path(_REPO_ROOT / "plots" / "cset"),
)