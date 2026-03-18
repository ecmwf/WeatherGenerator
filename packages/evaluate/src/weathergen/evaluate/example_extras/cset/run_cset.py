#!/usr/bin/env -S uv run --script
# ruff: noqa: E501
# /// script
# dependencies = [
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
from yaml.loader import SafeLoaderfrom importlib import resources as impresources
from CSET import recipes

inp_file = impresources.files(recipes) / 'surface_field/src/CSET/recipes/surface_fields/surface_structural_similarity_mean.yaml'
with inp_file.open("rt") as f:
    template = f.read()

from weathergen.common.config import _REPO_ROOT

recipe_path = 
with open(recipe_path) as stream:
    recipe = yaml.load(stream, Loader=SafeLoader)

execute_recipe(
    recipe,
    Path(_REPO_ROOT / "plots" / "cset")
)