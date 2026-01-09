from pathlib import Path
import CSET.operators
import yaml
from yaml.loader import SafeLoader
import numpy

print(numpy.__version__)

recipe_path = "/users/sowens/WeatherGenerator/test_CSETinput/air_temperature_spatial_plot.yaml"
with open(recipe_path) as stream:
    recipe = yaml.load(stream, Loader=SafeLoader)

CSET.operators.execute_recipe(
    recipe,
    #Path("/users/sowens/WeatherGenerator/test_CSETinput/air_temp.nc"),
    Path("/users/sowens/WeatherGenerator/test_CSETinput")
)