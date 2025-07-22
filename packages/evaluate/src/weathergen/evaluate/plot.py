#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///


import logging

import plotly.express as px

from weathergen.common import common_function


def plot():
    print(common_function())


if __name__ == "__main__":
    # Set up logging to display debug messages
    logging.basicConfig(level=logging.DEBUG)
    plot()
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x="year", y="pop")
    fig.write_image("fig1.png")
