import json
import logging
import os

import plotly.express as px
import polars as pl
import streamlit as st

_logger = logging.getLogger(__name__)
# List all the json files in ../stac/json:

# Find the current absolute location of this file
current_file_path = os.path.abspath(__file__)
_logger.info(f"Current file path: {current_file_path}")
# Get the directory:
current_dir = os.path.dirname(current_file_path)

stac_dir = os.path.abspath(os.path.join(current_dir, "../stac/jsons"))
_logger.info(f"STAC JSON directory: {stac_dir}")

json_files = sorted([f for f in os.listdir(stac_dir) if f.endswith(".json")])


stats = []
for json_file in json_files:
    with open(os.path.join(stac_dir, json_file)) as f:
        data = json.load(f)
        d_id = data.get("id")
        if "properties" not in data:
            continue
        name = data["properties"].get("name", "No title")
        data_stats = {}
        for fname, fprop in data.get("assets", {}).items():
            inodes = int(fprop.get("inodes", "0").replace(".", "").replace(",", ""))
            size = str(fprop.get("size", "0")).lower().replace(",", ".")
            # Only keep numbers or dots:
            size_ = float("".join([c for c in size if c.isdigit() or c == "."]))
            if "tb" in size:
                size_ *= 1024**4
            elif "gb" in size:
                size_ *= 1024**3
            elif "mb" in size:
                size_ *= 1024**2

            locations = list(fprop.get("locations", []))
            data_stats[fname] = {"inodes": inodes, "locations": locations, "size": size_}
            for loc in locations:
                stats.append(
                    {
                        "id": d_id,
                        "name": name,
                        "file": fname,
                        "location": loc,
                        "inodes": inodes,
                        "size": size_,
                    }
                )

        #
        # st.write(f"Data from {json_file}:", name, data_stats)

stats_df = pl.DataFrame(stats)

st.markdown("""
# INode counts

The number of inodes on each HPC
""")

st.plotly_chart(px.treemap(stats_df, path=["location", "name"], values="inodes"))

st.markdown(""" Duplication by HPC """)

st.plotly_chart(px.treemap(stats_df, path=["name", "location", "file"], values="inodes"))


st.markdown("""
# File sizes

The size of files on each HPC.
""")

st.plotly_chart(px.treemap(stats_df, path=["location", "name"], values="size"))

st.markdown("## Detailed stats")

st.write("JSON files:", json_files)
st.dataframe(stats_df)
