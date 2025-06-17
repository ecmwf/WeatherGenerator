```
# Make sure you have a recent version of gcc in your environment
# Make sure your python includes cpython (if you do not use uv's python)

# install uv 
# Suggested solution for HPC systems:
%>curl -LsSf https://astral.sh/uv/0.7.0/install.sh | sh
 
# git clone / fork WeatherGenerator repo
%>cd WeatherGenerator
%>uv sync
 
 
%>uv run train
```
