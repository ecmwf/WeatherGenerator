```
# Make sure you have a recent version of gcc in your environment
# Make sure your python includes cpython (if you do not use uv's python)

# install uv 
# Suggested solution for HPC systems:
%>curl -LsSf https://astral.sh/uv/0.6.5/install.sh | sh
 
# git clone / fork WeatherGenerator repo
%>cd WeatherGenerator
%>uv sync
 
# install flash_attn which is not uv compatible
# (this step might take very long, it is faster when ninja is available but then
# MAX_JOBS=4 should be set as environment variable; the step might need to be run
# on a compute node to ensure the correct CUDA environment is available) 
%>uv pip install torch
%>uv pip install flash_attn --no-build-isolation
 
%>uv run train
```
