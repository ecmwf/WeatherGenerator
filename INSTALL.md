```
# Make sure you have a recent version of gcc in your environment
# Make sure your python includes cpython (if you do not use uv's python)

# install uv 
# Suggested solution for HPC systems:
%>mkdir ~/.uv
%>wget https://github.com/astral-sh/uv/releases/download/0.5.28/uv-x86_64-unknown-linux-gnu.tar.gz ~/.uv
%~/.uv>tar xvzf uv-x86_64-unknown-linux-gnu.tar.gz
%~/.uv>ln -s uv-x86_64-unknown-linux-gnu/uv uv
%~/.uv>ln -s uv-x86_64-unknown-linux-gnu/uvx uvx
#Add to ~/.bashrc
export PATH=~/.uv:$PATH
 
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
