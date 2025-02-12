# Make sure you have a recent version of gcc in your environment 
# Make sure your python includes cpython

install uv
 
[Suggested solution for HPC systems:
%>mkdir ~/.uv
%>wget https://github.com/astral-sh/uv/releases/download/0.5.28/uv-x86_64-unknown-linux-gnu.tar.gz ~/.uv
%~/.uv>tar xvzf uv-x86_64-unknown-linux-gnu.tar.gz
%~/.uv>ln -s uv-x86_64-unknown-linux-gnu/uv uv
%~/.uv>ln -s uv-x86_64-unknown-linux-gnu/uvx uvx
#Add to ~/.bashrc
export PATH=~/.uv:$PATH
]
 
# git clone / fork WeatherGenerator repo
%>cd WeatherGenerator
%>uv sync
 
# install flash_attn which is not uv compatible
%>uv pip install torch
%>uv pip install flash_attn --no-build-isolation
 
%>uv run train
