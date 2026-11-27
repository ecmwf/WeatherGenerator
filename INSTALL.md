```
# Make sure you have a recent version of gcc in your environment
# Make sure your python includes cpython (if you do not use uv's python)

# install uv 
# The project needs exactly uv 0.13.7 (required-version in pyproject.toml).
# Suggested solution for HPC systems:
%>curl -LsSf https://astral.sh/uv/0.13.7/install.sh | sh
# If you already have another uv version from this installer:
%>uv self update 0.13.7
# A uv from Homebrew cannot switch to a chosen version: use the installer
# above and make sure its uv (~/.local/bin) comes first in your PATH.
 
# git clone / fork WeatherGenerator repo
%>cd WeatherGenerator
%>./scripts/actions.sh sync
 
 
%>uv run train
```
