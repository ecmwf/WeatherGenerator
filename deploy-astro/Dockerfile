FROM astrocrpublic.azurecr.io/runtime:3.0-14

# Install your workspace packages from pre-built wheels
COPY --chown=astro:0 include/wheels/ /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels/