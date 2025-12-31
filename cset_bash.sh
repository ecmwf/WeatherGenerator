export GRAPHVIZ_DIR="$HOME/.local/graphviz-12.2.1"
export CFLAGS="-I$GRAPHVIZ_DIR/include"
export LDFLAGS="-L$GRAPHVIZ_DIR/lib"
export PKG_CONFIG_PATH="$GRAPHVIZ_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"

uv pip install --no-binary :all: pygraphviz


export PREFIX="$HOME/.local/udunits-2.2.28"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
export UDUNITS2_XML_PATH="$PREFIX/share/udunits/udunits2.xml"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

uv pip install --no-binary :all: cf-units

#TODO: fix that usr/local was made too
export LIBMO_DIR="$HOME/.local/libmo_unpack/usr/local"
export LD_LIBRARY_PATH="$LIBMO_DIR/lib:$LD_LIBRARY_PATH"
export CFLAGS="-I$LIBMO_DIR/include"
export LDFLAGS="-L$LIBMO_DIR/lib"
export PKG_CONFIG_PATH="$LIBMO_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"

uv pip install --no-binary :all: git+https://github.com/SciTools/mo_pack.git
uv pip install numpy==1.26.4
uv pip install CSET==25.12.1

echo $LD_LIBRARY_PATH