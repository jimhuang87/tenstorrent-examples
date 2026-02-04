export TT_METAL_HOME=/path/to/your/tt-metal/repo
export PYTHONPATH=$TT_METAL_HOME

cd tt_add_one
mkdir build && cd build
cmake ..
make