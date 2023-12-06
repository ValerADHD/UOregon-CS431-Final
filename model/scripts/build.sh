module load cmake/3.15.4
pushd ..
mkdir build
pushd build
cmake .. && make
popd
popd
