module load cmake/3.15.4
pushd ..
mkdir build
pushd build
cmake .. && make
pushd bin/data
if [[ ! -d "drjohnson" ]] && [[ ! -d "playroom" ]] && [[ ! -d "train" ]] && [[ ! -d "truck" ]]; then
    ./fetch_training.sh
fi
popd
popd
popd
