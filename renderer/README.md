Currently this directory is just a basic CMake project to be built upon.

To build: 
```
module load cmake/3.15.4
cmake -S . -B ./build
pushd build && make && popd
```

or

```
module load cmake/3.15.4
mkdir build && pushd build
cmake .. && make
popd
```

This will create a new directory `build` that contains the `Makefile` and builds it. The executable can be found in `./build/bin`
