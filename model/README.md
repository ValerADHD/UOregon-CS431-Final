# Project Build Instructions

This directory serves as the base for a CMake project. The instructions provided below will guide you on how to build and run the project.

## Building the Project 

1. Ensure you have loaded the correct version of CMake. If you don't have it loaded, you can do so by running:

   ```bash
   module load cmake/3.15.4
   ```

2. Generate the Makefile and build the project. You can achieve this by running the following commands:

   ```bash
   cmake -S . -B ./build
   pushd build && make && popd
   ```

   Alternatively, you can generate the Makefile in a new `build` directory and then proceed with the build:

   ```bash
   mkdir build && pushd build
   cmake .. && make
   popd
   ```
These commands create a `build` directory that contains the `Makefile`, compiles the source files, and creates the binary files under the `./build/bin` path.

## Downloading and Preparing the Data

Once the binaries have been created, you can download and unpack the required data as follows:

```bash
pushd build/bin/data
./fetch_training.sh
popd
```

## Running the Program

The final executable named `out` is generated in the `./build/bin` directory. You can run this executable on any one of the following datasets: drjohnson, playroom, train, truck. Here's an example of how to run the program on the 'drjohnson' dataset:

```bash
./out drjohnson
```

Repeat the above step by replacing `drjohnson` with other dataset names (i.e., `playroom`, `train`, `truck`) to run the program on those datasets.
