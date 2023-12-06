cmake_minimum_required(VERSION 3.15.4)

# set the compiler result to be in ./build/bin
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
# enable compile commands to help VSCode intellisense
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
add_compile_options(-std=c++11)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project(renderer LANGUAGES CUDA CXX)

# set CUDA related variables
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


add_custom_target(copy-data ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_SOURCE_DIR}/src/data
${CMAKE_BINARY_DIR}/bin/data
)

# append all *.cpp files in ./src/ recursively to SOURCES env var
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

# Find necessary packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)

# Include OpenCV directories
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories("/packages/cuda/11.2.0/include")

add_executable(out ${SOURCES})
target_link_libraries( out ${OPENCV_LIBRARIES} )

# specify where to find header files to be included in compilation
target_include_directories(out PRIVATE ${PROJECT_SOURCE_DIR}/include/lib ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/lib/glm)

set_target_properties(out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
