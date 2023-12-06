cmake_minimum_required(VERSION 3.15.4)

# set the compiler result to be in ./build/bin
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
# enable compile commands to help VSCode intellisense
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
add_compile_options(-std=c++11)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project(renderer)

add_custom_target(copy-data ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_SOURCE_DIR}/src/data
${CMAKE_BINARY_DIR}/bin/data
)

# append all *.cpp files in ./src/ recursively to SOURCES env var
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Find necessary packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)

# Include OpenCV directories
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(out ${SOURCES})
target_link_libraries( out ${OPENCV_LIBRARIES} )

# specify where to find header files to be included in compilation
target_include_directories(out PRIVATE ${PROJECT_SOURCE_DIR}/include)
