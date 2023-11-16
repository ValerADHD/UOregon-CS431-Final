cmake_minimum_required(VERSION 3.15.4)

# set the compiler result to be in ./build/bin
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
# enable compile commands to help VSCode intellisense
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

project(renderer)

# on build, copy the ./src/resources directory to ./bin/resources
# (allows you to keep relative pathnames wrt resource files)
add_custom_target(copy-files ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory
${CMAKE_SOURCE_DIR}/src/resources
${CMAKE_BINARY_DIR}/bin/resources
)

# append all *.cpp files in ./src/ recursively to SOURCES env var
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(out ${SOURCES})

# specify where to find header files to be included in compilation
target_include_directories(out PRIVATE ${PROJECT_SOURCE_DIR}/include)
