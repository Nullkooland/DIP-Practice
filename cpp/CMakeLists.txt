cmake_minimum_required(VERSION 3.18)
project(opencv_fun)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -mavx -mavx2 -mfma -msse -msse2 -msse3)

find_package(argparse REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(utils)

add_subdirectory(side_window_filtering)
add_subdirectory(interpolation)
add_subdirectory(orga_no_care)
add_subdirectory(gapi_test)
add_subdirectory(optimization)