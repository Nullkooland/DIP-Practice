cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -mavx -mavx2 -mfma -msse -msse2 -msse3)

project(opencv_fun)
find_package(cxxopts REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(side_window_filtering)
add_subdirectory(interpolation)