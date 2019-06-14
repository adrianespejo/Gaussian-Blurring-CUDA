cmake_minimum_required(VERSION 3.5.1)   # CMake version check
project(Gaussian-Blurring-CUDA)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard

set(SOURCE_FILES main.cpp main_v2.cpp)            # Add main.cpp file of project root directory as source file
add_executable(main.exe main.cpp)       # Add executable target with source files listed in SOURCE_FILES variable
add_executable(main_v2.exe main_v2.cpp)
