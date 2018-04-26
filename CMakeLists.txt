cmake_minimum_required(VERSION 3.7)
project(Parallel_Proj)

set(CMAKE_C_STANDARD 11)

set(SOURCE_FILES main.c matrix.h minst.h minst.c matrix.c cnn.h cnn.c)
add_executable(Parallel_Proj ${SOURCE_FILES})