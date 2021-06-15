# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
# プロジェクト名と使用する言語を設定
project(test_cmake CXX)

add_executable(main base_program.cpp src/qubo_greedy.cpp)
target_include_directories(main PUBLIC ./include)

add_executable(test test.cpp src/model.cpp src/qubo_greedy.cpp)
target_include_directories(test PUBLIC ./include)
