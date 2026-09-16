cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../experimental_gtcrn")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(gtcrn_integer_benchmark)
