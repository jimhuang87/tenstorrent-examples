cmake_minimum_required(VERSION 3.10)
project(TTAddOne LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 設定 TT_METAL_HOME 環境變數路徑
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "Please set TT_METAL_HOME environment variable to your tt-metal repo path")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

# 包含路徑
include_directories(
    ${TT_METAL_HOME}
    ${TT_METAL_HOME}/tt_metal/include
    ${TT_METAL_HOME}/tt_metal/hw/inc
)

# 尋找庫
link_directories(${TT_METAL_HOME}/build/lib)

add_executable(tt_add_one src/main.cpp)

# 連結 TT Metal 庫
target_link_libraries(tt_add_one 
    tt_metal
    yaml-cpp # 通常需要
    pthread
    dl
)