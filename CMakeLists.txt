cmake_minimum_required(VERSION 3.10)

project(tennis_detector C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TPU_SDK_PATH)
  message(FATAL_ERROR "Please set TPU_SDK_PATH")
endif()
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

if(NOT DEFINED OPENCV_PATH)
  message(FATAL_ERROR "Please set OPENCV_PATH")
endif()
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

# 包含当前目录的头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 编译目标
add_executable(tennis main.cpp tennis.cpp)

target_link_libraries(tennis
    cviruntime cvikernel
    opencv_core opencv_imgcodecs opencv_imgproc
    dl pthread z
    -Wl,-Bstatic -latomic -Wl,-Bdynamic)

install(TARGETS tennis DESTINATION bin)
