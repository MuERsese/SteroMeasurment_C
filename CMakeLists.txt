cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 查找 pkg-config
find_package(PkgConfig REQUIRED)

# 使用 pkg-config 查找 OpenCV
pkg_check_modules(OPENCV REQUIRED opencv4)

# 打印调试信息
message(STATUS "OpenCV include dirs: ${OPENCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OPENCV_LIBRARIES}")

# 包含 OpenCV 的头文件目录
include_directories(${OPENCV_INCLUDE_DIRS})

# 设置源文件
set(SOURCES Measurement.cpp)

# 创建可执行文件
add_executable(Measurement ${SOURCES})

# 链接 OpenCV 库
target_link_libraries(Measurement PRIVATE ${OPENCV_LIBRARIES})