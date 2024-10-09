cmake_minimum_required(VERSION 3.24)

project(RealSenseBoxDetection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(RealSenseBoxDetection src/RealSenseBoxDetection.cpp)

find_package(realsense2 REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

add_subdirectory(include/tinyxml2)

include_directories(${realsense2_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} include/tinyxml2 )

target_link_libraries(RealSenseBoxDetection ${realsense2_LIBRARY} ${OpenCV_LIBS} Eigen3::Eigen tinyxml2)

