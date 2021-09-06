cmake_minimum_required (VERSION 2.8)
project (yolocv)

set(DARKNET_DIR ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
include_directories (${PROJECT_SOURCE_DIR}/include)
link_directories(${DARKNET_DIR})

message(STATUS "DARKNET_DIR=" ${DARKNET_DIR})

add_executable (mask_detection src/main.cpp src/utils/utils.cpp src/object_detection/ObjectDetection.cpp src/camera/Camera.cpp)
target_link_libraries (mask_detection ${OpenCV_LIBRARIES} darknet)
