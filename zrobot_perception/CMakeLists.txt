cmake_minimum_required(VERSION 3.20)
project(zrobot_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# RKNN API path (adjust for your platform)
set(RKNN_API_PATH "/usr/lib" CACHE PATH "Path to RKNN API library")
set(RKNN_INCLUDE_PATH "/usr/include" CACHE PATH "Path to RKNN headers")

link_directories(/usr/lib)

include_directories(
  include
  /opt/ros/jazzy/include
  ${RKNN_INCLUDE_PATH}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(yolo_detector_node
  src/yolo_detector_node.cpp
  src/yolo_detector.cpp
  src/object_search.cpp
  src/colors.cpp
)

ament_target_dependencies(yolo_detector_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  cv_bridge
)

target_link_libraries(yolo_detector_node
  ${OpenCV_LIBS}
  rknnrt
)

install(TARGETS
  yolo_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
