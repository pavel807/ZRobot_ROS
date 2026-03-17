cmake_minimum_required(VERSION 3.20)
project(zrobot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Add subdirectories
add_subdirectory(zrobot_perception)
add_subdirectory(zrobot_control)
add_subdirectory(zrobot_interfaces)
add_subdirectory(zrobot_bringup)
add_subdirectory(web_interface)

# Install launch and config files
install(DIRECTORY
  zrobot_bringup/launch
  zrobot_bringup/config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
