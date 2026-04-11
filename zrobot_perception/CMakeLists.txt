cmake_minimum_required(VERSION 3.20)
project(zrobot_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  src/${PROJECT_NAME}/yolo_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME yolo_detector_node
)

# Install launch, config, and other resource directories
install(DIRECTORY
  launch
  config
  resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
