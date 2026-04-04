cmake_minimum_required(VERSION 3.20)
project(web_interface)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

install(PROGRAMS
  src/web_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME web_server
)

install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
