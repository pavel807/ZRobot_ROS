cmake_minimum_required(VERSION 3.20)
project(zrobot_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(
  PROGRAMS
  launch/zrobot_launch.py
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
