# Meta-package — all sub-packages are built independently by colcon

cmake_minimum_required(VERSION 3.20)
project(zrobot)

find_package(ament_cmake REQUIRED)

ament_package()