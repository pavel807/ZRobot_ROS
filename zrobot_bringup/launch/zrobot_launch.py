#!/usr/bin/env python3
"""
ZRobot Bringup Launch File
Automatically finds zrobot_config.yaml in project root
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable, UnsetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def find_config_file():
    """Find zrobot_config.yaml - checks source dir first, then package share"""
    # Try source workspace first
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent.parent

    config_in_root = project_root / "zrobot_config.yaml"
    if config_in_root.exists():
        return str(config_in_root)

    # Fallback to package share (installed)
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_dir = get_package_share_directory("zrobot_bringup")
        return os.path.join(pkg_dir, "config", "zrobot_config.yaml")
    except:
        pass

    # Last resort: relative from this file
    return str(current.parent.parent.parent / "zrobot_config.yaml")


def generate_launch_description():
    config_file = find_config_file()

    return LaunchDescription(
        [
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            UnsetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE"),
            DeclareLaunchArgument("config_file", default_value=config_file),
            DeclareLaunchArgument("model_path", default_value="models/yolo26s-rk3588.rknn"),
            DeclareLaunchArgument("camera_id", default_value="1"),
            DeclareLaunchArgument("conf_threshold", default_value="0.30"),
            DeclareLaunchArgument("target_object", default_value="person"),
            DeclareLaunchArgument("uart_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("baud_rate", default_value="115200"),
            DeclareLaunchArgument("max_speed", default_value="245"),
            DeclareLaunchArgument("min_speed", default_value="165"),
            DeclareLaunchArgument("max_linear_speed", default_value="0.3"),
            DeclareLaunchArgument("turn_speed", default_value="0.5"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("min_safe_distance", default_value="0.3"),
            DeclareLaunchArgument("web_port", default_value="8080"),
            Node(
                package="zrobot_perception",
                executable="yolo_detector_node",
                name="yolo_detector",
                output="screen",
                emulate_tty=True,
                parameters=[
                    config_file,
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "camera_id": LaunchConfiguration("camera_id"),
                        "width": 640,
                        "height": 640,
                        "fps": 30,
                        "obj_thresh": LaunchConfiguration("conf_threshold"),
                        "nms_thresh": 0.45,
                        "num_npu_cores": 3,
                        "target_object": LaunchConfiguration("target_object"),
                        "enable_tracking": True,
                        "enable_auto_follow": True,
                        "max_linear_speed": LaunchConfiguration("max_linear_speed"),
                        "turn_speed": LaunchConfiguration("turn_speed"),
                    },
                ],
            ),
            Node(
                package="zrobot_control",
                executable="motor_controller_node",
                name="motor_controller",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "uart_port": LaunchConfiguration("uart_port"),
                        "baud_rate": LaunchConfiguration("baud_rate"),
                        "max_speed": LaunchConfiguration("max_speed"),
                        "min_speed": LaunchConfiguration("min_speed"),
                    },
                ],
            ),
            Node(
                package="web_interface",
                executable="web_server",
                name="web_interface",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "host": "0.0.0.0",
                        "port": LaunchConfiguration("web_port"),
                    },
                ],
            ),
            Node(
                package="ldlidar_stl_ros2",
                executable="ldlidar_stl_ros2_node",
                name="ldlidar",
                output="screen",
                respawn=False,
                respawn_delay=2.0,
                parameters=[
                    config_file,
                    {
                        "product_name": "LDLiDAR_LD19",
                        "topic_name": "scan",
                        "port_name": LaunchConfiguration("lidar_port"),
                        "port_baudrate": 230400,
                        "frame_id": "base_laser",
                        "laser_scan_dir": True,
                        "enable_angle_crop_func": False,
                    },
                ],
            ),
            Node(
                package="zrobot_obstacle_avoidance",
                executable="obstacle_avoidance_node",
                name="obstacle_avoidance",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "enabled": True,
                        "min_safe_distance": LaunchConfiguration("min_safe_distance"),
                    },
                ],
            ),
        ]
    )
