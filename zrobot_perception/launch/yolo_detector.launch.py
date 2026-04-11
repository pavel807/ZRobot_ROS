#!/usr/bin/env python3
"""
YOLO Detector Launch File for zrobot_perception (Python RKNN node)
Can be launched standalone or included from zrobot_bringup.
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def find_config_file():
    """Find zrobot_config.yaml - checks source dir first, then package share"""
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent.parent

    config_in_root = project_root / "zrobot_config.yaml"
    if config_in_root.exists():
        return str(config_in_root)

    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_dir = get_package_share_directory("zrobot_bringup")
        return os.path.join(pkg_dir, "config", "zrobot_config.yaml")
    except Exception:
        pass

    return str(current.parent.parent.parent / "zrobot_config.yaml")


def generate_launch_description():
    config_file = find_config_file()

    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=config_file),
        DeclareLaunchArgument("model_path", default_value="models/yolo26s-rk3588.rknn"),
        DeclareLaunchArgument("camera_id", default_value="1"),
        DeclareLaunchArgument("width", default_value="640"),
        DeclareLaunchArgument("height", default_value="640"),
        DeclareLaunchArgument("fps", default_value="30"),
        DeclareLaunchArgument("obj_thresh", default_value="0.25"),
        DeclareLaunchArgument("nms_thresh", default_value="0.45"),
        DeclareLaunchArgument("num_npu_cores", default_value="3"),
        DeclareLaunchArgument("target_object", default_value="person"),
        DeclareLaunchArgument("enable_tracking", default_value="True"),
        DeclareLaunchArgument("enable_auto_follow", default_value="True"),
        DeclareLaunchArgument("max_linear_speed", default_value="0.3"),
        DeclareLaunchArgument("turn_speed", default_value="0.5"),

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
                    "width": LaunchConfiguration("width"),
                    "height": LaunchConfiguration("height"),
                    "fps": LaunchConfiguration("fps"),
                    "obj_thresh": LaunchConfiguration("obj_thresh"),
                    "nms_thresh": LaunchConfiguration("nms_thresh"),
                    "num_npu_cores": LaunchConfiguration("num_npu_cores"),
                    "target_object": LaunchConfiguration("target_object"),
                    "enable_tracking": LaunchConfiguration("enable_tracking"),
                    "enable_auto_follow": LaunchConfiguration("enable_auto_follow"),
                    "max_linear_speed": LaunchConfiguration("max_linear_speed"),
                    "turn_speed": LaunchConfiguration("turn_speed"),
                },
            ],
        ),
    ])
