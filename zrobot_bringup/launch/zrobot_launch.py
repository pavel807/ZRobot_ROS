#!/usr/bin/env python3
"""
ZRobot Bringup Launch File
Starts all ZRobot nodes: perception, control, and web interface
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/yolov8s.rknn',
        description='Path to YOLO RKNN model'
    )

    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='1',
        description='Camera device ID'
    )

    uart_port_arg = DeclareLaunchArgument(
        'uart_port',
        default_value='/dev/ttyACM0',
        description='UART port for Arduino'
    )

    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.3',
        description='Confidence threshold for detection'
    )

    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='person',
        description='Target object to track'
    )

    # Load config file
    config_file = PathJoinSubstitution([
        FindPackageShare('zrobot_bringup'),
        'config',
        'zrobot_config.yaml'
    ])

    # YOLO Detector Node
    yolo_detector_node = Node(
        package='zrobot_perception',
        executable='yolo_detector_node',
        name='yolo_detector',
        output='screen',
        parameters=[
            config_file,
            {
                'model_path': LaunchConfiguration('model_path'),
                'camera_id': LaunchConfiguration('camera_id'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'target_object': LaunchConfiguration('target_object'),
            }
        ]
    )

    # Motor Controller Node
    motor_controller_node = Node(
        package='zrobot_control',
        executable='motor_controller_node',
        name='motor_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'uart_port': LaunchConfiguration('uart_port'),
            }
        ]
    )

    # Web Interface Node (Python)
    web_interface_node = Node(
        package='web_interface',
        executable='web_server',
        name='web_interface',
        output='screen',
        parameters=[{
            'host': '0.0.0.0',
            'port': 8080,
        }]
    )

    return LaunchDescription([
        model_path_arg,
        camera_id_arg,
        uart_port_arg,
        conf_threshold_arg,
        target_object_arg,
        yolo_detector_node,
        motor_controller_node,
        web_interface_node,
    ])
